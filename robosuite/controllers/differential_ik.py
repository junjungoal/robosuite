import os
from os.path import join as pjoin
import copy

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers.joint_vel import JointVelocityController
from robosuite.utils.control_utils import *

class DifferentialInverseKinematicsController(JointVelocityController):
    """
    Controller for controlling robot arm via inverse kinematics. Allows position and orientation control of the
    robot's end effector.

    Inverse kinematics solving is handled by pybullet.

    NOTE: Control input actions are assumed to be relative to the current position / orientation of the end effector
    and are taken as the array (x_dpos, y_dpos, z_dpos, x_rot, y_rot, z_rot).

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        robot_name (str): Name of robot being controlled. Can be {"Sawyer", "Panda", or "Baxter"}

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        eef_rot_offset (4-array): Quaternion (x,y,z,w) representing rotational offset between the final
            robot arm link coordinate system and the end effector coordinate system (i.e: the gripper)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        ik_pos_limit (float): Limit (meters) above which the magnitude of a given action's
            positional inputs will be clipped

        ik_ori_limit (float): Limit (radians) above which the magnitude of a given action's
            orientation inputs will be clipped

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current state to
            the goal state during each timestep between inputted actions

        converge_steps (int): How many iterations to run the pybullet inverse kinematics solver to converge to a
            solution

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Unsupported robot]
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        robot_name,
        actuator_range,
        eef_rot_offset,
        eef_body_name='robot0_right_hand',
        policy_freq=20,
        load_urdf=True,
        ik_pos_limit=None,
        ik_ori_limit=None,
        interpolator_pos=None,
        interpolator_ori=None,
        converge_steps=5,
        use_ori=True,
        use_z_ori=False,
        **kwargs,
    ):

        # Run sueprclass inits
        super().__init__(
            sim=sim,
            eef_name=eef_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            input_max=1,
            input_min=-1,
            output_max=1,
            output_min=-1,
            kv=0.25,
            policy_freq=policy_freq,
            velocity_limits=[-1, 1],
            **kwargs,
        )


        # Initialize ik-specific attributes
        self.robot_name = robot_name  # Name of robot (e.g.: "Panda", "Sawyer", etc.)
        self.eef_body_name = eef_body_name

        # IK method
        self._ik_params = {"lambda_val": 0.1}

        # Override underlying control dim
        self.use_ori = use_ori
        self.use_z_ori = use_z_ori
        self.control_dim = 6 if self.use_ori else 3
        if self.use_z_ori:
            self.control_dim += 1
        self.name_suffix = "Pose" if self.use_ori else "Position"

        # Rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.eef_rot_offset = eef_rot_offset
        self.rotation_offset = None
        self.rest_poses = None

        # Set the reference robot target pos / orientation (to prevent drift / weird ik numerical behavior over time)
        self.target_pos = self.ee_pos
        self.target_orn = T.mat2quat(self.ee_ori_mat)

        # Interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # Interpolator-related attributes
        self.ori_ref = None
        self.relative_ori = None

        # Values for initializing pybullet env
        self.ik_robot = None
        self.robot_urdf = None
        self.num_bullet_joints = None
        self.bullet_ee_idx = None
        self.bullet_joint_indexes = None  # Useful for splitting right and left hand indexes when controlling bimanual
        self.ik_command_indexes = None  # Relevant indices from ik loop; useful for splitting bimanual left / right
        self.ik_robot_target_pos_offset = None
        self.base_orn_offset_inv = None  # inverse orientation offset from pybullet base to world
        self.converge_steps = converge_steps

        # Set ik limits and override internal min / max
        self.ik_pos_limit = ik_pos_limit
        self.ik_ori_limit = ik_ori_limit

        # Target pos and ori
        self.ik_robot_target_pos = None
        self.ik_robot_target_orn = None  # note: this currently isn't being used at all

        # Commanded pos and resulting commanded vel
        self.commanded_joint_positions = None
        self.commanded_joint_velocities = None
        self.ee_min_limit = np.array([0.15, -0.4, 0.83])
        self.ee_max_limit = np.array([0.7, 0.4, 1.3])


    def get_control(self, dpos=None, rotation=None, z_rot=None, update_targets=False):
        """
        Returns joint velocities to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint velocities will be computed based
        on the previously recorded target.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            np.array: a flat array of joint velocity commands to apply to try and achieve the desired input control.
        """

        # Compute new target joint positions if arguments are provided
        self.commanded_joint_positions = np.array(
            self.joint_positions_for_eef_command(dpos, rotation, update_targets)
        )


        if z_rot is not None:
            # self.commanded_joint_positions[-1] += z_rot
            self.commanded_joint_positions[-1] = self.joint_pos[-1]
            self.commanded_joint_positions[-1] += z_rot
        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(self.joint_dim)
        deltas = self._get_current_error(self.joint_pos, self.commanded_joint_positions)
        for i, delta in enumerate(deltas):
            velocities[i] = -10.0 * delta

        self.commanded_joint_velocities = velocities
        return velocities

    def inverse_kinematics(self, target_position, target_orientation):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position (3-tuple): desired position
            target_orientation (4-tuple): desired orientation quaternion

        Returns:
            list: list of size @num_joints corresponding to the joint angle solution.
        """
        raise NotImplementedError

    def _compute_delta_dof_pos(self, delta_pose, jacobian):
        """Computes the change in dos-position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired end-effector
        position.

        Args:
            delta_pose : The desired delta pose in shape [N, 3 or 6].
            jacobian : The geometric jacobian matrix in shape [N, 3 or 6, num-dof]

        Returns:
            np.array: The desired delta in joint space.
        """
        # parameters
        lambda_val = self._ik_params["lambda_val"]
        # computation
        jacobian_T = np.transpose(jacobian, (0, 2, 1))
        lambda_matrix = (lambda_val**2) * np.eye(jacobian.shape[1])
        delta_dof_pos = jacobian_T @ np.linalg.inv(jacobian @ jacobian_T + lambda_matrix) @ np.expand_dims(delta_pose, axis=-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos[0]

    def joint_positions_for_eef_command(self, dpos, rotation=None, desired_rotation=None, update_targets=False):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            list: A list of size @num_joints corresponding to the target joint angles.
        """

        jacobian = np.concatenate([self.sim.data.get_body_jacp(self.eef_body_name),
                                   self.sim.data.get_body_jacr(self.eef_body_name)])[:, self.joint_index]

        quat = T.mat2quat(rotation)
        axisangle = T.quat2axisangle(quat)
        delta_pose = np.concatenate([dpos, axisangle])
        delta_joint_position = self._compute_delta_dof_pos(delta_pose[None], jacobian[None])
        return self.joint_pos + delta_joint_position

    def set_goal(self, delta, set_ik=None):
        """
        Sets the internal goal state of this controller based on @delta

        Note that this controller wraps a VelocityController, and so determines the desired velocities
        to achieve the inputted pose, and sets its internal setpoint in terms of joint velocities

        TODO: Add feature so that using @set_ik automatically sets the target values to these absolute values

        Args:
            delta (Iterable): Desired relative position / orientation goal state
            set_ik (Iterable): If set, overrides @delta and sets the desired global position / orientation goal state
        """
        # Update state
        self.update()


        # Run ik prepropressing to convert pos, quat ori to desired velocities
        requested_control = self._make_input(delta)
        dpos = requested_control['dpos']
        z_rot = None
        rotation = requested_control['rotation']

        if self.use_z_ori:
            z_rot = requested_control['z_rot']

        # Compute desired velocities to achieve eef pos / ori
        velocities = self.get_control(dpos, rotation=rotation, z_rot=z_rot, update_targets=True)

        # Set the goal velocities for the underlying velocity controller
        super().set_goal(velocities)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        # Update interpolated action if necessary
        desired_pos = None
        rotation = None
        update_velocity_goal = False

        # Update interpolated goals if active
        desired_pos = self.target_pos
        rotation = T.quat2mat(self.target_orn)

        # Only update the velocity goals if we're interpolating
        if update_velocity_goal:
            velocities = self.get_control(dpos=(desired_pos - self.ee_pos), rotation=rotation)
            super().set_goal(velocities)

        # Run controller with given action
        return super().run_controller()

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # Then, update the rest pose from the initial joints
        self.rest_poses = list(self.initial_joint)

    def reset_goal(self):
        """
        Resets the goal to the current pose of the robot
        """
        self.target_pos = self.ee_pos
        self.target_orn = T.mat2quat(self.ee_ori_mat)

    def _make_input(self, action):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat. Additionally clips @action as well

        Args:
            action (np.array) should have form: [dx, dy, dz, ax, ay, az] (orientation in
                scaled axis-angle form)
            old_quat (np.array) the old target quaternion that will be updated with the relative change in @action
        """
        # Clip action appropriately
        # Get requested delta inputs if we're using interpolators
        dpos = action[:3] * self.ik_pos_limit
        rotation = action[3:7] * self.ik_ori_limit

        self.target_pos = self.ee_pos + dpos
        self.target_pos = np.minimum(np.maximum(self.target_pos, self.ee_min_limit), self.ee_max_limit)
        dpos = self.target_pos - self.ee_pos
        current_quat = T.mat2quat(self.ee_ori_mat)

        if self.use_ori:
            # Update reference targets
            self.target_orn = T.quat_multiply(current_quat, rotation)
            return {"dpos": dpos, "rotation": T.quat2mat(rotation)}
        else:
            set_ori = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            self.target_orn = T.mat2quat(set_ori)
            # rotation = T.quat_multiply(current_quat, T.quat_inverse(self.target_orn))
            rotation = T.quat_multiply(self.target_orn, T.quat_inverse(current_quat))
            output = {"dpos": dpos, "rotation": T.quat2mat(rotation)}
            if self.use_z_ori:
                output['z_rot'] = rotation[0]
            return output



    @staticmethod
    def _get_current_error(current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current (np.array): the current joint positions
            set_point (np.array): the joint positions that are desired as a numpy array

        Returns:
            np.array: the current error in the joint positions
        """
        error = current - set_point
        return error

    @property
    def control_limits(self):
        """
        The limits over this controller's action space, as specified by self.ik_pos_limit and self.ik_ori_limit
        and overriding the superclass method

        Returns:
            2-tuple:

                - (np.array) minimum control values
                - (np.array) maximum control values
        """
        if self.use_ori:
            max_limit = np.concatenate([self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(3)])
        elif self.use_z_ori:
            max_limit = np.concatenate([self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(1)])
        else:
            max_limit = self.ik_pos_limit * np.ones(3)
        return -max_limit, max_limit

    @property
    def name(self):
        return "DIFFERENTIAL_IK_" + self.name_suffix
