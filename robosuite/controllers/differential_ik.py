import os
from os.path import join as pjoin

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
        self.reference_target_pos = self.ee_pos
        self.reference_target_orn = T.mat2quat(self.ee_ori_mat)
        import pdb
        pdb.set_trace()

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

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 0.3


    def get_control(self, dpos=None, rotation=None, desired_rotation=None, z_rot=None, update_targets=False):
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
        if (dpos is not None) and (rotation is not None):
            self.commanded_joint_positions = np.array(
                self.joint_positions_for_eef_command(dpos, rotation, update_targets)
            )
        else:
            self.commanded_joint_positions = np.array(
                self.joint_positions_for_eef_command(dpos, desired_rotation=desired_rotation, update_targets=update_targets)
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

        import pdb
        pdb.set_trace()
        # Calculate the rotation
        # This equals: inv base offset * eef * offset accounting for deviation between mujoco eef and pybullet eef
        if rotation is not None:
            rotation = self.base_orn_offset_inv @ self.ee_ori_mat @ rotation @ self.rotation_offset[:3, :3]
        else:
            rotation = self.base_orn_offset_inv @ desired_rotation @ self.rotation_offset[:3, :3]

        # Determine targets based on whether we're using interpolator(s) or not
        if self.interpolator_pos or self.interpolator_ori:
            targets = (self.ee_pos + dpos + self.ik_robot_target_pos_offset, T.mat2quat(rotation))
        else:
            targets = (self.ik_robot_target_pos + dpos, T.mat2quat(rotation))

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets = self.bullet_base_pose_to_world_pose(targets)

        # Update targets if required
        if update_targets:
            # Scale and increment target position
            self.ik_robot_target_pos += dpos

            # Convert the desired rotation into the target orientation quaternion
            self.ik_robot_target_orn = T.mat2quat(rotation)

        # Converge to IK solution
        arm_joint_pos = None
        return arm_joint_pos

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

        # Get requested delta inputs if we're using interpolators
        (dpos, dquat) = self._clip_ik_input(delta[:3], delta[3:7])

        # Set interpolated goals if necessary
        if self.interpolator_pos is not None:
            # Absolute position goal
            self.interpolator_pos.set_goal(dpos * self.user_sensitivity + self.reference_target_pos)

        if self.interpolator_ori is not None:
            # Relative orientation goal
            self.interpolator_ori.set_goal(dquat)  # goal is the relative change in orientation
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        # Run ik prepropressing to convert pos, quat ori to desired velocities
        requested_control = self._make_input(delta, self.reference_target_orn)
        dpos = requested_control['dpos']
        desired_rotation=None
        rotation = None
        z_rot = None
        if self.use_ori:
            rotation = requested_control['rotation']
        else:
            desired_rotation = requested_control['desired_rotation']

        if self.use_z_ori:
            z_rot = requested_control['z_rot']

        # Compute desired velocities to achieve eef pos / ori
        velocities = self.get_control(dpos, rotation=rotation,
                                      desired_rotation=desired_rotation, z_rot=z_rot, update_targets=True)

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
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
            update_velocity_goal = True
        else:
            desired_pos = self.reference_target_pos

        if self.interpolator_ori is not None:
            # Linear case
            if self.interpolator_ori.order == 1:
                # relative orientation based on difference between current ori and ref
                self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
                ori_error = self.interpolator_ori.get_interpolated_goal()
                rotation = T.quat2mat(ori_error)
            else:
                # Nonlinear case not currently supported
                pass
            update_velocity_goal = True
        else:
            rotation = T.quat2mat(self.reference_target_orn)

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
        self.reference_target_pos = self.ee_pos
        self.reference_target_orn = T.mat2quat(self.ee_ori_mat)

    def _clip_ik_input(self, dpos, rotation=None):
        """
        Helper function that clips desired ik input deltas into a valid range.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): relative rotation in scaled axis angle form (ax, ay, az)
                corresponding to the (relative) desired orientation of the end effector.

        Returns:
            2-tuple:

                - (np.array) clipped dpos
                - (np.array) clipped rotation
        """
        # scale input range to desired magnitude
        if dpos.any():
            dpos, _ = T.clip_translation(dpos, self.ik_pos_limit)

        if rotation is not None:
            # Map input to quaternion
            rotation = T.axisangle2quat(rotation)

            # Clip orientation to desired magnitude
            rotation, _ = T.clip_rotation(rotation, self.ik_ori_limit)

            return dpos, rotation
        else:
            return dpos

    def _make_input(self, action, old_quat):
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
        if self.use_ori:
            dpos, rotation = self._clip_ik_input(action[:3], action[3:])
            # Update reference targets
            self.reference_target_orn = T.quat_multiply(old_quat, rotation)
            self.reference_target_pos += dpos * self.user_sensitivity
            return {"dpos": dpos * self.user_sensitivity, "rotation": T.quat2mat(rotation)}
        else:
            dpos = self._clip_ik_input(action[:3])
            set_ori = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            self.reference_target_orn = T.mat2quat(set_ori)
            rotation = T.quat_multiply(old_quat, T.quat_inverse(self.reference_target_orn))
            self.reference_target_pos += dpos * self.user_sensitivity
            output = {"dpos": dpos * self.user_sensitivity, "desired_rotation": set_ori}
            if self.use_z_ori:
                output['z_rot'] = self.ik_ori_limit * action[3]
            # return {"dpos": dpos * self.user_sensitivity, "rotation": T.quat2mat(rotation)}
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
