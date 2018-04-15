import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.models import *


class SawyerStackEnv(SawyerEnv):
    def __init__(self, 
                 gripper='TwoFingerGripper',
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 camera_height=256,
                 camera_width=256,
                 camera_depth=True,
                 visualize_gripper_site=False,
                 **kwargs):
        """
            @table_size, the FULL size of the table 
        """
        # initialize objects of interest
        cubeA = RandomBoxObject(size_min=[0.025, 0.025, 0.03],
                                size_max=[0.05, 0.05, 0.05])
        cubeB = RandomBoxObject(size_min=[0.025, 0.025, 0.03],
                                size_max=[0.05, 0.05, 0.05])
        self.mujoco_objects = OrderedDict([
            ('cubeA', cubeA),
            ('cubeB', cubeB)
        ])

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # settings for camera observation
        self.use_camera_obs = use_camera_obs
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth
        self.visualize_gripper_site = visualize_gripper_site

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(gripper=gripper, **kwargs)

        # information of objects
        self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_site_ids = [self.sim.model.site_name2id(ob_name) for ob_name in self.object_names]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)
        self.model.place_objects()

        self.object_metadata = self.model.object_metadata
        self.n_objects = len(self.object_metadata)

    def _get_reference(self):
        super()._get_reference()
        self.cubeA_body_id = self.sim.model.body_name2id('cubeA')
        self.cubeB_body_id = self.sim.model.body_name2id('cubeB')

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

    def reward(self, action):
        reward = 0
        #TODO(yukez): implementing a stacking reward
        return reward

    def _get_observation(self):
        """
            Adds hand_position, hand_velocity or 
            (current_position, current_velocity, target_velocity) of all targets
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                         width=self.camera_width,
                                         height=self.camera_height,
                                         depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # position and rotation of the first cube
            cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
            cubeA_quat = self.sim.data.body_xquat[self.cubeA_body_id]
            di['cubeA_pos'] = cubeA_pos
            di['cubeA_quat'] = cubeA_quat

            # position and rotation of the second cube
            cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
            cubeB_quat = self.sim.data.body_xquat[self.cubeB_body_id]
            di['cubeB_pos'] = cubeB_pos
            di['cubeB_quat'] = cubeB_quat

            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            di['gripper_to_cubeA'] = gripper_site_pos - cubeA_pos
            di['gripper_to_cubeB'] = gripper_site_pos - cubeB_pos
            di['cubeA_to_cubeB'] = cubeA_pos - cubeB_pos

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.finger_names or \
               self.sim.model.geom_id2name(contact.geom2) in self.finger_names:
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        #TODO(yukez): define termination conditions
        return False

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.visualize_gripper_site:
            # find closest object
            square_dist = lambda x : np.sum(np.square(x - self.sim.data.get_site_xpos('grip_site')))
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[self.object_site_ids] # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba