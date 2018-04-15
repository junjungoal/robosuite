import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.models.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.models.world import MujocoWorldBase
from MujocoManip.models.model_util import *
from MujocoManip.miscellaneous.utils import *

class TableTopTask(MujocoWorldBase):

    """
        Table top manipulation task can be specified 
        by three elements of the environment.
        @mujoco_arena, MJCF robot workspace (e.g., table top)
        @mujoco_robot, MJCF robot model
        @mujoco_objects, a list of MJCF objects of interest
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects):
        super().__init__()
        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)

    def merge_robot(self, mujoco_robot):
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        self.n_objects = len(mujoco_objects)
        self.mujoco_objects = mujoco_objects
        self.objects = [] # xml manifestation
        self.targets = [] # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_full(name=obj_name, site=True)
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.object_metadata.append({
                'object_name': obj_name,
                'object_bottom_offset': obj_mjcf.get_bottom_offset(),
                'object_top_offset': obj_mjcf.get_top_offset(),
                'object_horizontal_radius': obj_mjcf.get_horizontal_radius(),
            })
            self.max_horizontal_radius = max(self.max_horizontal_radius,
                                             obj_mjcf.get_horizontal_radius())

    def place_objects(self):
        """
        Place objects randomly until no more collisions or max iterations hit.
        """
        # Objects
        placed_objects = []
        index = 0
        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for i in range(1000): # 1000 retries
                table_x_half = self.table_size[0] / 2 - horizontal_radius
                table_y_half = self.table_size[1] / 2 - horizontal_radius
                object_x = np.random.uniform(high=table_x_half, low=-table_x_half)
                object_y = np.random.uniform(high=table_y_half, low=-1 * table_y_half)
                # objects cannot overlap
                location_valid = True
                for (x, y, z), r in placed_objects:
                    if np.linalg.norm([object_x - x, object_y - y], 2) <= r + horizontal_radius:
                        location_valid = False
                        break
                if location_valid: # bad luck, reroll
                    pos = self.table_top_offset - bottom_offset + np.array([object_x, object_y, 0])
                    placed_objects.append((pos, horizontal_radius))
                    self.objects[index].set('pos', array_to_string(pos))
                    success = True
                    break
                # location is valid, put the object down
                # quarternions, later we can add random rotation
            if not success:
                raise RandomizationError('Cannot place all objects on the desk')
            index += 1