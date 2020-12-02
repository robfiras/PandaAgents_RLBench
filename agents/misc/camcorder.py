import os
import csv
import numpy as np
from PIL import Image

from rlbench.backend.observation import Observation


class Camcorder:

    def __init__(self, path_to_save, unique_id, format="JPEG"):
        self.id = unique_id
        self.path_to_camcorder = os.path.join(path_to_save, "camcorder")
        self.path_to_camcorder_task_low_dim = os.path.join(path_to_save, "camcorder_task_low_dim")
        self.path_to_csv = os.path.join(self.path_to_camcorder_task_low_dim, "%i_task_low_dim.csv" % self.id)
        # make both directories
        if not os.path.exists(self.path_to_camcorder):
            os.mkdir(self.path_to_camcorder)
        if not os.path.exists(self.path_to_camcorder_task_low_dim):
            os.mkdir(self.path_to_camcorder_task_low_dim)

        self.format = format
        self.counter = 0

    def save(self, obs: Observation, obj_poses=None):

        def reformat(float_image):
            return (float_image * 255).astype(np.uint8)

        def get_all_poses(obj_poses, camera):
            if camera == "left_shoulder_camera":
                return [pose["left_shoulder_camera"] for pose in obj_poses if pose["left_shoulder_camera"] is not None]
            if camera == "right_shoulder_camera":
                return [pose["right_shoulder_camera"] for pose in obj_poses if pose["right_shoulder_camera"] is not None]
            if camera == "front_camera":
                return [pose["front_camera"] for pose in obj_poses if pose["front_camera"] is not None]
            if camera == "wrist_camera":
                return [pose["wrist_camera"] for pose in obj_poses if pose["wrist_camera"] is not None]

        if obs.left_shoulder_rgb is not None:
            left_shoulder_image = Image.fromarray(reformat(obs.left_shoulder_rgb))
            image_name = "%i_%i_left_shoulder_camera.jpg" % (self.id, self.counter)
            image_path = os.path.join(self.path_to_camcorder, image_name)
            left_shoulder_image.save(image_path, format=self.format)
        if obs.right_shoulder_rgb is not None:
            right_shoulder_image = Image.fromarray(reformat(obs.right_shoulder_rgb))
            image_name = "%i_%i_right_shoulder_camera.jpg" % (self.id, self.counter)
            image_path = os.path.join(self.path_to_camcorder, image_name)
            right_shoulder_image.save(image_path, format=self.format)
        if obs.wrist_rgb is not None:
            wrist_image = Image.fromarray(reformat(obs.wrist_rgb))
            image_name = "%i_%i_wrist_camera.jpg" % (self.id, self.counter)
            image_path = os.path.join(self.path_to_camcorder, image_name)
            wrist_image.save(image_path, format=self.format)
        if obs.front_rgb is not None:
            front_image = Image.fromarray(reformat(obs.front_rgb))
            image_name = "%i_%i_front_camera.jpg" % (self.id, self.counter)
            image_path = os.path.join(self.path_to_camcorder, image_name)
            front_image.save(image_path, format=self.format)
        if obj_poses:
            with open(self.path_to_csv, mode='a') as label_file:
                writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for camera in ["left_shoulder_camera", "right_shoulder_camera", "front_camera", "wrist_camera"]:
                    poses = get_all_poses(obj_poses, camera)
                    for pose in poses:
                        writer.writerow(["%i_%i_%s" % (self.id, self.counter, camera)] + pose.tolist())
        self.counter += 1


