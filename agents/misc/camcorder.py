import os
import csv
import numpy as np
from PIL import Image
import yaml

from rlbench.backend.observation import Observation
from pyrep.objects.object import Object


class Camcorder:

    def __init__(self, path_to_save, unique_id, format="JPEG"):
        self.id = unique_id
        self.path_to_camcorder = os.path.join(path_to_save, "camcorder")
        self.path_to_camcorder_task_low_dim = os.path.join(path_to_save, "camcorder_task_low_dim")
        self.path_to_csv = os.path.join(self.path_to_camcorder_task_low_dim, "%i_task_low_dim.csv" % self.id)
        self.path_to_masks = os.path.join(path_to_save, "masks")
        self.path_to_mask_labels = os.path.join(path_to_save, "mask_labels")
        self.path_to_mask_label_file = os.path.join(self.path_to_mask_labels, "%i_mask_labels.yaml" % self.id)
        self.mask_label_dict = {}
        # make both directories
        if not os.path.exists(self.path_to_camcorder) and unique_id == 0:
            os.mkdir(self.path_to_camcorder)
        if not os.path.exists(self.path_to_camcorder_task_low_dim) and unique_id == 0:
            os.mkdir(self.path_to_camcorder_task_low_dim)
        if not os.path.exists(self.path_to_masks) and unique_id == 0:
            os.mkdir(self.path_to_masks)
        if not os.path.exists(self.path_to_mask_labels) and unique_id == 0:
            os.mkdir(self.path_to_mask_labels)

        self.format = format
        self.counter = 0

    def save(self, obs: Observation, robot_visuals=None, graspable_objs=None):

        def reformat(float_image):
            return (float_image * 255).astype(np.uint8)

        # id used for RGB, masks and their respective labels
        curr_id = "%i_%i" % (self.id, self.counter)

        # --- RGB ---
        rgb_available = False
        if obs.left_shoulder_rgb is not None:
            left_shoulder_image = Image.fromarray(reformat(obs.left_shoulder_rgb))
            image_name = "%s_left_shoulder_camera.jpg" % curr_id
            image_path = os.path.join(self.path_to_camcorder, image_name)
            left_shoulder_image.save(image_path, format=self.format)
            rgb_available = True
        if obs.right_shoulder_rgb is not None:
            right_shoulder_image = Image.fromarray(reformat(obs.right_shoulder_rgb))
            image_name = "%s_right_shoulder_camera.jpg" % curr_id
            image_path = os.path.join(self.path_to_camcorder, image_name)
            right_shoulder_image.save(image_path, format=self.format)
            rgb_available = True
        if obs.wrist_rgb is not None:
            wrist_image = Image.fromarray(reformat(obs.wrist_rgb))
            image_name = "%s_wrist_camera.jpg" % curr_id
            image_path = os.path.join(self.path_to_camcorder, image_name)
            wrist_image.save(image_path, format=self.format)
            rgb_available = True
        if obs.front_rgb is not None:
            front_image = Image.fromarray(reformat(obs.front_rgb))
            image_name = "%s_front_camera.jpg" % curr_id
            image_path = os.path.join(self.path_to_camcorder, image_name)
            front_image.save(image_path, format=self.format)
            rgb_available = True

        # save object poses as labels for RGB images
        if graspable_objs and rgb_available:
            with open(self.path_to_csv, mode='a') as label_file:
                writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for obj in graspable_objs:
                    obj_name = obj.get_object_name(obj.get_handle())
                    writer.writerow(["%s_%s" % (curr_id, obj_name)] + obj.get_pose().tolist())

        # --- MASK ---
        mask_available = False
        if obs.left_shoulder_mask is not None:
            left_shoulder_mask_image = Image.fromarray(reformat(obs.left_shoulder_mask))
            image_name = "%s_left_shoulder_camera_mask.jpg" % curr_id
            image_path = os.path.join(self.path_to_masks, image_name)
            left_shoulder_mask_image.save(image_path, format=self.format)
            mask_available = True
        if obs.right_shoulder_mask is not None:
            right_shoulder_mask_image = Image.fromarray(reformat(obs.right_shoulder_mask))
            image_name = "%s_right_shoulder_camera_mask.jpg" % curr_id
            image_path = os.path.join(self.path_to_masks, image_name)
            right_shoulder_mask_image.save(image_path, format=self.format)
            mask_available = True
        if obs.front_mask is not None:
            front_mask_image = Image.fromarray(reformat(obs.front_mask))
            image_name = "%s_front_camera_mask.jpg" % curr_id
            image_path = os.path.join(self.path_to_masks, image_name)
            front_mask_image.save(image_path, format=self.format)
            mask_available = True
        if obs.wrist_mask is not None:
            wrist_mask_image = Image.fromarray(reformat(obs.wrist_mask))
            image_name = "%s_wrist_camera_mask.jpg" % curr_id
            image_path = os.path.join(self.path_to_masks, image_name)
            wrist_mask_image.save(image_path, format=self.format)
            mask_available = True

        # save mask labels (only graspable objects and Panda robot)
        if mask_available:
            relevant_handles = []
            relevant_names = []
            if graspable_objs:
                graspable_obj_handles = [go.get_handle() for go in graspable_objs]
                relevant_names += [Object.get_object_name(goh) for goh in graspable_obj_handles]
                relevant_handles += graspable_obj_handles
            if robot_visuals:
                robot_visual_handles = [rv.get_handle() for rv in robot_visuals]
                relevant_names += [Object.get_object_name(rvh) for rvh in robot_visual_handles]
                relevant_handles += robot_visual_handles
            if relevant_handles:
                self.mask_label_dict[curr_id] = dict(zip(relevant_handles, relevant_names))

        self.counter += 1

    def save_dict_to_yaml(self):
        if os.path.exists(self.path_to_mask_label_file):
            with open(self.path_to_mask_label_file, 'r') as mask_label_file:
                cur_mask_label_file = yaml.safe_load(mask_label_file) or {}
                cur_mask_label_file.update(self.mask_label_dict)
            if mask_label_file:
                with open(self.path_to_mask_label_file, 'w') as new_mask_label_file:
                    yaml.safe_dump(cur_mask_label_file, new_mask_label_file)
        else:
            with open(self.path_to_mask_label_file, 'w') as mask_label_file:
                yaml.safe_dump(self.mask_label_dict, mask_label_file)

    def shutdown(self):
        self.save_dict_to_yaml()
