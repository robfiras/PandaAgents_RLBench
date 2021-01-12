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
        # make both directories
        if not os.path.exists(self.path_to_camcorder) and unique_id == 0:
            os.mkdir(self.path_to_camcorder)
        if not os.path.exists(self.path_to_camcorder_task_low_dim) and unique_id == 0:
            os.mkdir(self.path_to_camcorder_task_low_dim)
        if not os.path.exists(self.path_to_masks) and unique_id == 0:
            os.mkdir(self.path_to_masks)

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
        relevant_handles = self.get_relevant_handles(robot_visuals, graspable_objs)
        if relevant_handles:
            if obs.left_shoulder_mask is not None:
                left_shoulder_mask_image = self.filter_for_handles(obs.left_shoulder_mask, relevant_handles)
                left_shoulder_mask_image = Image.fromarray(reformat(left_shoulder_mask_image))
                image_name = "%s_left_shoulder_camera.jpg" % curr_id
                image_path = os.path.join(self.path_to_masks, image_name)
                left_shoulder_mask_image.save(image_path, format=self.format)
            if obs.right_shoulder_mask is not None:
                right_shoulder_mask_image = self.filter_for_handles(obs.right_shoulder_mask, relevant_handles)
                right_shoulder_mask_image = Image.fromarray(reformat(right_shoulder_mask_image))
                image_name = "%s_right_shoulder_camera.jpg" % curr_id
                image_path = os.path.join(self.path_to_masks, image_name)
                right_shoulder_mask_image.save(image_path, format=self.format)
            if obs.wrist_mask is not None:
                wrist_mask_image = self.filter_for_handles(obs.wrist_mask, relevant_handles)
                wrist_mask_image = Image.fromarray(reformat(wrist_mask_image))
                image_name = "%s_wrist_camera.jpg" % curr_id
                image_path = os.path.join(self.path_to_masks, image_name)
                wrist_mask_image.save(image_path, format=self.format)
            if obs.front_mask is not None:
                front_mask_image = self.filter_for_handles(obs.front_mask, relevant_handles)
                front_mask_image = Image.fromarray(reformat(front_mask_image))
                image_name = "%s_front_camera.jpg" % curr_id
                image_path = os.path.join(self.path_to_masks, image_name)
                front_mask_image.save(image_path, format=self.format)

        self.counter += 1

    @staticmethod
    def get_relevant_handles(robot_visuals, graspable_objs):
        relevant_handles = []
        if graspable_objs:
            relevant_handles += [go.get_handle() for go in graspable_objs]
        if robot_visuals:
            relevant_handles += [rv.get_handle() for rv in robot_visuals]
        return relevant_handles

    @staticmethod
    def filter_for_handles(mask, handles):
        new_mask = np.zeros_like(mask)
        for handle in handles:
            new_mask += (mask == handle).astype(float)
        return new_mask
