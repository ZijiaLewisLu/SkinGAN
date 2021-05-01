import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
from . import landmark_tools
import json
import pickle

class Dataset():

    def __init__(self, datafolder, image_list, landmark_dict, 
                        image_size=[64, 64],
                        transforms=None, 
                        preprocess=True,
                        size_ratio=10, filter_magic=1.5):
        self.datafolder = datafolder
        self.image_list = image_list
        self.landmark_dict = landmark_dict

        self.data_dict = {}
        if preprocess:
            for image_name in self.image_list:
                self._process_one_image(image_name)

        self.image_size = image_size
        self.transforms = transforms
        self.size_ratio = size_ratio
        self.filter_magic = filter_magic

    def _process_one_image(self, image_name):
        img = plt.imread( os.path.join(self.datafolder, image_name)+'.jpg' ) 
        landmark_list = self.landmark_dict[image_name]
        # tri_points, simplices, triangle_weight, named_lm 
        tri_params = landmark_tools.obtain_preprocess_triangles(img, landmark_list)
        self.data_dict[image_name] = [ img, tri_params ]

    def __getitem__(self, index):
        image_name = self.image_list[index]
        if image_name not in self.data_dict:
            self._process_one_image(image_name)


        img, tri_params = self.data_dict[image_name]        

        ptr, patch = landmark_tools.generate_filtered_patch(img, tri_params,
                                                size_ratio=self.size_ratio, 
                                                filter_magic=self.filter_magic)

        patch = cv2.resize(patch, self.image_size)

        if self.transforms is not None:
            patch = self.transforms(patch)

        return patch

    def __len__(self):
        return len(self.image_list)

def load_imagelist_landmark(datafolder):

    with open(os.path.join(datafolder, 'annotated_images.json'), 'r') as fp:
        image_list = json.load(fp)

    landmark_dict = {}
    for image_name in image_list:
        lm_file = os.path.join(datafolder, 'Results', image_name + '.pkl')
        with open(lm_file, 'rb') as fp:
            landmark = pickle.load(fp)
        landmark_dict[image_name] = landmark

    level_dict = { i:[] for i in range(4) }

    for image_name in image_list:
        level = int(image_name[5])
        level_dict[level].append(image_name)

    no_acne = level_dict[0] # + level_dict[1]
    acne    = level_dict[2] + level_dict[3]
    
    imagefolder = os.path.join(datafolder, "Classification", "JPEGImages")

    return imagefolder, acne, no_acne, landmark_dict


