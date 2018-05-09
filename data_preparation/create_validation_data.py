# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:26:14 2018

@author: mmoshe
"""

import os
import numpy as np

HMAP_FNAME = 'hmap_images.npy'
NUM_HMAPS_PER_FOLDER = 20

def validation_data_for_folders_and_save(folders_list, out_path):
    
    valid_data = get_validation_data_for_folders(folders_list)
    np.save(out_path, valid_data)

def get_validation_data_for_folders(folders_list):
    
    hmaps = list()
    stl_z = list()
    for fold in folders_list:
        hm, sz = create_validation_data_for_folder(fold)
        hmaps += hm
        stl_z += sz
    return {'hmaps': np.array(hmaps), 'stl_z': np.array(stl_z)}

def create_validation_data_for_folder(folder):
    
    hmaps_dat = np.load(os.path.join(folder, HMAP_FNAME)).tolist()
    dk = [key for key, value in hmaps_dat.items() if 'height_mat_mm' in value]
    order = np.random.permutation(len(dk))
    num_maps = min(NUM_HMAPS_PER_FOLDER, len(order))
    hmaps = list()
    stl_z = list()
    for index in order[:num_maps]:
        hmaps.append(hmaps_dat[dk[index]]['height_mat_mm'])
        stl_z.append(hmaps_dat[dk[index]]['stl_z'])
    return hmaps, stl_z