#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:18:56 2018

@author: deeplearning
"""

import numpy as np
import os

SUB_FOLDS = {'lower', 'upper'}
IN_FNAME = 'hmap_images.npy'

class FileSeperator:
    
    def __init__(self, in_fold, out_fold):
        
        assert os.path.exists(in_fold), 'Input folder do not exist, probably wrong name.'
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)
            
        self.in_fold = in_fold
        self.out_fold = out_fold
        
    def save(self):
        
        for sub_fld in SUB_FOLDS:
            self.__save_file(IN_FNAME, sub_fld)
        
    def __save_file(self, fname, sub_fld):
        
        in_path = os.path.join(self.in_fold, '{}_jaw'.format(sub_fld), fname)
        data = np.load(in_path).tolist()
        batch_name = os.path.basename('{}_{}'.format(self.in_fold, sub_fld))
        self.__save_batch(batch_name, data)

    def __save_batch(self, batch_name, data):
        
        dk = [key for key, value in data.items() if 'height_mat_mm' in value]
        for index, key in enumerate(dk):
            out_fname = batch_name + '_{}.npy'.format(index)
            out_path = os.path.join(self.out_fold, out_fname)
            dat = {'hmap': data[key]['height_mat_mm'],'stl_z': data[key]['stl_z']}
            np.save(out_path, dat)

if __name__ == '__main__':

    in_fold = r'//fs05/Shared Box/Image4D_database/IO/IO_1.5.0.243-31132_Nofar_Shlomit'
    out_fold = r'\\fs05\Shared Box\Image4D_database\IO\seperate_file_data'
    FileSeperator(in_fold, out_fold).save()