#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:37:46 2018

@author: deeplearning
"""

def get_all_npy_from_folder(folder):
    import os
    
    fnames = list()
    for fname in os.listdir(folder):
        if fname.endswith(".npy"):
            fnames.append(fname)
    
    return fnames