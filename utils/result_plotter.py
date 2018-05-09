# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 18:02:09 2018

@author: mmoshe
"""

import numpy as np
from mayavi import mlab

def plot_height_map_for_data(dat_path, num_hmap):
    
    data = np.load(dat_path, encoding = 'latin1').tolist()
    z = data['input'][num_hmap, :, :]
    x, y = get_xy(z.shape)
    prd_z = data['predicted z']
    stl_z = data['truth']
    plot_hmap(x, y, z, stl_z[num_hmap], prd_z[num_hmap], 'entire hmap {} {}', num_hmap, dat_path)
    T = np.abs(stl_z[num_hmap] - z) > 1; S = np.abs(stl_z[num_hmap] - prd_z[num_hmap]) > 1
    Q = np.logical_and(T, np.logical_not(S))
    plot_hmap(x[Q], y[Q], z[Q], stl_z[num_hmap][Q], prd_z[num_hmap][Q], 'bad original good prediction {} {}', num_hmap, dat_path)
    Q = np.logical_and(np.logical_not(T), S)
    plot_hmap(x[Q], y[Q], z[Q], stl_z[num_hmap][Q], prd_z[num_hmap][Q], 'good original bad prediction {} {}', num_hmap, dat_path)
    Q = np.logical_and(T, S)
    plot_hmap(x[Q], y[Q], z[Q], stl_z[num_hmap][Q], prd_z[num_hmap][Q], 'bad original bad prediction {} {}', num_hmap, dat_path)
    
def plot_hmap(x, y, z, stl_z, prd_z, name, index, dat_path):
    if len(x.ravel()) == 0:
        return
    tot_name = name.format(index, dat_path)
    fig = mlab.figure(tot_name)
    mlab.points3d(x.ravel(), y.ravel(), z.ravel(), color = (1,0,0), mode = 'point')
    mlab.points3d(x.ravel(), y.ravel(), stl_z.ravel(), color = (0,1,0), mode = 'point')
    mlab.points3d(x.ravel(), y.ravel(), prd_z.ravel(), color = (0,0,1), mode = 'point')
    return fig
    
def get_xy(shape):
    
    x = 0.09*np.array(range(shape[0]))
    y = 0.09*np.array(range(shape[1]))
    X, Y = np.meshgrid(x, y, indexing = 'ij')
    return X, Y