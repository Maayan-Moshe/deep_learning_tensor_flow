# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 09:03:43 2018

@author: mmoshe
"""

import height_map_reconstruction.height_map_reducer as height_map_reducer
import height_map_reconstruction.data_expander_averager as data_expander_averager
import tensorflow as tf

class ReducerExpander:
    
    def __init__(self, params):
        
        self.params = params
        self.reducer_class = getattr(height_map_reducer, params['reduction']['reducer'])
        self.expander_class = getattr(data_expander_averager, params['expansion']['expander'])
        
    def reduce_expand_average_hmap(self, pre_data = {'hmap': None, 'weights': None}):
        pre_shape = pre_data['hmap'].get_shape()
        if pre_shape[1].value < 3 or pre_shape[2].value < 3:
                return pre_data
        reduced_data, pre_reduced_dat = self.__reduce_hmap(pre_data)
        average_data = self.__get_expanded_hmap(reduced_data, pre_reduced_dat, pre_data)                          
        return average_data
        
    def __reduce_hmap(self, pre_data):
        pre_shape = pre_data['hmap'].get_shape()
        ps = [-1, pre_shape[1].value, pre_shape[2].value, 1]
        
        pre_reduced_dat = self.__get_reduced_hmap(pre_data, ps[1], ps[2])
        average_data = self.reduce_expand_average_hmap(pre_reduced_dat)
        return average_data, pre_reduced_dat
        
    def __get_expanded_hmap(self, reduced_data, pre_reduced_dat, pre_data):
        ps = pre_data['hmap'].get_shape()
        with tf.name_scope('expanding_hmap_{}x{}'.format(ps[1].value, ps[2].value)):
             expander_inst = self.expander_class(self.params['expansion'])
             average_data = expander_inst.get_hmap(reduced_data, pre_reduced_dat, pre_data)
             return average_data
        
    def __get_reduced_hmap(self, pre_data, num_rows, num_cols):
        with tf.name_scope('reducing_hmap_{}x{}'.format(num_rows, num_cols)):
            reducer_inst = self.reducer_class(num_rows, num_cols, self.params['reduction'])
            reduced_dat = reducer_inst.get_reduced(pre_data)
            return reduced_dat 
    

        