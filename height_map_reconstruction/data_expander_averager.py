# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:05:30 2018

@author: mmoshe
"""

from utils.utils import get_normalized_weights, stack_input, reshape_hmap
from .data_expander import expand_data
import tensorflow as tf

class DataExpanderAveragerBase:
    
    def __init__(self, params = None):
        
        pass
    
    def get_hmap(self, reduced_data, pre_reduced_dat, pre_data):
        
        raise NotImplementedError

class DataExpanderAverager(DataExpanderAveragerBase):
    
    def __init__(self, params = None):
        
        pass
    
    def get_hmap(self, reduced_data, pre_reduced_dat, pre_data):
        
        pre_shape = pre_data['hmap'].get_shape()
        scope = 'expand_average_hmap_{}x{}'.format(pre_shape[1].value, pre_shape[2].value)
        with tf.name_scope(scope):
            expanded_data = expand_data(reduced_data, pre_reduced_dat, pre_shape)
            average_data = average_hmaps(expanded_data, pre_data)
                                      
            return average_data

class DataExpanderAveragerAdditioner(DataExpanderAveragerBase):
    
    def __init__(self, params = {'regularization': {'kernel': 0.5, 'bias': 0.5}}):
        
        DataExpanderAveragerBase.__init__(self, params)
        self.reg = params['regularization']
        
    def get_hmap(self, reduced_data, pre_reduced_dat, pre_data):
        
        pre_shape = pre_data['hmap'].get_shape()
        scope = 'expand_average_addition_hmap_{}x{}'.format(pre_shape[1].value, pre_shape[2].value)
        with tf.name_scope(scope):
            expanded_data = expand_data(reduced_data, pre_reduced_dat, pre_shape)
            average_data = average_hmaps(expanded_data, pre_data)
            average_data['addition_hmap'], addition_weights = self.__get_addition_hmap(average_data, expanded_data, pre_data)
            average_data['hmap'] = average_data['hmap'] + average_data['addition_hmap']
            average_data['weights'] = tf.concat([average_data['weights'], addition_weights], axis = 3)
                                      
            return average_data
        
    def __get_addition_hmap(self, average_data, expanded_data, pre_data):
        
        with tf.name_scope('additional_hmap'):
            stacked_input = get_stacked_input([average_data, expanded_data, pre_data])
            
            first_layer = tf.contrib.layers.conv2d(inputs = stacked_input, stride = 1,
                                        num_outputs = 32, kernel_size = 3, activation_fn = tf.nn.relu,
                                        weights_regularizer = tf.contrib.layers.l2_regularizer(self.reg['kernel']),
                                        biases_regularizer = tf.contrib.layers.l2_regularizer(self.reg['bias']), 
                                        biases_initializer=tf.initializers.truncated_normal(mean = 2e-3, stddev = 4e-4))
                                        
            second_layer = tf.contrib.layers.conv2d(inputs = first_layer, stride = 1,
                                        num_outputs = 8, kernel_size = 3, activation_fn = tf.nn.relu,
                                        weights_regularizer = tf.contrib.layers.l2_regularizer(self.reg['kernel']),
                                        biases_regularizer = tf.contrib.layers.l2_regularizer(self.reg['bias']), 
                                        biases_initializer=tf.initializers.truncated_normal(mean = 2e-3, stddev = 4e-4))
                                        
            addition_hmap = tf.contrib.layers.conv2d(inputs = second_layer, stride = 1,
                                        num_outputs = 1, kernel_size = 3, activation_fn = None,
                                        weights_regularizer = tf.contrib.layers.l2_regularizer(self.reg['kernel']),
                                        biases_regularizer = tf.contrib.layers.l2_regularizer(self.reg['bias']), 
                                        biases_initializer=tf.initializers.truncated_normal(mean = 2e-3, stddev = 4e-4))
                                        
            return addition_hmap, second_layer
        
def average_hmaps(expanded_data, pre_data):
    
    with tf.name_scope('averaging_hmap'):
        stacked_input = get_stacked_input([expanded_data, pre_data])
        
        first_layer = tf.contrib.layers.conv2d(inputs = stacked_input, stride = 1,
                                    num_outputs = 32, kernel_size = 3, activation_fn = tf.nn.relu,
                                    biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))
                                    
        weights = tf.contrib.layers.conv2d(inputs = first_layer, stride = 1,
                            num_outputs = 3, kernel_size = 3, activation_fn = tf.nn.relu,
                            biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))
                            
        normed_weights = get_normalized_weights(weights)
        averaged_hmap = expanded_data['hmap'][:, :, :, 0]*normed_weights[:, :, :, 0] + \
                        reshape_hmap(pre_data['hmap'])[:, :, :, 0]*normed_weights[:, :, :, 1] + \
                        tf.zeros(tf.shape(expanded_data['hmap']))[:, :, :, 0]*normed_weights[:, :, :, 2]
        return {'hmap': reshape_hmap(averaged_hmap), 'weights': weights}
    
def get_stacked_input(data_dict_list):
    
    with tf.name_scope('stacking_input'):
        pre_st_input = dict()
        for index, dat_dict in enumerate(data_dict_list):
            pre_st_input.update({'{}_{}'.format(index, key): value for key, value in dat_dict.items()})
        stacked_input = stack_input(pre_st_input)
                
        return stacked_input