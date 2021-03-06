#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:34:24 2017

@author: deeplearning
"""
import tensorflow as tf
import data_preparation.DataFeeders as data_feeders
import os
from time import gmtime, strftime
from utils.trainer import Trainer
from utils.validator import Validator

MAX_LOSS = 1e10

PARAMS = {'train_params':{
                'initial_learning_rate': 1e-3,
                'minimum_learning_rate': 1e-5,
                'state_fname': 'real_data_multiscale_18_Mar_2018_14_31.ckpt',
                'state_folder': r'C:\Users\mmoshe\Documents\teeth_segmentation\hmap_data\saved_models\hmap_multi_scale',
                'summaries_dir': r'C:\Users\mmoshe\Documents\teeth_segmentation\hmap_data\results\summaries_hmaps',
                'saved_state_fname': 'real_data_multiscale',
                'num_iterations': 200,
                'summary_name': 'nn_real_data_multiscale',
                'validation_rate_step': 5},
          'graph_params':{
                'module_path': 'height_map_reconstruction.multiscale_graph_builder',
                'image_shape': (178, 233),
                'cost': 'huber_plus_abs_derivative_cost',
                'cost_params': {'derivative_strength': 0e-1,
                                'moving_tissue_slope': 0e0},
                'loss_producer': 'LossTrainingProducerFiniteRangeMovingTissue',
                'range_size_pix': 15,
                'max_z_mm': 25, 'min_z_mm': 0,
                'reduction': {
                        'reducer': 'HeightMapReducerFiller',
                        'regularization': {'kernel': 5e-1, 'bias': 5e-1}},
                'expansion': {
                        'expander': 'DataExpanderAveragerAdditioner',
                        'regularization': {'kernel': 5e-1, 'bias': 5e-1}}},
          'data_params':{
                'folder': r'//fs05/Shared Box/Image4D_database/IO/seperate_file_data',
                'data_path': r'//fs05/Shared Box/Image4D_database/IO/validation_data/aviya_shlomit_validation.npy',
                'batch_size': 60,
                'cropping': 0,
                'feeder': 'SeparateFilesFeeder',
                'validation_feeder': 'OneFileValidatorFeeder'}
          }

class SessionTrainer:
    
    def __init__(self, sess, components, params):
        
        self.sess = sess
        self.params = params
        self.trainer = Trainer(self.sess, components, self.params)
        self.__init_session(components)
        self.validator = Validator(self.sess, components, params)
        self.validation_loss = MAX_LOSS
        self.validation_rate_step = params['validation_rate_step']
            
    def train(self):

        index = 0
        while self.trainer.should_continue():
            self.__train(index)
            index += 1
        
        self.__save_state()
    
    def __init_session(self, components):
        
        self.saver = components['saver']
        self.sess.run(tf.global_variables_initializer())
        if self.params['state_fname']:
            session_path = os.path.join(self.params['state_folder'], self.params['state_fname'])
            self.saver.restore(self.sess, session_path)
            
    def __train(self, index):
        
        check = index%self.validation_rate_step == 0
        self.trainer.train_batch(index, check)
        if check:
            self.__validate_loss(index)
            
    def __validate_loss(self, index):
        
        va = self.validator.validate(index)
        if va > self.validation_loss:
            self.trainer.reduce_learning_rate()
        self.validation_loss = va
            
    def __save_state(self):
        
        time_str = strftime("%d_%b_%Y_%H_%M", gmtime())
        out_state_fname = self.params['saved_state_fname'] + '_{}.ckpt'.format(time_str)
        session_path = os.path.join(self.params['state_folder'], out_state_fname)
        self.saver.save(self.sess, os.path.join(self.params['state_folder'], session_path))

def prepare_components(params):
    from importlib import import_module
    
    graph_module = import_module(params['graph_params']['module_path'])
    graph_p = graph_module.prepare_graph(params['graph_params'])
    train_feeder = getattr(data_feeders, params['data_params']['feeder'])(**params['data_params'])
    validation_feeder = getattr(data_feeders, params['data_params']['validation_feeder'])(**params['data_params'])
    components = {'graph': graph_p, 'train feeder': train_feeder, 
                  'saver': tf.train.Saver(), 'summary merge': tf.summary.merge_all(),
                  'validation feeder': validation_feeder}
    return components

if __name__ == '__main__':
    
    params = PARAMS

    components = prepare_components(params)
    
    with tf.Session() as sess:
        SessionTrainer(sess, components, params['train_params']).train()
