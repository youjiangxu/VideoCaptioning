import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import tensorflow as tf

import numpy as np




class Huber_Loss(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self,y_pred, y_true, delta=0.5):
		self.loss_name = 'huber_loss'
		self.y_pred = y_pred
		self.y_true = y_true
		self.delta = delta
	def implementation(self):
		diff_abs = tf.abs(self.y_true-self.y_pred,name='abs')
		delta_c = tf.constant(self.delta,name='delta')
		loss_abs = delta_c*(diff_abs-0.5*delta_c)
		loss_square = 0.5*diff_abs*diff_abs
		return tf.where(diff_abs<=delta_c,loss_square,loss_abs)

	def build(self):
		return self.implementation()
	

