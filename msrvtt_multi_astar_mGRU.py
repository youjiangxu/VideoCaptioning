import numpy as np
import os
import h5py
import math

from utils import MsrDataUtil
from model import mGRUAstarCaptionModel 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import cPickle as pickle
import time
import json


		
def exe_train(sess, data, batch_size, v2i, hf1, hf2, hf3, feature_shape1, feature_shape2, 
	train, loss, input_video1, input_video2, input_video3, input_captions, y, capl=16):

	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):
	# for batch_idx in xrange(500):

		# if batch_idx < 100:
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]

		data_v1 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf1,feature_shape1)
		data_v2 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf2,feature_shape2)
		data_v3 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf3,feature_shape3)

		data_c, data_y = MsrDataUtil.getBatchTrainCaptionWithSparseLabel(batch_caption, v2i, capl=capl)

		_, l = sess.run([train,loss],feed_dict={input_video1:data_v1, input_video2:data_v2, input_video3:data_v3, input_captions:data_c,  y:data_y})
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f' %(batch_idx+1,num_batch,l))
	total_loss = total_loss/num_batch
	return total_loss

def exe_test(sess, data, batch_size, v2i, i2v, hf1, hf2, hf3, feature_shape1, feature_shape2, 
	predict_words, input_video1, input_video2, input_video3, input_captions, y, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))+1

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v1 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf1,feature_shape1)
		data_v2 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf2,feature_shape2)
		data_v3 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf3,feature_shape3)

		data_c, data_y = MsrDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		[gw] = sess.run([predict_words],feed_dict={input_video1:data_v1, input_video2:data_v2, input_video3:data_v3, input_captions:data_c, y:data_y})

		generated_captions = MsrDataUtil.convertCaptionI2V(batch_caption, gw, i2v)

		for idx, sen in enumerate(generated_captions):
			print('%s : %s' %(batch_caption[idx].keys()[0],sen))
			caption_output.append({'image_id':batch_caption[idx].keys()[0],'caption':sen})
	
	js = {}
	js['val_predictions'] = caption_output

	return js


def evaluate_mode_by_shell(res_path,js):
	with open(res_path, 'w') as f:
		json.dump(js, f)

	command ='/home/xyj/usr/local/tools/caption/caption_eval/msrvtt_eval.sh '+ res_path
	os.system(command)


def main(hf1, hf2, hf3,f_type,capl=16, d_w2v=512, output_dim=512,
		feature_shape1=None, feature_shape2=None, feature_shape3=None, lr=0.01,
		batch_size=64,total_epoch=100,
		file=None,pretrained_model=None):
	'''
		capl: the length of caption
	'''

	# Create vocabulary
	v2i, train_data, val_data, test_data = MsrDataUtil.create_vocabulary_word2vec(file, capl=capl, word_threshold=1, v2i={'': 0, 'UNK':1,'BOS':2, 'EOS':3})

	i2v = {i:v for v,i in v2i.items()}

	print('building model ...')
	voc_size = len(v2i)

	input_video1 = tf.placeholder(tf.float32, shape=(None,)+feature_shape1,name='input_video1')
	input_video2 = tf.placeholder(tf.float32, shape=(None,)+feature_shape2,name='input_video2')
	input_video3 = tf.placeholder(tf.float32, shape=(None,)+feature_shape3,name='input_video3')

	input_captions = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
	y = tf.placeholder(tf.int32,shape=(None, capl))

	attentionCaptionModel = mGRUAstarCaptionModel.mGRUMultiAstarAttentionCaptionModel(input_video1, input_video2, input_video3, input_captions, voc_size, d_w2v, output_dim)
	predict_score, predict_words, loss_mask = attentionCaptionModel.build_model()
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predict_score)

	loss = tf.reduce_sum(loss,reduction_indices=[-1])/tf.reduce_sum(loss_mask,reduction_indices=[-1])

	loss = tf.reduce_mean(loss)+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

	optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
	

	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_global_norm([grad], 10)[0][0], var) for grad, var in gvs ]
	train = optimizer.apply_gradients(capped_gvs)

	# optimizer = tf.train.RMSPropOptimizer(lr,decay=0.9, momentum=0.0, epsilon=1e-8)
	# train = optimizer.minimize(loss)

	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
	# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	config.log_device_placement=False

	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
		if pretrained_model is not None:
			saver.restore(sess, pretrained_model)
			print('restore pre trained file:' + pretrained_model)

		for epoch in xrange(total_epoch):
			# # shuffle
			print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
			# # train phase
			tic = time.time()
			total_loss = exe_train(sess, train_data, batch_size, v2i, hf1, hf2, hf3, feature_shape1, feature_shape2, 
										train, loss, input_video1, input_video2, input_video3, input_captions, y, capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			tic = time.time()
			js = exe_test(sess, test_data, batch_size, v2i, i2v, hf1, hf2, hf3, feature_shape1, feature_shape2,
										predict_words, input_video1, input_video2, input_video3, input_captions, y, capl=capl)
			print('    --Val--, .......Time:%.3f' %(time.time()-tic))

			

			#save model
			export_path = '/home/xyj/usr/local/saved_model/msrvtt2017/'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape1[0])+'_B'+str(batch_size)
			if not os.path.exists(export_path+'/model'):
				os.makedirs(export_path+'/model')
				print('mkdir %s' %export_path+'/model')
			if not os.path.exists(export_path+'/res'):
				os.makedirs(export_path+'/res')
				print('mkdir %s' %export_path+'/res')

			# eval
			res_path = export_path+'/res/'+f_type+'_E'+str(epoch+1)+'.json'
			evaluate_mode_by_shell(res_path,js)


			save_path = saver.save(sess, export_path+'/model/'+'E'+str(epoch+1)+'_L'+str(total_loss)+'.ckpt')
			print("Model saved in file: %s" % save_path)
		

if __name__ == '__main__':


	lr = 0.0001

	d_w2v = 512
	output_dim = 512
	batch_size = 128
	total_epoch = 20

	
	
	timesteps_v=5 # sequences length for video
	video_feature_dims=4096
	feature_shape1 = (timesteps_v,video_feature_dims)

	video_conv5_dims = 512
	video_conv5_height = 4
	video_conv5_width = 4
	feature_shape2 = (timesteps_v,video_conv5_dims,video_conv5_height,video_conv5_width)

	video_conv4_dims = 512
	video_conv4_height = 7
	video_conv4_width = 7
	feature_shape3 = (timesteps_v,video_conv4_dims,2,video_conv4_height,video_conv4_width)

	f_type = 'multi_astar_sparse_c3d_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)
	# feature_path = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	feature_path1 = '/mnt/data3/yzw/MSRVTT2017/c3d_info/msrvtt_c3d_fc7_f5.h5'
	feature_path2 = '/mnt/data3/yzw/MSRVTT2017/c3d_info/msrvtt_c3d_pool5_f5.h5'
	feature_path3 = '/mnt/data3/yzw/MSRVTT2017/c3d_info/msrvtt_c3d_pool4_f5.h5'
	'''
	---------------------------------
	'''
	
	hf1 = h5py.File(feature_path1,'r')['images']
	hf2 = h5py.File(feature_path2,'r')['images']
	hf3 = h5py.File(feature_path3,'r')['images']
	# pretrained_model = '/home/xyj/usr/local/saved_model/msrvtt2017/s2s_mgru_attention_resnet152/lr0.0002_f40/model/E25_L0.736578735637.ckpt'
	
	main(hf1,hf2,hf3,f_type,capl=20, d_w2v=d_w2v, output_dim=output_dim,
		feature_shape1=feature_shape1,feature_shape2=feature_shape2,feature_shape3=feature_shape3,lr=lr,
		batch_size=batch_size,total_epoch=total_epoch,
		file='/home/xyj/usr/local/data/msrvtt',pretrained_model=None)
	

	
	
	
	


	