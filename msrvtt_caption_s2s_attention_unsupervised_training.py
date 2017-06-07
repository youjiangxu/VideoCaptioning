import numpy as np
import os
import h5py
import math

from utils import MsrDataUtil
from model import CaptionModel 
from model import Losses

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import cPickle as pickle
import time
import json


		
def exe_unsup_train(sess, data, batch_size, v2i, hf, unsup_training_feature_shape, 
	train, loss, unsup_input_feature, unsup_decoder_feature, true_video, flip=True, capl=16):

	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):
	# for batch_idx in xrange(500):

		# if batch_idx < 100:
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]

		data_v = MsrDataUtil.getBatchVideoFeature(batch_caption,hf,(80,2048))

		# if flip:
		# 	isflip = np.random.randint(0,2)
		# 	if isflip==1:
		# 		data_v = data_v[:,::-1]

		# start = np.random.randint(0,10)
		# data_v = data_v[:,start:start+30]

		input_v = data_v[:,0:40,:]

		input_pred_v = np.zeros((batch_size,40,2048),dtype=np.float32)

		input_pred_v[:,1::]=data_v[:,40:79]
		gt_video = data_v[:,40::]

		_, l = sess.run([train,loss],feed_dict={unsup_input_feature:input_v, unsup_decoder_feature:input_pred_v, true_video:gt_video})
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f' %(batch_idx+1,num_batch,l))
	total_loss = total_loss/num_batch
	return total_loss

def exe_unsup_test(sess, data, batch_size, v2i, hf, unsup_training_feature_shape, 
	loss, unsup_input_feature, unsup_decoder_feature, true_video, flip=True, capl=16):


	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):
	# for batch_idx in xrange(500):

		# if batch_idx < 100:
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		number_sample = len(batch_caption)
		data_v = MsrDataUtil.getBatchVideoFeature(batch_caption,hf,(80,2048))

		# if flip:
		# 	isflip = np.random.randint(0,2)
		# 	if isflip==1:
		# 		data_v = data_v[:,::-1]

		# start = np.random.randint(0,10)
		# data_v = data_v[:,start:start+30]

		input_v = data_v[:,0:40,:]

		input_pred_v = np.zeros((number_sample,40,2048),dtype=np.float32)

		input_pred_v[:,1::]=data_v[:,40:79]
		gt_video = data_v[:,40::]

		l = sess.run(loss,feed_dict={unsup_input_feature:input_v, unsup_decoder_feature:input_pred_v, true_video:gt_video})
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f' %(batch_idx+1,num_batch,l))
	total_loss = total_loss/num_batch

	return total_loss

def exe_train(sess, data, batch_size, v2i, hf, feature_shape, 
	train, loss, input_video, input_captions, y, capl=16):

	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):
	# for batch_idx in xrange(500):

		# if batch_idx < 100:
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]

		data_v = MsrDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		
		data_c, data_y = MsrDataUtil.getBatchTrainCaptionWithSparseLabel(batch_caption, v2i, capl=capl)

		_, l = sess.run([train,loss],feed_dict={input_video:data_v, input_captions:data_c, y:data_y})
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f' %(batch_idx+1,num_batch,l))
	total_loss = total_loss/num_batch
	return total_loss

def exe_test(sess, data, batch_size, v2i, i2v, hf, feature_shape, 
	predict_words, input_video, input_captions, y, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))+1

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v = MsrDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = MsrDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		[gw] = sess.run([predict_words],feed_dict={input_video:data_v, input_captions:data_c, y:data_y})

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


def main(hf,f_type,capl=16, d_w2v=512, output_dim=512,
		feature_shape=None,unsup_training_feature_shape=None,
		lr=0.01,
		batch_size=64,total_epoch=100,unsup_epoch=None,
		file=None,pretrained_model=None):
	'''
		capl: the length of caption
	'''

	# Create vocabulary
	v2i, train_data, val_data, test_data = MsrDataUtil.create_vocabulary_word2vec(file, capl=capl, v2i={'': 0, 'UNK':1,'BOS':2, 'EOS':3})

	i2v = {i:v for v,i in v2i.items()}

	print('building model ...')
	voc_size = len(v2i)
	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_captions = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
	y = tf.placeholder(tf.int32,shape=(None, capl))

	unsup_input_video = tf.placeholder(tf.float32, shape=(None,)+(40,2048),name='unsup_input_video')
	unsup_decoder_feature = tf.placeholder(tf.float32, shape=(None,)+(40,2048),name='unsup_decoder_feature')
	true_video = tf.placeholder(tf.float32, shape=(None,)+(40,2048),name='true_video')


	#
	#
	attentionCaptionModel = CaptionModel.UnsupTrainingAttentionCaptionModel(input_video, input_captions, unsup_input_video, 
															unsup_decoder_feature, voc_size, d_w2v, output_dim,
															T_k=[1,2,4,8])
	predict_score, predict_words, predict_vector = attentionCaptionModel.build_model()
	
	huber_Loss = Losses.Huber_Loss(predict_vector, true_video)
	unsup_training_loss = huber_Loss.build()
	print('unsup_training_loss.get_shape().as_list()',unsup_training_loss.get_shape().as_list())
	unsup_training_loss = tf.reduce_mean(tf.reduce_sum(unsup_training_loss,axis=[1,2])+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
	gvs = optimizer.compute_gradients(unsup_training_loss)
	capped_gvs = [(tf.clip_by_global_norm([grad], 10)[0][0], var) for grad, var in gvs ]
	unsup_training = optimizer.apply_gradients(capped_gvs)


	caption_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predict_score)+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	caption_loss = tf.reduce_mean(caption_loss)#

	caption_gvs = optimizer.compute_gradients(caption_loss)
	caption_capped_gvs = [(tf.clip_by_global_norm([grad], 10)[0][0], var) for grad, var in caption_gvs ]
	caption_training = optimizer.apply_gradients(caption_capped_gvs)

	# caption_training = optimizer.minimize(caption_loss)
	# 

	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
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


		export_path = '/home/xyj/usr/local/saved_model/msrvtt2017/'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])+'_B'+str(batch_size)
		
		# #unsupervised training 
		for epoch in xrange(unsup_epoch):
			print('Unsupervised Epoch: %d/%d, Batch_size: %d' %(epoch+1,unsup_epoch,batch_size))
			# # train phase
			tic = time.time()
			total_loss = exe_unsup_train(sess, train_data, batch_size, v2i, hf, unsup_training_feature_shape, unsup_training, unsup_training_loss, unsup_input_video, unsup_decoder_feature, true_video,capl=capl)

			print('    --Unsupervised Training--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))
			tic = time.time()
			total_loss = exe_unsup_test(sess, test_data, batch_size, v2i, hf, unsup_training_feature_shape, unsup_training_loss, unsup_input_video, unsup_decoder_feature, true_video,capl=capl)
			print('    --Unsupervised Testing--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			if not os.path.exists(export_path+'/unsupervised'):
				os.makedirs(export_path+'/unsupervised')
				print('mkdir %s' %export_path+'/unsupervised')
			save_path = saver.save(sess, export_path+'/unsupervised/'+'E'+str(epoch+1)+'_L'+str(total_loss)+'.ckpt')

		for epoch in xrange(total_epoch):
			# # shuffle
			

			# if epoch % 5==0:
				
			# train phase
			print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
			tic = time.time()
			total_loss = exe_train(sess, train_data, batch_size, v2i, hf, feature_shape, caption_training, caption_loss, input_video, input_captions, y,capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			tic = time.time()
			js = exe_test(sess, test_data, batch_size, v2i, i2v, hf, feature_shape, 
										predict_words, input_video, input_captions, y, capl=capl)
			print('    --Val--, .......Time:%.3f' %(time.time()-tic))

			

			#save model
			# export_path = '/home/xyj/usr/local/saved_model/msrvtt2017/s2s'+'_'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])+'_B'+str(batch_size)
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


	
	video_feature_dims=2048
	timesteps_v=80 # sequences length for video
	feature_shape = (timesteps_v,video_feature_dims)
	unsup_training_feature_shape = (timesteps_v/2,video_feature_dims)

	f_type = 'adam_regu_unsup_attention_resnet152'
	feature_path = '/home/xyj/usr/local/data/msrvtt/resnet152_pool5_f80.h5'
	# feature_path = '/home/xyj/usr/local/data/msrvtt/resnet152_pool5_f'+str(timesteps_v)+'.h5'

	# video_feature_dims=1024
	# timesteps_v=40 # sequences length for video
	# feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'attention_GoogleNet'
	# feature_path = '/mnt/data3/yzw/MSRVTT2017/features/googlenet_pl5_f'+str(timesteps_v)+'.h5'


	'''
	---------------------------------
	'''
	hf = h5py.File(feature_path,'r')['images']

	# pretrained_model = '/home/xyj/usr/local/saved_model/msrvtt2017/adam_regu_flip_unsup_attention_resnet152/lr0.0001_f80_B64/unsupervised/E2_L8667.36795628.ckpt'
	
	main(hf,f_type,capl=20, d_w2v=512, output_dim=512,
		feature_shape=feature_shape,unsup_training_feature_shape=unsup_training_feature_shape,
		lr=lr,batch_size=64,total_epoch=20,unsup_epoch=1,
		file='/home/xyj/usr/local/data/msrvtt',pretrained_model=None)
	

	
	
	
	


	
