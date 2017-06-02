import numpy as np
import os
import h5py
import math

from utils import MsrDataUtil
from model import mGRUCaptionModel 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import cPickle as pickle
import time
import json

import sys
		
def exe_train(sess, data, batch_size, v2i, hf1, hf2, feature_shape, 
	train, loss, input_video, input_captions, y, capl=16):

	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):
	# for batch_idx in xrange(500):

		# if batch_idx < 100:
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]

		data_v1 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf1,(feature_shape[0],2048))
		data_v2 = MsrDataUtil.getBatchC3DVideoFeature(batch_caption,hf2,(feature_shape[0],4096))

		# data_v1 = data_v1/(np.linalg.norm(data_v1, ord=None, axis=-1, keepdims=True)+sys.float_info.epsilon)
		# data_v2 = data_v2/(np.linalg.norm(data_v2, ord=None, axis=-1, keepdims=True)+sys.float_info.epsilon)

		data_v = np.concatenate((data_v1,data_v2),axis=-1)
		data_c, data_y = MsrDataUtil.getBatchTrainCaptionWithSparseLabel(batch_caption, v2i, capl=capl)

		_, l = sess.run([train,loss],feed_dict={input_video:data_v, input_captions:data_c,  y:data_y})
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f' %(batch_idx+1,num_batch,l))
	total_loss = total_loss/num_batch
	return total_loss

def exe_test(sess, data, batch_size, v2i, i2v, hf1, hf2, feature_shape, 
	predict_words, input_video, input_captions, y, finished_beam, logprobs_finished_beams, past_logprobs, beam_hidden_state, past_symbols_states, finished_beams_states, capl=16):
	
	caption_output = []
	total_data = len(data)
	
	num_batch = int(math.ceil((total_data*1.0/batch_size)))
	print(num_batch)
	


	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v1 = MsrDataUtil.getBatchVideoFeature(batch_caption,hf1,(feature_shape[0],2048))
		data_v2 = MsrDataUtil.getBatchC3DVideoFeature(batch_caption,hf2,(feature_shape[0],4096))

		# data_v1 = data_v1/(np.linalg.norm(data_v1, ord=None, axis=-1, keepdims=True)+sys.float_info.epsilon)
		# data_v2 = data_v2/(np.linalg.norm(data_v2, ord=None, axis=-1, keepdims=True)+sys.float_info.epsilon)

		data_v = np.concatenate((data_v1,data_v2),axis=-1)
		
		data_c, data_y = MsrDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		[gw,tw,gp, gl, pp, pss, fbs] = sess.run([finished_beam, predict_words, logprobs_finished_beams, past_logprobs, beam_hidden_state, past_symbols_states, finished_beams_states],feed_dict={input_video:data_v, input_captions:data_c, y:data_y})

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


def main(hf1,hf2,f_type,capl=16, d_w2v=512, output_dim=512,
		feature_shape=None,lr=0.01,
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

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_captions = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
	y = tf.placeholder(tf.int32,shape=(None, capl))

	attentionCaptionModel = mGRUCaptionModel.mGRUAttentionBeamsearchCaptionMergedFeaureModel(input_video, input_captions, voc_size, d_w2v, output_dim,
		max_len = 16, beamsearch_batchsize = 1, beam_size=5)


	predict_score, predict_words, loss_mask, finished_beam, logprobs_finished_beams, past_logprobs, beam_hidden_state, past_symbols_states, finished_beams_states = attentionCaptionModel.build_model()
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
			# train phase
			tic = time.time()
			total_loss = exe_train(sess, train_data, batch_size, v2i, hf1, hf2, feature_shape, train, loss, input_video, input_captions, y, capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			tic = time.time()
			js = exe_test(sess, test_data, 1, v2i, i2v, hf1, hf2, feature_shape, 
										predict_words, input_video, input_captions, y, finished_beam, logprobs_finished_beams, past_logprobs, beam_hidden_state, past_symbols_states, finished_beams_states, capl=capl)
			print('    --Val--, .......Time:%.3f' %(time.time()-tic))

			

			#save model
			export_path = '/home/xyj/usr/local/saved_model/msrvtt2017/s2s'+'_'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])+'_B'+str(batch_size)
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

	d_w2v = 1024
	output_dim = 1024

	video_feature_dims=2048+4096
	timesteps_v=40 # sequences length for video
	# feature_path1 = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	feature_path1 = '/home/xyj/usr/local/data/msrvtt/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	feature_shape1 = (timesteps_v,2048)
	

	f_type = 'norm_sparse_mgru_merge_c3d_resnet_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)
	
	# feature_path2 = '/data/xyj/msrvtt_c3d_fc6.h5'
	feature_path2 = '/home/xyj/usr/local/data/msrvtt/msrvtt_c3d_fc6.h5'
	feature_shape2 = (timesteps_v,4096)

	feature_shape = (timesteps_v,video_feature_dims)
	'''
	---------------------------------
	'''

	# video_feature_dims=4096
	# timesteps_v=40 # sequences length for video
	# feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'mgru_attention_places205-alex-fc7_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)
	# feature_path = '/data/xyj/places205-alex-fc7-'+str(timesteps_v)+'f.h5'
	# # feature_path = '/home/xyj/usr/local/data/msrvtt/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	'''
	---------------------------------
	'''
	# video_feature_dims=2048
	# timesteps_v=40 # sequences length for video
	# feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'mgru_attention_resnet152_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)
	# feature_path = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	# feature_path = '/home/xyj/usr/local/data/msrvtt/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	'''
	---------------------------------
	'''
	# video_feature_dims=1024
	# timesteps_v=40 # sequences length for video
	# feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'attention_GoogleNet'
	# feature_path = '/mnt/data3/yzw/MSRVTT2017/features/googlenet_pl5_f'+str(timesteps_v)+'.h5'


	'''
	---------------------------------
	'''
	hf1 = h5py.File(feature_path1,'r')['images']
	hf2 = h5py.File(feature_path2,'r')['images']
	# pretrained_model = '/home/xyj/usr/local/saved_model/msrvtt2017/s2s_sparse_mgru_merge_c3d_resnet_dw2v1024_outputdim1024/lr0.0001_f40_B128/model/E5_L2.84255223031.ckpt'
	
	main(hf1,hf2,f_type,capl=20, d_w2v=1024, output_dim=1024,
		feature_shape=feature_shape,lr=lr,
		batch_size=128,total_epoch=40,
		file='/home/xyj/usr/local/data/msrvtt',pretrained_model=None)
	

	
	
	
	


	
