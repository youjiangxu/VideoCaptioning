import numpy as np
import os
import h5py
import math

from utils import MsrDataUtil
from model import mGRUAudioModel 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import cPickle as pickle
import time
import json


		
def exe_train(sess, data, audio_info, batch_size, v2i, hf, feature_shape, 
	train, loss, input_video, input_captions, input_audio, y, capl=16):

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
		data_audio = MsrDataUtil.getBatchVideoAudioInfo(batch_caption, audio_info)
		_, l = sess.run([train,loss],feed_dict={input_video:data_v, input_captions:data_c, input_audio:data_audio, y:data_y})
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f' %(batch_idx+1,num_batch,l))
	total_loss = total_loss/num_batch
	return total_loss

def exe_test(sess, data, audio_info, batch_size, v2i, i2v, hf, feature_shape, 
	predict_words, input_video, input_captions, input_audio, y, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))+1

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v = MsrDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = MsrDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		data_audio = MsrDataUtil.getBatchVideoAudioInfo(batch_caption, audio_info)
		
		[gw] = sess.run([predict_words],feed_dict={input_video:data_v, input_captions:data_c, input_audio:data_audio, y:data_y})

		generated_captions = MsrDataUtil.convertCaptionI2V(batch_caption, gw, i2v)

		for idx, sen in enumerate(generated_captions):
			print('%s : %s' %(batch_caption[idx].keys()[0],sen))
			caption_output.append({'image_id':batch_caption[idx].keys()[0],'caption':sen})
	
	js = {}
	js['val_predictions'] = caption_output

	return js

def beam_search_exe_test(sess, data, audio_info, batch_size, v2i, i2v, hf, feature_shape, 
	predict_words, input_video, input_captions, input_audio, y, finished_beam, logprobs_finished_beams, past_logprobs, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v = MsrDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = MsrDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		data_audio = MsrDataUtil.getBatchVideoAudioInfo(batch_caption, audio_info)
		[gw,tw,gp, gl] = sess.run([finished_beam, predict_words, logprobs_finished_beams, past_logprobs],
			                  feed_dict={input_video:data_v, input_captions:data_c, input_audio:data_audio, y:data_y})

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


def main(hf, audio_info, f_type,capl=16, d_w2v=512, output_dim=512,
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
	input_audio = tf.placeholder(tf.float32, shape=(None,34,2), name='input_audio')
	y = tf.placeholder(tf.int32,shape=(None, capl))

	attentionCaptionModel = mGRUAudioModel.mGRUAudioAttentionBeamsearchCaptionModel(input_video, input_captions, input_audio, voc_size, d_w2v, output_dim, 
		input_audio_dim=34, T_k=[1,2,4,8], max_len = 20, beamsearch_batchsize = 1, beam_size=5)

	predict_score, predict_words, loss_mask, finished_beam, logprobs_finished_beams, past_logprobs, = attentionCaptionModel.build_model()
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predict_score)

	loss = tf.reduce_sum(loss,reduction_indices=[-1])/tf.reduce_sum(loss_mask,reduction_indices=[-1])+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

	loss = tf.reduce_mean(loss)

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

		for epoch in xrange(total_epoch):
			# shuffle
			print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
			# # train phase
			tic = time.time()
			total_loss = exe_train(sess, train_data, audio_info, batch_size, v2i, hf, feature_shape, train, loss, input_video, input_captions, input_audio, y, capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			tic = time.time()
			js = exe_test(sess, test_data, audio_info, batch_size, v2i, i2v, hf, feature_shape, 
										predict_words, input_video, input_captions, input_audio, y, capl=capl)
			print('    --Val--, .......Time:%.3f' %(time.time()-tic))

			# tic = time.time()
			

			# js = beam_search_exe_test(sess, test_data, audio_info, 1, v2i, i2v, hf, feature_shape, 
			# 							predict_words, input_video, input_captions, input_audio, y, finished_beam, logprobs_finished_beams, past_logprobs, capl=capl)
			# print('    --beam search--, .......Time:%.3f' %(time.time()-tic))
			

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

	d_w2v = 512
	output_dim = 512
	
	capl=20
	batch_size=128
	total_epoch=20
	# video_feature_dims=4096
	# timesteps_v=40 # sequences length for video
	# feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'mgru_attention_places205-alex-fc7_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)
	# feature_path = '/data/xyj/places205-alex-fc7-'+str(timesteps_v)+'f.h5'
	# feature_path = '/home/xyj/usr/local/data/msrvtt/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	'''
	---------------------------------
	'''
	video_feature_dims=2048
	timesteps_v=40 # sequences length for video
	feature_shape = (timesteps_v,video_feature_dims)

	f_type = 'audio_sparse_mgru1248_attention_resnet152_dw2v'+str(d_w2v)+'_outdim'+str(output_dim)
	feature_path = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'

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
	hf = h5py.File(feature_path,'r')['images']
	audio_info = np.load('/home/xyj/usr/local/data/msrvtt/audio2feats_ch1_mean_std.npy')
	print('audio_info:','/home/xyj/usr/local/data/msrvtt/audio2feats_ch1_mean_std.npy')
	# pretrained_model = '/home/xyj/usr/local/saved_model/msrvtt2017/s2s_audio_sparse_mgru1248_attention_resnet152_dw2v512_outdim512/lr0.0001_f40_B64/model/E2_L3.76600125387.ckpt'
	
	main(hf, audio_info, f_type,capl=capl, d_w2v=d_w2v, output_dim=output_dim,
		feature_shape=feature_shape,lr=lr,
		batch_size=batch_size,total_epoch=total_epoch,
		file='/home/xyj/usr/local/data/msrvtt',pretrained_model=None)
	

	
	
	
	


	
