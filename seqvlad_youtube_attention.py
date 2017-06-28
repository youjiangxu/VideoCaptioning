import numpy as np
import os
import h5py
import math

from utils import SeqVladDataUtil
from utils import DataUtil
from model import SeqVladModel 
from utils import CenterUtil

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import cPickle as pickle
import time
import json


		
def exe_train(sess, data, epoch, batch_size, v2i, hf, feature_shape, 
	train, loss, input_video, input_captions, y, merged, train_writer, capl=16):

	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):

		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		tic = time.time()
		data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = SeqVladDataUtil.getBatchTrainCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		data_time = time.time()-tic
		tic = time.time()

		# print('data_v mean:', np.mean(data_v),' std:', np.std(data_v))
		summary, _, l = sess.run([merged,train,loss],feed_dict={input_video:data_v, input_captions:data_c,  y:data_y})

		train_writer.add_summary(summary,epoch*num_batch+batch_idx)

		run_time = time.time()-tic
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f, data_time:%.3f, run_time:%.3f' %(batch_idx+1,num_batch,l,data_time,run_time))
	total_loss = total_loss/num_batch
	return total_loss

def exe_test(sess, data, batch_size, v2i, i2v, hf, feature_shape, 
	predict_words, input_video, input_captions, y, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))+1

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = SeqVladDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		[gw] = sess.run([predict_words],feed_dict={input_video:data_v, input_captions:data_c, y:data_y})

		generated_captions = SeqVladDataUtil.convertCaptionI2V(batch_caption, gw, i2v)

		for idx, sen in enumerate(generated_captions):
			print('%s : %s' %(batch_caption[idx].keys()[0],sen))
			caption_output.append({'image_id':batch_caption[idx].keys()[0],'caption':sen})
	
	js = {}
	js['val_predictions'] = caption_output

	return js

def beamsearch_exe_test(sess, data, batch_size, v2i, i2v, hf, feature_shape, 
	predict_words, input_video, input_captions, y, finished_beam, logprobs_finished_beams, past_symbols, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = SeqVladDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
		[fb, lfb, ps] = sess.run([finished_beam, logprobs_finished_beams, past_symbols],feed_dict={input_video:data_v, input_captions:data_c, y:data_y})

		generated_captions = SeqVladDataUtil.convertCaptionI2V(batch_caption, fb, i2v)

		for idx, sen in enumerate(generated_captions):
			print('%s : %s' %(batch_caption[idx].keys()[0],sen))
			caption_output.append({'image_id':batch_caption[idx].keys()[0],'caption':sen})
	
	js = {}
	js['val_predictions'] = caption_output

	return js

def evaluate_mode_by_shell(res_path,js):
	with open(res_path, 'w') as f:
		json.dump(js, f)

	command ='/home/xyj/usr/local/tools/caption/caption_eval/call_python_caption_eval.sh '+ res_path
	os.system(command)


def main(hf,f_type,
		activation = 'tanh',
		centers_num = 32, kernel_size=1, capl=16, d_w2v=512, output_dim=512,
		feature_shape=None,lr=0.01,
		batch_size=64,total_epoch=100,
		file=None,pretrained_model=None):
	'''
		capl: the length of caption
	'''

	# Create vocabulary
	v2i, train_data, val_data, test_data = DataUtil.create_vocabulary_word2vec(file, capl=capl,  v2i={'': 0, 'UNK':1,'BOS':2, 'EOS':3})

	i2v = {i:v for v,i in v2i.items()}

	(init_w,init_b,init_centers) = CenterUtil.get_centers('/home/xyj/usr/local/centers/youtube_centers_k'+str(centers_num)+'.pkl', hf, 1024, centers_num)
	print('init_w mean:', np.mean(init_w), ' std:', np.std(init_w))
	print('init_b mean:', np.mean(init_b), ' std:', np.std(init_b))
	print('init_centers mean:', np.mean(init_centers), ' std:', np.std(init_centers))

	print('building model ...')
	voc_size = len(v2i)

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_captions = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
	y = tf.placeholder(tf.int32,shape=(None, capl))

	attentionCaptionModel = SeqVladModel.SeqVladAttentionModel(input_video, input_captions, voc_size, d_w2v, output_dim,
								init_w, init_b, init_centers,
								activation=activation,
								centers_num=centers_num, 
								filter_size=kernel_size,
								done_token=3, max_len = capl, beamsearch_batchsize = 1, beam_size=5)
	predict_score, predict_words, loss_mask, finished_beam, logprobs_finished_beams, past_symbols = attentionCaptionModel.build_model()
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predict_score)

	loss = tf.reduce_sum(loss,reduction_indices=[-1])/tf.reduce_sum(loss_mask,reduction_indices=[-1])


	loss = tf.reduce_mean(loss)+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

	optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
	

	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_global_norm([grad], 10)[0][0], var) for grad, var in gvs ]
	train = optimizer.apply_gradients(capped_gvs)


	tf.summary.scalar('cross_entropy',loss)


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
	
	'''
		tensorboard configure
	'''
	merged = tf.summary.merge_all()
	export_path = '/home/xyj/usr/local/saved_model/youtube/'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])+'_B'+str(batch_size)

	if not os.path.exists(export_path+'/log'):
		os.makedirs(export_path+'/log')
		print('mkdir %s' %export_path+'/log')

	train_writer = tf.summary.FileWriter(export_path + '/log',
                                      sess.graph)


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
			total_loss = exe_train(sess, train_data, epoch, batch_size, v2i, hf, feature_shape, train, loss, input_video, input_captions, y, merged, train_writer, capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			# tic = time.time()
			# js = exe_test(sess, test_data, batch_size, v2i, i2v, hf, feature_shape, 
			# 							predict_words, input_video, input_captions, y, capl=capl)
			# print('    --Val--, .......Time:%.3f' %(time.time()-tic))


			# #do beamsearch
			tic = time.time()
			js = beamsearch_exe_test(sess, test_data, 1, v2i, i2v, hf, feature_shape, 
										predict_words, input_video, input_captions, y, finished_beam, logprobs_finished_beams, past_symbols, capl=capl)

			print('    --Val--, .......Time:%.3f' %(time.time()-tic))

			#save model
			
			if not os.path.exists(export_path+'/model'):
				os.makedirs(export_path+'/model')
				print('mkdir %s' %export_path+'/model')
			if not os.path.exists(export_path+'/res'):
				os.makedirs(export_path+'/res')
				print('mkdir %s' %export_path+'/res')

			# eval
			res_path = export_path+'/res/E'+str(epoch+1)+'.json'
			evaluate_mode_by_shell(res_path,js)


			# save_path = saver.save(sess, export_path+'/model/'+'E'+str(epoch+1)+'_L'+str(total_loss)+'.ckpt')
			# print("Model saved in file: %s" % save_path)
		

if __name__ == '__main__':


	lr = 0.0001

	d_w2v = 512
	output_dim = 512

	epoch = 40

	kernel_size = 3
	centers_num = 32
	capl = 16
	activation = 'tanh' ## can be one of 'tanh,softmax,relu,sigmoid'
	'''
	---------------------------------
	'''
	video_feature_dims = 1024
	timesteps_v = 10 # sequences length for video
	height = 7
	width = 7
	feature_shape = (timesteps_v,video_feature_dims,height,width)

	f_type = 'filterw3_'+str(activation)+'_seqvlad_withoutinit_attention_google_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)+'_k'+str(kernel_size)+'_c'+str(centers_num)+'_capl'+str(capl)
	# feature_path = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	# feature_path = '/home/xyj/usr/local/data/youtube/in5b-'+str(timesteps_v)+'fpv.h5'
	feature_path = '/data/xyj/in5b-'+str(timesteps_v)+'fpv.h5'
	'''
	---------------------------------
	'''
	hf = h5py.File(feature_path,'r')

	# pretrained_model = '/home/xyj/usr/local/saved_model/youtube/seqvlad_withinit_attention_google_dw2v512_outputdim512_k1_c16/lr0.0001_f10_B64/model/E8_L1.93834684881.ckpt'
	
	main(hf,f_type, 
		activation=activation,
		centers_num=centers_num, kernel_size=kernel_size, capl=capl, d_w2v=d_w2v, output_dim=output_dim,
		feature_shape=feature_shape,lr=lr,
		batch_size=64,total_epoch=epoch,
		file='./data',pretrained_model=None)
	

	
	
	
	


	
