import numpy as np
import os
import h5py
import math

from utils import SeqVladDataUtil
from utils import DataUtil
from model import SeqVladModel 
from utils import CenterUtil

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import cPickle as pickle
import time
import json

import argparse
		
def exe_train(sess, data, epoch, batch_size, v2i, hf, feature_shape, 
	train, loss, input_video, input_captions, y, merged, train_writer, 
	bidirectional=False, step=False, capl=16):
	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):

		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		tic = time.time()
		
		if step:
			data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,(40,feature_shape[1],7,7))
			interval = np.random.randint(1,5)
			data_v = data_v[:,0::interval][:,0:10]
			# data_v = data_v[:,0::interval]
			# start = np.random.randint(0,data_v.shape[1]-10+1)
			# data_v = data_v[:,start:start+10]
		else:
			data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		if bidirectional:
			flag = np.random.randint(0,2)
			if flag==1:
				data_v = data_v[:,::-1]
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
	predict_words, input_video, input_captions, y, step=False, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))+1

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		if step:
			data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,(40,feature_shape[1],7,7))
			data_v = data_v[:,0::4]
		else:
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
	predict_words, input_video, input_captions, y, finished_beam, logprobs_finished_beams, past_symbols, step=False, capl=16):
	
	caption_output = []
	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	for batch_idx in xrange(num_batch):
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		if step:
			data_v = SeqVladDataUtil.getBatchVideoFeature(batch_caption,hf,(40,feature_shape[1],7,7))
			data_v = data_v[:,0::4]
		else:
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
		step=False,
		bidirectional=False,
		reduction_dim=512,
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



	print('building model ...')
	voc_size = len(v2i)

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_captions = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
	y = tf.placeholder(tf.int32,shape=(None, capl))

	attentionCaptionModel = SeqVladModel.SeqVladWithReduAttentionModel(input_video, input_captions, voc_size, d_w2v, output_dim,
								reduction_dim=reduction_dim,
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
	config.gpu_options.per_process_gpu_memory_fraction = 0.6
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

	print('building writer')
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
			total_loss = exe_train(sess, train_data, epoch, batch_size, v2i, hf, feature_shape, train, loss, input_video, input_captions, y, 
				merged, train_writer, bidirectional=bidirectional, step=step, capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			# tic = time.time()
			# js = exe_test(sess, test_data, batch_size, v2i, i2v, hf, feature_shape, 
			# 							predict_words, input_video, input_captions, y, step=step, capl=capl)
			# print('    --Val--, .......Time:%.3f' %(time.time()-tic))


			# #do beamsearch
			tic = time.time()
			js = beamsearch_exe_test(sess, test_data, 1, v2i, i2v, hf, feature_shape, 
										predict_words, input_video, input_captions, y, finished_beam, logprobs_finished_beams, past_symbols, step=step, capl=capl)

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
		
def parseArguments():
	parser = argparse.ArgumentParser(description='seqvlad, youtube, video captioning, reduction app')
	
	parser.add_argument('--feature', type=str, default='google',
							help='google or resnet')

	parser.add_argument('--step', action='store_true',
							help='step training')
	parser.add_argument('--bidirectional', action='store_true',
							help='bidirectional training')

	parser.add_argument('--gpu_id', type=str, default="0",
							help='specify gpu id')

	parser.add_argument('--lr', type=float, default=0.0001,
							help='learning reate')
	parser.add_argument('--epoch', type=int, default=20,
							help='total runing epoch')
	parser.add_argument('--d_w2v', type=int, default=512,
							help='the dimension of word 2 vector')
	parser.add_argument('--output_dim', type=int, default=512,
							help='the hidden size')
	parser.add_argument('--centers_num', type=int, default=16,
							help='the number of centers')
	parser.add_argument('--reduction_dim', type=int, default=512,
							help='the reduction dim of input feature, e.g., 1024->512')

	

	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parseArguments()
	print(args)
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
	lr = args.lr
	epoch = args.epoch

	d_w2v = args.d_w2v
	output_dim = args.output_dim

	reduction_dim=args.reduction_dim
	centers_num = args.centers_num
	bidirectional = args.bidirectional
	step = args.step

	feature = args.feature

	kernel_size = 3
	
	capl = 16
	activation = 'tanh' ## can be one of 'tanh,softmax,relu,sigmoid'
	'''
	---------------------------------
	'''
	if feature=='google':
		video_feature_dims = 1024
		timesteps_v = 10 # sequences length for video
		height = 7
		width = 7
		feature_shape = (timesteps_v,video_feature_dims,height,width)
		f_type = str(activation)+'_seqvlad_attention_'+feature+'_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)+'_k'+str(kernel_size)+'_c'+str(centers_num)+'_redu'+str(reduction_dim)
		if step:
			timesteps_v = 40
		feature_path = '/data/xyj/in5b-'+str(timesteps_v)+'fpv.h5'
	'''
	---------------------------------
	'''
	if feature=='resnet':
		video_feature_dims = 2048
		timesteps_v = 10 # sequences length for video
		height = 7
		width = 7
		feature_shape = (timesteps_v,video_feature_dims,height,width)
		f_type = str(activation)+'_seqvlad_attention_'+feature+'_dw2v'+str(d_w2v)+'_outputdim'+str(output_dim)+'_k'+str(kernel_size)+'_c'+str(centers_num)+'_redu'+str(reduction_dim)
		if step:
			timesteps_v = 40
		feature_path = '/data/xyj/ResNet200-res5c-relu-f'+str(timesteps_v)+'.h5'
	'''
	---------------------------------
	'''
	if step:
		
		f_type = 'step_'+ f_type
	
	if bidirectional:
		f_type = 'bi_'+ f_type

	f_type = 'new_'+f_type
	# feature_path = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	# feature_path = '/home/xyj/usr/local/data/youtube/in5b-'+str(timesteps_v)+'fpv.h5'

	
	
	hf = h5py.File(feature_path,'r')

	# pretrained_model = '/home/xyj/usr/local/saved_model/youtube/seqvlad_withinit_attention_google_dw2v512_outputdim512_k1_c16/lr0.0001_f10_B64/model/E8_L1.93834684881.ckpt'
	
	main(hf,f_type, 
		step=step,
		bidirectional=bidirectional,
		reduction_dim=reduction_dim,
		activation=activation,
		centers_num=centers_num, kernel_size=kernel_size, capl=capl, d_w2v=d_w2v, output_dim=output_dim,
		feature_shape=feature_shape,lr=lr,
		batch_size=64,total_epoch=epoch,
		file='./data',pretrained_model=None)
	

	
	
	
	


	
