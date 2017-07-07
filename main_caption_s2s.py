import numpy as np
import os
import h5py
import math

from utils import DataUtil
from model import CaptionModel 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import cPickle as pickle
import time
import json


		
def exe_train(sess, data, batch_size, v2i, hf, feature_shape, 
	train, loss, input_video, input_captions, y, capl=16):

	np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_loss = 0.0
	for batch_idx in xrange(num_batch):
		# if batch_idx < 100:
		batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]

		data_v = DataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = DataUtil.getNewBatchTrainCaption(batch_caption, v2i, capl=capl)

		_, l = sess.run([train,loss],feed_dict={input_video:data_v, input_captions:data_c,  y:data_y})
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
		
		data_v = DataUtil.getBatchVideoFeature(batch_caption,hf,feature_shape)
		data_c, data_y = DataUtil.getBatchTestCaption(batch_caption, v2i, capl=capl)
		[gw] = sess.run([predict_words],feed_dict={input_video:data_v, input_captions:data_c, y:data_y})

		generated_captions = DataUtil.convertCaptionI2V(batch_caption, gw, i2v)

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


def main(hf,f_type,capl=16, d_w2v=512, output_dim=512,
		feature_shape=None,lr=0.01,
		batch_size=64,total_epoch=100,
		file=None,pretrained_model=None):
	'''
		capl: the length of caption
	'''

	# Create vocabulary
	v2i, train_data, val_data, test_data = DataUtil.create_vocabulary_word2vec(file, capl=capl, v2i={'': 0, 'UNK':1,'BOS':2, 'EOS':3})

	i2v = {i:v for v,i in v2i.items()}

	print('building model ...')
	voc_size = len(v2i)

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_captions = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
	y = tf.placeholder(tf.int32,shape=(None, capl,len(v2i)))

	captionModel = CaptionModel.AttentionCaptionModel(input_video, input_captions, voc_size, d_w2v, output_dim)
	predict_score, predict_words = captionModel.build_model()
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict_score)
	loss = tf.reduce_mean(loss)+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	# optimizer = tf.train.RMSPropOptimizer(lr,decay=0.9, momentum=0.0, epsilon=1e-8)
	# train = optimizer.minimize(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
	

	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_global_norm([grad], 10)[0][0], var) for grad, var in gvs ]
	train = optimizer.apply_gradients(capped_gvs)
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
			# # shuffle
			print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
			# # train phase
			tic = time.time()
			total_loss = exe_train(sess, train_data, batch_size, v2i, hf, feature_shape, train, loss, input_video, input_captions, y, capl=capl)

			print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

			tic = time.time()
			js = exe_test(sess, test_data, batch_size, v2i, i2v, hf, feature_shape, 
										predict_words, input_video, input_captions, y, capl=capl)
			print('    --Val--, .......Time:%.3f' %(time.time()-tic))

			

			#save model
			export_path = '/home/xyj/usr/local/saved_model/caption/s2s'+'_'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])
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
	timesteps_v=40 # sequences length for video
	# hight = 7
	# width = 7
	feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'ResNet'
	# # feature_path = '/home/xyj/usr/local/data/in5b-'+str(timesteps_v)+'fpv.h5'
	# feature_path = '/data/xyj/youtube-resnet-pool5-'+str(timesteps_v)+'fpv.h5'
	f_type = 'ResNet-pool5'
	feature_path = '/data/xyj/ResNet152-pool5-f'+str(timesteps_v)+'.h5'

	
	# video_feature_dims=1024
	# timesteps_v=10 # sequences length for video
	# # hight = 7
	# # width = 7
	# feature_shape = (timesteps_v,video_feature_dims)

	# f_type = 'GoogleNet'
	# # feature_path = '/home/xyj/usr/local/data/in5b-'+str(timesteps_v)+'fpv.h5'
	# feature_path = '/home/lyb/XYJ/YouTube2Text/feature/pool5_7x7_s1-'+str(timesteps_v)+'f.h5'

	'''
	---------------------------------
	'''
	hf = h5py.File(feature_path,'r')
	# print(hf.keys())

	pretrained_model = None
	
	main(hf,f_type,capl=16, d_w2v=512, output_dim=512,
		feature_shape=feature_shape,lr=lr,
		batch_size=64,total_epoch=30,
		file='./data',pretrained_model=None)
	

	
	
	
	


	