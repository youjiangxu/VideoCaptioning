import numpy as np
import os
import re
import h5py
import math
import json
from collections import Counter
def create_vocabulary_word2vec(file, capl=None, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}, word_threshold=2, sen_length=5, num_training=7010):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	limit_sen: the number sentence for training per video
	'''
	json_file = file+'/videodatainfo_2017.json'
	train_info = json.load(open(json_file,'r'))
	videos = train_info['videos']
	sentences = train_info['sentences']
	train_video = [v['video_id'] for v in videos if v['id']<num_training]
	test_video = [v['video_id'] for v in videos if v['id']>=num_training]

	train_data = []
	test_data = []

	

	print('preprocess sentence...')
	for idx, sentence in enumerate(sentences):
		video_id = sentence['video_id']
		caption = sentence['caption'].strip().split(' ')
		# print caption
		if(video_id in train_video):
			if len(caption)<capl and len(caption)>=sen_length:
				train_data.append({video_id:caption})
				
			

	def generate_test_data():
		captions = []
		
		for idx in range(num_training,10000):
			cap = {}
			cap['video'+str(idx)] = ['']
			captions.append(cap)
		return captions

	test_data = generate_test_data()
	print('build vocabulary...')
	all_word = []
	for data in train_data:
		for k,v in data.items():
			all_word.extend(v)
	

	vocab = Counter(all_word)
	vocab = [k for k in vocab.keys() if vocab[k] >= word_threshold]

	# create vocabulary index
	for w in vocab:
		if w not in v2i.keys():
			v2i[w] = len(v2i)

	# new training set and validation set
	
	# if limit_sen is None:

	print('size of vocabulary: %d '%(len(v2i)))
	print('size of train, test: %d, %d' %(len(train_data),len(test_data)))
	return v2i, train_data, test_data	


def get_test_data(file='/home/xyj/usr/local/data/msrvtt'):
	json_file = file+'/test_videodatainfo_nosen_2017.json'
	test_info = json.load(open(json_file,'r'))
	videos = test_info['videos']
	cate_info = {}
	test_data = []
	for idx,video in enumerate(videos):
		cate_info[video['video_id']]=video['category']
		# print(video)
		cap = {}
		cap[video['video_id']] = ['']
		test_data.append(cap)
	return cate_info, test_data

def getBatchVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			vid = int(k[5:])-10000
			feature = hf[vid]
			input_video[idx] = np.reshape(feature,feature_shape)
	return input_video

def getBatchC3DVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			vid = int(k[5:])-10000
			feature = hf[vid]
			input_video[idx] = np.reshape(feature[0:40,:],feature_shape)
	return input_video
	
if __name__=='__main__':
	main()