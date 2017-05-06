import numpy as np
import os
import re
import h5py
import math


def create_vocabulary_word2vec(file, capl=None, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	'''
	vocab_file = file+'/vocabulary.txt'
	train_file = file+'/sents_train_lc_nopunc.txt'
	val_file = file+'/sents_val_lc_nopunc.txt'
	test_file = file+'/sents_test_lc_nopunc.txt'
	
	with open(vocab_file, 'r') as voc_f:
		for line in voc_f:
			word = line.strip()
			v2i[word]=len(v2i)

	train_data = []
	val_data = []
	test_data = []
	def parse_file_2_dict(temp_file):
		captions = []
		with open(temp_file, 'r') as voc_f:
			for line in voc_f:
				cap = {}
				temp = line.strip().split('\t')
				words = temp[1].split(' ')
				if len(words)<capl:# and len(words)>=4: 
				# if len(words)<16:
					cap[temp[0]] = words
					captions.append(cap)
		return captions
	def generate_test_data():
		captions = []
		
		for idx in xrange(1300,1971):
			cap = {}
			cap['vid'+str(idx)] = ['']
			captions.append(cap)
		return captions

	train_data = parse_file_2_dict(train_file)
	val_data = parse_file_2_dict(val_file)
	test_data = generate_test_data()

	# v2i = generate_vocab(train_data, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3})
	print('len v2i:',len(v2i))
	return v2i, train_data, val_data, test_data




def generate_vocab(train_data, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}):


	for caption_info in train_data:
		for k,v in caption_info.items():
			for w in v:
				if not v2i.has_key(w):
					v2i[w] = len(v2i)


	print('vocab size %d' %(len(v2i)))
	return v2i
	


def getBatchVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			feature = hf[k]
			input_video[idx] = np.reshape(feature,feature_shape)
	return input_video


def getBatchStepVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	feature_shape = (40,1024)
	step = np.random.randint(1,5)
	# print(step)
	input_video = np.zeros((batch_size,)+tuple((10,1024)),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			feature = hf[k]
			input_video[idx] = np.reshape(feature,feature_shape)[0::step][0:10]
	return input_video

def getBatchBidirectVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			feature = hf[k]
			flag = np.random.randint(0,2)
			if flag==0:
				input_video[idx] = np.reshape(feature,feature_shape)
			else:
				input_video[idx] = np.reshape(feature,feature_shape)[::-1]
	return input_video

def getBatchTrainCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)

	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')

	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']

	for idx, caption in enumerate(batch_caption):
		for vid, sen in caption.items():

			for k, w in enumerate(sen):
				
				if w in v2i.keys():
					labels[idx][k][v2i[w]] = 1
					input_captions[idx][k+1] = v2i[w]
				else:
					labels[idx][k][v2i['UNK']] = 1
					input_captions[idx][k+1] = v2i['UNK']
			# if len(sen)+1<capl:
			# 	input_captions[idx][len(sen)+1] = v2i['EOS']
			labels[idx][len(sen)][v2i['EOS']] = 1
	# print(batch_caption)
	# print(input_captions)
	# print(np.sum(labels,-1))
	return input_captions, labels

def getNewBatchTrainCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)

	

	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']

	for idx, caption in enumerate(batch_caption):
		for vid, sen in caption.items():

			for k, w in enumerate(sen):
				
				if w in v2i.keys():
					input_captions[idx][k+1] = v2i[w]
				else:
					input_captions[idx][k+1] = v2i['UNK']

	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')
	for i, sentence in enumerate(input_captions):
		for j, word in enumerate(sentence):
			if j>=1:
				if word != 0:
					labels[i,j-1,word]=1
				elif word == 0:
					labels[i,j-1,v2i['EOS']]=1
					# break
				# labels[i,j-1,word]=1

	# print(batch_caption)
	# print(input_captions)
	# print(np.sum(labels,-1))
	return input_captions, labels
def getBatchTestCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)
	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')
	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']


	return input_captions, labels


def convertCaptionI2V(batch_caption, generated_captions,i2v):
	captions = []
	for idx, sen in enumerate(generated_captions):
		caption = ''
		for word in sen:
			if i2v[word]=='EOS':
				break
			caption+=i2v[word]+' '
		captions.append(caption)
	return captions
if __name__=='__main__':
	main()