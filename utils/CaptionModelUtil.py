import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import numpy as np
from sklearn.decomposition import PCA
import ModelUtil
import InitUtil

rng = np.random
rng.seed(1234)


def matmul_wx(x, w, b, output_dims):
	
	return tf.matmul(x, w)+tf.reshape(b,(1,output_dims))
	

def matmul_uh(u,h_tm1):
	return tf.matmul(h_tm1,u)

def init_linear_projection(rng, nrows, ncols, pca_mat=None):
    """ Linear projection (for example when using fixed w2v as LUT """
    if nrows == ncols:
        P = np.eye(nrows)
        print "Linear projection: initialized as identity matrix"
    else:
        assert([nrows, ncols] == pca_mat.shape, 'PCA matrix not of same size as RxC')
        P = 0.1 * pca_mat
        print "Linear projection: initialized with 0.1 PCA"

    return P.astype(np.float32)

def setWord2VecModelConfiguration(v2i,w2v,d_w2v,d_lproj=512,w2v_type='googleNews'):
	'''
		v2i: vocab(word) to int(index)
		w2v: word to vector
		d_w2v:dimension of w2v
		d_lproj: dimension of projection
	'''

	voc_size = len(v2i)
	np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
	T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

	pca_mat = None
	print "Initialize LUTs as word2vec and use linear projection layer"


	LUT = np.zeros((voc_size, d_w2v), dtype='float32')
	found_words = 0

	for w, v in v2i.iteritems():
		if w in w2v.vocab:
			LUT[v] = w2v[w]
			found_words +=1
		else:
			LUT[v] = rng.randn(d_w2v)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

	print "Found %d / %d words" %(found_words, len(v2i))


	# word 0 is blanked out, word 1 is 'UNK'
	LUT[0] = np.zeros((d_w2v))


	# setup LUT!
	T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)


	if d_lproj != LUT.shape[1]:
            pca = PCA(n_components=d_lproj, whiten=True)
            pca_mat = pca.fit_transform(LUT.T)  # 300 x 100?

	# T_B = InitUtil.init_weight_variable((d_w2v,d_lproj),init_method='uniform',name="B")
	T_B = tf.Variable(init_linear_projection(rng, d_w2v, d_lproj, pca_mat), name='B')


	return T_B, T_w2v, T_mask

def getEmbeddingWithWord2Vec(words, T_w2v, T_mask, T_B, d_lproj):
	input_shape = words.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()

	mask =  tf.not_equal(words,0)

	embeded_words = tf.gather(T_w2v,words)*tf.gather(T_mask,words)
	embeded_words = tf.nn.l2_normalize(embeded_words,-1) # ????

	embeded_words = tf.reshape(embeded_words,(-1,T_w2v_shape[-1]))
	embeded_words = tf.matmul(embeded_words,T_B)
	embeded_words = tf.reshape(embeded_words,(-1,input_shape[1],d_lproj))

	# print(embeded_words.get_shape().as_list())
	return embeded_words, mask 






def getVisualEncoder(input_visual, T_w2v, T_B, visual_pca_mat=None, return_sequences=True):
	'''
		input_visual: visual feature, (batch_size, timesteps, feature_dim, height, width)
		T_w2v: word 2 vec (|v|,dim)
		visual_pca_mat : mapping the input feature dimension (e.g., the channels) to d_w2v 
	'''

	input_shape = input_visual.get_shape().as_list()
	
	assert len(input_shape)==5, 'the input rank should be 5, but got'+str(len(input_shape))
	
	T_w2v_shape = T_w2v.get_shape().as_list()
	T_B_shape = T_B.get_shape().as_list()

	axis = [0,1,3,4,2]
	input_visual = tf.transpose(input_visual,perm=axis)
	input_visual = tf.reshape(input_visual,(-1,input_shape[2]))
	# input_visual = tf.nn.l2_normalize(input_visual,-1)

	if visual_pca_mat is not None:
		linear_proj = tf.Variable(0.1*visual_pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = InitUtil.init_weight_variable((input_shape[2],T_w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	input_visual = tf.matmul(input_visual,linear_proj) 
	input_visual = tf.nn.l2_normalize(input_visual,-1)

	T_w2v_cov = tf.matmul(tf.transpose(T_w2v,perm=[1,0]),T_w2v)

	input_visual = tf.matmul(input_visual,T_w2v_cov) # (batch_size*timesteps*height*width, |V|)

	input_visual = tf.reshape(input_visual,(-1,input_shape[1],input_shape[3],input_shape[4],T_w2v_shape[-1]))
	axis = [0,1,4,2,3]
	input_visual = tf.transpose(input_visual,perm=axis)
	
	# can be einput_visualtended to different architecture
	if return_sequences:
		input_visual = tf.reduce_sum(input_visual,reduction_indices=[3,4])
	else:
		input_visual = tf.reduce_sum(input_visual,reduction_indices=[1,3,4])

	input_visual = tf.nn.l2_normalize(input_visual,-1)
	input_visual = tf.reshape(input_visual,(-1,T_w2v_shape[-1]))
	input_visual = tf.matmul(input_visual,T_B)
	input_visual = tf.reshape(input_visual,(-1,input_shape[1],T_B_shape[-1]))
	return input_visual


def getCaptionEncoderDecoder(em_vf, em_cap, T_w2v, T_mask, T_B, mask):
	'''
		em_vf: embeded visual feature, (batch_size, timesteps, embeded_dims)
		em_cap: embeded caption, (batch_size, len_sequences, d_w2v), d_w2v--the dimension of w2v(word to vector)
		T_w2v: word 2 vec (|v|,dim)
	'''
	input_shape = em_vf.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	em_cap_shape = em_cap.get_shape().as_list()

	input_dims = input_shape[-1]
	output_dims = input_dims

	timesteps = em_cap_shape[1]
	print('input visual feature shape: ', input_shape)
	print('input sentence shape: ',em_cap_shape)
	print('T_B shape', T_B.get_shape().as_list())

	assert input_dims==output_dims, 'input_dims != output_dims'
	initial_state = tf.reduce_sum(em_vf,reduction_indices=1)
	initial_state = tf.nn.l2_normalize(initial_state,-1)

	# em_cap = tf.nn.l2_normalize(em_cap,-1)

	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_r")
	W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_z")
	W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_h")

	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_r")
	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_z")
	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_h")

	b_r = InitUtil.init_bias_variable((output_dims,),name="b_q_r")
	b_z = InitUtil.init_bias_variable((output_dims,),name="b_q_z")
	b_h = InitUtil.init_bias_variable((output_dims,),name="b_q_h")


	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")


	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	em_cap = tf.transpose(em_cap, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=em_cap.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(em_cap)
	else:
		input_embedded_words = input_embedded_words.unpack(em_cap)	


	# preprocess mask
	if len(mask.get_shape()) == len(em_cap_shape)-1:
		mask = tf.expand_dims(mask,dim=-1)
	
	mask = tf.transpose(mask,perm=axis)

	input_mask = tf.TensorArray(
		dtype=mask.dtype,
		size=timesteps,
		tensor_array_name='input_mask'
		)

	if hasattr(input_mask, 'unstack'):
		input_mask = input_mask.unstack(mask)
	else:
		input_mask = input_mask.unpack(mask)


	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')


	def train_step(time, train_hidden_state, h_tm1):
		x_t = input_embedded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1
		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

		h = tf.where(tiled_mask_t, h, h_tm1) # (batch_size, output_dims)
		
		train_hidden_state = train_hidden_state.write(time, h)

		return (time+1,train_hidden_state,h)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	train_hidden_state = train_out[1]
	train_last_output = train_out[-1] 
	
	if hasattr(train_hidden_state, 'stack'):
		train_outputs = train_hidden_state.stack()
	else:
		train_outputs = train_hidden_state.pack()

	axis = [1,0] + list(range(2,3))
	train_outputs = tf.transpose(train_outputs,perm=axis)



	train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	# train_outputs = tf.nn.l2_normalize(train_outputs,-1)
	train_outputs = tf.nn.dropout(train_outputs, 0.5)
	predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	# train_outputs = tf.matmul(train_outputs,tf.transpose(T_B))
	# predict_score = tf.matmul(train_outputs,tf.transpose(T_w2v,perm=[1,0]))
	predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# predict_score = tf.nn.softmax(predict_score,-1)
	# test phase


	test_input_embedded_words = tf.TensorArray(
            dtype=em_cap.dtype,
            size=timesteps+1,
            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	test_input_embedded_words = test_input_embedded_words.write(0,em_cap[0])

	def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
		x_t = test_input_embedded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)

		# normed_h = tf.nn.l2_normalize(h,-1)
		# normed_h = tf.matmul(normed_h,tf.transpose(T_B))

		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		# predict_score_t = tf.matmul(normed_h,tf.transpose(T_w2v,perm=[1,0]))
		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)
		predict_word_t = tf.nn.l2_normalize(predict_word_t,-1)
		predict_word_t = tf.matmul(predict_word_t,T_B)

		test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-2]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return predict_score, predict_words


'''
-----------------------------------------direct embedding---------------------------------------
'''
def setDirectWord2VecModelConfiguration(v2i,d_lproj=512):
	'''
		v2i: vocab(word) to int(index)
		d_lproj: dimension of projection
	'''

	voc_size = len(v2i)
	np_mask = np.vstack((np.zeros(d_lproj),np.ones((voc_size-1,d_lproj))))
	T_mask = tf.constant(np_mask, tf.float32, name='T_mask')

	T_w2v = InitUtil.init_weight_variable((voc_size,d_lproj),init_method='uniform',name="T_w2v")

	return T_w2v, T_mask

def getDirectEmbeddingWithWord2Vec(captions, T_w2v, T_mask):
	mask =  tf.not_equal(captions,0)
	embeded_captions = tf.gather(T_w2v,captions)*tf.gather(T_mask,captions)
	return embeded_captions, mask 

def getDirectEmbeddingWithWord2VecWithIntMask(captions, T_w2v, T_mask):
	mask = tf.where(captions==0,tf.zeros_like(captions),tf.ones_like(captions))
	embeded_captions = tf.gather(T_w2v,captions)*tf.gather(T_mask,captions)
	return embeded_captions, mask 

def getDirectVisualEncoder(input_visual, T_w2v, visual_pca_mat=None, return_sequences=True):
	'''
		input_visual: visual feature, (batch_size, timesteps, feature_dim, height, width)
		T_w2v: word 2 vec (|v|,dim)
		visual_pca_mat : mapping the input feature dimension (e.g., the channels) to d_w2v 
	'''

	input_shape = input_visual.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	
	assert len(input_shape)==5, 'the input rank should be 5, but got'+str(len(input_shape))
	
	axis = [0,1,3,4,2]
	input_visual = tf.transpose(input_visual,perm=axis)
	input_visual = tf.reshape(input_visual,(-1,input_shape[2]))

	if visual_pca_mat is not None:
		linear_proj = tf.Variable(0.1*visual_pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = InitUtil.init_weight_variable((input_shape[2],T_w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')


	input_visual = tf.matmul(input_visual,linear_proj) 
	input_visual = tf.nn.l2_normalize(input_visual,-1)

	T_w2v_cov = tf.matmul(tf.transpose(T_w2v,perm=[1,0]),T_w2v)

	input_visual = tf.matmul(input_visual,T_w2v_cov) # (batch_size*timesteps*height*width, |V|)

	input_visual = tf.reshape(input_visual,(-1,input_shape[1],input_shape[3],input_shape[4],T_w2v_shape[-1]))
	axis = [0,1,4,2,3]
	input_visual = tf.transpose(input_visual,perm=axis)
	
	input_visual = tf.reduce_mean(input_visual,reduction_indices=[3,4])

	return input_visual

def getDirectCaptionEncoderDecoder(embedded_visual_feature, embeded_captions, T_w2v, T_mask, mask):
	'''
		embedded_visual_feature: embeded visual feature, (batch_size, timesteps, embeded_dims)
		embeded_captions: embeded caption, (batch_size, len_sequences, d_w2v), d_w2v--the dimension of w2v(word to vector)
		T_w2v: word 2 vec (|v|,dim)
	'''
	input_shape = embedded_visual_feature.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	embeded_captions_shape = embeded_captions.get_shape().as_list()

	input_dims = input_shape[-1]
	output_dims = input_dims

	timesteps = embeded_captions_shape[1]
	print('input visual feature shape: ', input_shape)
	print('input sentence shape: ',embeded_captions_shape)

	assert input_dims==output_dims, 'input_dims != output_dims'
	initial_state = tf.reduce_mean(embedded_visual_feature,reduction_indices=1)
	initial_state = tf.nn.l2_normalize(initial_state,-1)

	# embeded_captions = tf.nn.l2_normalize(embeded_captions,-1)

	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_r")
	W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_z")
	W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_h")

	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_r")
	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_z")
	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_h")

	b_r = InitUtil.init_bias_variable((output_dims,),name="b_r")
	b_z = InitUtil.init_bias_variable((output_dims,),name="b_z")
	b_h = InitUtil.init_bias_variable((output_dims,),name="b_h")


	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='glorot_uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")


	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_captions = tf.transpose(embeded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(embeded_captions)
	else:
		input_embedded_words = input_embedded_words.unpack(embeded_captions)	


	# # preprocess mask
	# if len(mask.get_shape()) == len(embeded_captions_shape)-1:
	# 	mask = tf.expand_dims(mask,dim=-1)
	
	# mask = tf.transpose(mask,perm=axis)

	# input_mask = tf.TensorArray(
	# 	dtype=mask.dtype,
	# 	size=timesteps,
	# 	tensor_array_name='input_mask'
	# 	)

	# if hasattr(input_mask, 'unstack'):
	# 	input_mask = input_mask.unstack(mask)
	# else:
	# 	input_mask = input_mask.unpack(mask)


	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')


	def train_step(time, train_hidden_state, h_tm1):
		x_t = input_embedded_words.read(time) # batch_size * dim
		# mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		
		train_hidden_state = train_hidden_state.write(time, h)

		return (time+1,train_hidden_state,h)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	train_hidden_state = train_out[1]
	train_last_output = train_out[-1] 
	
	if hasattr(train_hidden_state, 'stack'):
		train_outputs = train_hidden_state.stack()
	else:
		train_outputs = train_hidden_state.pack()

	axis = [1,0] + list(range(2,3))
	train_outputs = tf.transpose(train_outputs,perm=axis)



	train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	train_outputs = tf.nn.dropout(train_outputs, 0.5)
	predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# predict_score = tf.nn.softmax(predict_score,-1)

	# # test phase
	# test_input_embedded_words = tf.TensorArray(
 #            dtype=embeded_captions.dtype,
 #            size=timesteps+1,
 #            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	# test_input_embedded_words = test_input_embedded_words.write(0,embeded_captions[0])

	def test_step(time, test_hidden_state, x_t, predict_words, h_tm1):
		# x_t = test_input_embedded_words.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)


		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)
		# predict_word_t =  tf.nn.l2_normalize(predict_word_t,-1)

		# test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, predict_word_t, predict_words, h)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, embeded_captions[0], predict_words, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-2]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return predict_score, predict_words


def get_init_state(x, output_dims):
	initial_state = tf.zeros_like(x)
	initial_state = tf.reduce_sum(initial_state,axis=[1,2])
	initial_state = tf.expand_dims(initial_state,dim=-1)
	initial_state = tf.tile(initial_state,[1,output_dims])
	return initial_state

def getDirectCaptionGRUEncoderDecoder(embeded_feature, embeded_captions, T_w2v, T_mask, mask):
	'''
		embeded_feature: embeded visual feature, (batch_size, timesteps, embeded_dims)
		embeded_captions: embeded caption, (batch_size, len_sequences, d_w2v), d_w2v--the dimension of w2v(word to vector)
		T_w2v: word 2 vec (|v|,dim)
	'''
	input_shape = embeded_feature.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	embeded_captions_shape = embeded_captions.get_shape().as_list()

	input_dims = input_shape[-1]
	output_dims = input_dims

	
	print('input visual feature shape: ', input_shape)
	print('input sentence shape: ',embeded_captions_shape)

	assert input_dims==output_dims, 'input_dims != output_dims'

	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_r")
	W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_z")
	W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_h")

	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_r")
	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_z")
	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_h")

	b_r = InitUtil.init_bias_variable((output_dims,),name="b_q_r")
	b_z = InitUtil.init_bias_variable((output_dims,),name="b_q_z")
	b_h = InitUtil.init_bias_variable((output_dims,),name="b_q_h")


	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")

	'''
		visual feature part
	'''
	timesteps = input_shape[1]
	embeded_feature = tf.nn.l2_normalize(embeded_feature,-1)
	feature_init_state = get_init_state(embeded_feature, output_dims)
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_feature = tf.transpose(embeded_feature, perm=axis)

	input_feature = tf.TensorArray(
            dtype=embeded_feature.dtype,
            size=timesteps,
            tensor_array_name='input_feature')
	if hasattr(input_feature, 'unstack'):
		input_feature = input_feature.unstack(embeded_feature)
	else:
		input_feature = input_feature.unpack(embeded_feature)	


	feature_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='feature_hidden_state')


	def feature_step(time, feature_hidden_state, h_tm1):
		x_t = input_feature.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		feature_hidden_state = feature_hidden_state.write(time, h)

		return (time+1,feature_hidden_state,h)

	

	time = tf.constant(0, dtype='int32', name='time')


	feature_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=feature_step,
            loop_vars=(time, feature_hidden_state, feature_init_state),
            parallel_iterations=32,
            swap_memory=True)

	initial_state = feature_out[-1] 
	



	'''
		caption part code
	'''



	timesteps = embeded_captions_shape[1]
	embeded_captions = tf.nn.l2_normalize(embeded_captions,-1)
	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_captions = tf.transpose(embeded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(embeded_captions)
	else:
		input_embedded_words = input_embedded_words.unpack(embeded_captions)	



	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')


	def train_step(time, train_hidden_state, h_tm1):
		x_t = input_embedded_words.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		
		train_hidden_state = train_hidden_state.write(time, h)

		return (time+1,train_hidden_state,h)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	train_hidden_state = train_out[1]
	train_last_output = train_out[-1] 
	
	if hasattr(train_hidden_state, 'stack'):
		train_outputs = train_hidden_state.stack()
	else:
		train_outputs = train_hidden_state.pack()

	axis = [1,0] + list(range(2,3))
	train_outputs = tf.transpose(train_outputs,perm=axis)



	train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	train_outputs = tf.nn.dropout(train_outputs, 0.5)
	predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# test phase


	test_input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps+1,
            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	test_input_embedded_words = test_input_embedded_words.write(0,embeded_captions[0])

	def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
		x_t = test_input_embedded_words.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)


		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)
		predict_word_t =  tf.nn.l2_normalize(predict_word_t,-1)

		test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-2]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return predict_score, predict_words




'''
	sequence 2 sequence caption 

'''
def getSequence2SequenceCaption(input_video, embeded_captions, T_w2v, T_mask):

	input_video = tf.reduce_mean(input_video,reduction_indices=[3,4])
	input_shape = input_video.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	embeded_captions_shape = embeded_captions.get_shape().as_list()

	input_dims = input_shape[-1]
	output_dims = embeded_captions_shape[-1]

	
	# visula encoder 
	W_v_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_r")
	W_v_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_z")
	W_v_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_h")

	U_v_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_r")
	U_v_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_z")
	U_v_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_h")

	b_v_r = InitUtil.init_bias_variable((output_dims,),name="b_v_r")
	b_v_z = InitUtil.init_bias_variable((output_dims,),name="b_v_z")
	b_v_h = InitUtil.init_bias_variable((output_dims,),name="b_v_h")

	timesteps_v = input_shape[1]
	feature_init_state = get_init_state(input_video, output_dims)

	axis = [1,0]+list(range(2,3))  
	input_video = tf.transpose(input_video, perm=axis)

	input_feature = tf.TensorArray(
            dtype=input_video.dtype,
            size=timesteps_v,
            tensor_array_name='input_feature')
	if hasattr(input_feature, 'unstack'):
		input_feature = input_feature.unstack(input_video)
	else:
		input_feature = input_feature.unpack(input_video)

	feature_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps_v,
            tensor_array_name='feature_hidden_state')

	def feature_step(time, feature_hidden_state, h_tm1):
		x_t = input_feature.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_v_r, b_v_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_v_z, b_v_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_v_h, b_v_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_v_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_v_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_v_h,r*h_tm1))
		
		h = (1-z)*hh + z*h_tm1

		feature_hidden_state = feature_hidden_state.write(time, h)

		return (time+1,feature_hidden_state,h)

	time = tf.constant(0, dtype='int32', name='time')

	feature_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps_v,
            body=feature_step,
            loop_vars=(time, feature_hidden_state, feature_init_state),
            parallel_iterations=32,
            swap_memory=True)

	initial_state = feature_out[-1]

	'''
		Dense layer
	'''

	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")


	'''
		caption part code
	'''

	# visula encoder 
	W_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_r")
	W_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_z")
	W_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_h")

	U_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_r")
	U_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_z")
	U_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_h")

	b_c_r = InitUtil.init_bias_variable((output_dims,),name="b_c_r")
	b_c_z = InitUtil.init_bias_variable((output_dims,),name="b_c_z")
	b_c_h = InitUtil.init_bias_variable((output_dims,),name="b_c_h")


	timesteps = embeded_captions_shape[1]
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_captions = tf.transpose(embeded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(embeded_captions)
	else:
		input_embedded_words = input_embedded_words.unpack(embeded_captions)	



	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')


	def train_step(time, train_hidden_state, h_tm1):
		x_t = input_embedded_words.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		
		train_hidden_state = train_hidden_state.write(time, h)

		return (time+1,train_hidden_state,h)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	train_hidden_state = train_out[1]
	train_last_output = train_out[-1] 
	
	if hasattr(train_hidden_state, 'stack'):
		train_outputs = train_hidden_state.stack()
	else:
		train_outputs = train_hidden_state.pack()

	axis = [1,0] + list(range(2,3))
	train_outputs = tf.transpose(train_outputs,perm=axis)



	train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	train_outputs = tf.nn.dropout(train_outputs, 0.5)
	predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# test phase


	test_input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps+1,
            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	test_input_embedded_words = test_input_embedded_words.write(0,embeded_captions[0])

	def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
		x_t = test_input_embedded_words.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)


		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)

		test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-2]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return predict_score, predict_words


def hard_sigmoid(x):
	x = (0.2 * x) + 0.5
	x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),tf.cast(1., dtype=tf.float32))
	return x

def getSequence2SequenceCaptionWithAttention(input_video, embeded_captions, T_w2v, T_mask):

	# input_video = tf.reduce_mean(input_video,reduction_indices=[3,4])

	ori_input_feature = input_video


	input_shape = input_video.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	embeded_captions_shape = embeded_captions.get_shape().as_list()

	input_dims = input_shape[-1]
	output_dims = embeded_captions_shape[-1]

	
	# visula encoder 
	W_v_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_r")
	W_v_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_z")
	W_v_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_h")

	U_v_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_r")
	U_v_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_z")
	U_v_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_h")

	b_v_r = InitUtil.init_bias_variable((output_dims,),name="b_v_r")
	b_v_z = InitUtil.init_bias_variable((output_dims,),name="b_v_z")
	b_v_h = InitUtil.init_bias_variable((output_dims,),name="b_v_h")

	timesteps_v = input_shape[1]
	feature_init_state = get_init_state(input_video, output_dims)

	axis = [1,0]+list(range(2,3))  
	input_video = tf.transpose(input_video, perm=axis)

	input_feature = tf.TensorArray(
            dtype=input_video.dtype,
            size=timesteps_v,
            tensor_array_name='input_feature')
	if hasattr(input_feature, 'unstack'):
		input_feature = input_feature.unstack(input_video)
	else:
		input_feature = input_feature.unpack(input_video)

	feature_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps_v,
            tensor_array_name='feature_hidden_state')

	def feature_step(time, feature_hidden_state, h_tm1):
		x_t = input_feature.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_v_r, b_v_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_v_z, b_v_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_v_h, b_v_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_v_r,h_tm1))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_v_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_v_h,r*h_tm1))
		
		h = (1-z)*hh + z*h_tm1

		feature_hidden_state = feature_hidden_state.write(time, h)

		return (time+1,feature_hidden_state,h)

	time = tf.constant(0, dtype='int32', name='time')

	feature_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps_v,
            body=feature_step,
            loop_vars=(time, feature_hidden_state, feature_init_state),
            parallel_iterations=32,
            swap_memory=True)

	initial_state = feature_out[-1]
	# feature_hidden_state = feature_out[-2]
	# if hasattr(feature_hidden_state, 'stack'):
	# 	initial_state = feature_hidden_state.stack()
	# else:
	# 	initial_state = feature_hidden_state.pack()


	# initial_state = tf.reduce_sum(initial_state,reduction_indices=0)

	'''
		Dense layer
	'''

	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='glorot_uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")


	'''
		caption part code
	'''

	# visula encoder 
	W_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_r")
	W_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_z")
	W_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_h")

	U_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_r")
	U_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_z")
	U_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_h")

	b_c_r = InitUtil.init_bias_variable((output_dims,),name="b_c_r")
	b_c_z = InitUtil.init_bias_variable((output_dims,),name="b_c_z")
	b_c_h = InitUtil.init_bias_variable((output_dims,),name="b_c_h")

	# attention
	attention_dim = 100
	W_a = InitUtil.init_weight_variable((input_dims,attention_dim),init_method='glorot_uniform',name="W_a")
	U_a = InitUtil.init_weight_variable((output_dims,attention_dim),init_method='orthogonal',name="U_a")
	b_a = InitUtil.init_bias_variable((attention_dim,),name="b_a")

	W = InitUtil.init_weight_variable((attention_dim,1),init_method='glorot_uniform',name="W")

	A_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_z")

	A_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_r")

	A_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_h")



	timesteps = embeded_captions_shape[1]
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_captions = tf.transpose(embeded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(embeded_captions)
	else:
		input_embedded_words = input_embedded_words.unpack(embeded_captions)	



	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')

	train_predict_scores = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_predict_scores')

	


	def train_step(time, train_hidden_state, h_tm1, ori_input_feature, train_predict_scores):
		x_t = input_embedded_words.read(time) # batch_size * dim

		ori_feature = tf.reshape(ori_input_feature,(-1,input_dims))
		attend_wx = tf.reshape(matmul_wx(ori_feature, W_a, b_a, attention_dim),(-1,timesteps_v,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(matmul_uh(U_a, h_tm1),dim=1),[1,timesteps_v,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

		attend_fea = ori_input_feature * tf.tile(attend_e,[1,1,input_dims])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1) + matmul_uh(A_r,attend_fea))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1) + matmul_uh(A_z,attend_fea))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1) + matmul_uh(A_h,attend_fea))

		
		h = (1-z)*hh + z*h_tm1


		train_hidden_state = train_hidden_state.write(time, h)

		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
		train_predict_scores = train_predict_scores.write(time,predict_score_t)




		return (time+1,train_hidden_state,h, ori_input_feature,train_predict_scores)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state,ori_input_feature,train_predict_scores),
            parallel_iterations=32,
            swap_memory=True)


	# train_hidden_state = train_out[1]
	
	# train_last_output = train_out[-2] 
	
	# if hasattr(train_hidden_state, 'stack'):
	# 	train_outputs = train_hidden_state.stack()
	# else:
	# 	train_outputs = train_hidden_state.pack()

	# axis = [1,0] + list(range(2,3))
	# train_outputs = tf.transpose(train_outputs,perm=axis)
	train_predict_scores = train_out[-1]
	if hasattr(train_predict_scores, 'stack'):
		predict_score = train_predict_scores.stack()
	else:
		predict_score = train_predict_scores.pack()

	axis = [1,0] + list(range(2,3))
	predict_score = tf.transpose(predict_score,perm=axis)


	# train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	# train_outputs = tf.nn.dropout(train_outputs, 0.5)
	# predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	# predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# test phase


	test_input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps+1,
            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	test_input_embedded_words = test_input_embedded_words.write(0,embeded_captions[0])

	def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1, ori_input_feature):
		x_t = test_input_embedded_words.read(time) # batch_size * dim

		ori_feature = tf.reshape(ori_input_feature,(-1,input_dims))
		attend_wx = tf.reshape(matmul_wx(ori_feature, W_a, b_a, attention_dim),(-1,timesteps_v,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(matmul_uh(U_a, h_tm1),dim=1),[1,timesteps_v,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

		attend_fea = ori_input_feature * tf.tile(attend_e,[1,1,input_dims])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1) + matmul_uh(A_r,attend_fea))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1) + matmul_uh(A_z,attend_fea))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1) + matmul_uh(A_h,attend_fea))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)


		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)

		test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h, ori_input_feature)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state, ori_input_feature),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-3]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return predict_score, predict_words


def getS2SCaptionWithAttentionWithMaskLoss(input_video, embeded_captions, T_w2v, T_mask, mask, labels):

	# input_video = tf.reduce_mean(input_video,reduction_indices=[3,4])

	ori_input_feature = input_video


	input_shape = input_video.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	embeded_captions_shape = embeded_captions.get_shape().as_list()

	input_dims = input_shape[-1]
	output_dims = embeded_captions_shape[-1]

	
	# visula encoder 
	W_v_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_r")
	W_v_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_z")
	W_v_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_h")

	U_v_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_r")
	U_v_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_z")
	U_v_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_h")

	b_v_r = InitUtil.init_bias_variable((output_dims,),name="b_v_r")
	b_v_z = InitUtil.init_bias_variable((output_dims,),name="b_v_z")
	b_v_h = InitUtil.init_bias_variable((output_dims,),name="b_v_h")

	timesteps_v = input_shape[1]
	feature_init_state = get_init_state(input_video, output_dims)

	axis = [1,0]+list(range(2,3))  
	input_video = tf.transpose(input_video, perm=axis)

	input_feature = tf.TensorArray(
            dtype=input_video.dtype,
            size=timesteps_v,
            tensor_array_name='input_feature')
	if hasattr(input_feature, 'unstack'):
		input_feature = input_feature.unstack(input_video)
	else:
		input_feature = input_feature.unpack(input_video)

	feature_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps_v,
            tensor_array_name='feature_hidden_state')

	def feature_step(time, feature_hidden_state, h_tm1):
		x_t = input_feature.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_v_r, b_v_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_v_z, b_v_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_v_h, b_v_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_v_r,h_tm1))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_v_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_v_h,r*h_tm1))
		
		h = (1-z)*hh + z*h_tm1

		feature_hidden_state = feature_hidden_state.write(time, h)

		return (time+1,feature_hidden_state,h)

	time = tf.constant(0, dtype='int32', name='time')

	feature_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps_v,
            body=feature_step,
            loop_vars=(time, feature_hidden_state, feature_init_state),
            parallel_iterations=32,
            swap_memory=True)

	initial_state = feature_out[-1]


	'''
		Dense layer
	'''

	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='glorot_uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")


	'''
		caption part code
	'''

	# visula encoder 
	W_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_r")
	W_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_z")
	W_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_h")

	U_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_r")
	U_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_z")
	U_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_h")

	b_c_r = InitUtil.init_bias_variable((output_dims,),name="b_c_r")
	b_c_z = InitUtil.init_bias_variable((output_dims,),name="b_c_z")
	b_c_h = InitUtil.init_bias_variable((output_dims,),name="b_c_h")

	# attention
	attention_dim = 100
	W_a = InitUtil.init_weight_variable((input_dims,attention_dim),init_method='glorot_uniform',name="W_a")
	U_a = InitUtil.init_weight_variable((output_dims,attention_dim),init_method='orthogonal',name="U_a")
	b_a = InitUtil.init_bias_variable((attention_dim,),name="b_a")

	W = InitUtil.init_weight_variable((attention_dim,1),init_method='glorot_uniform',name="W")

	A_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_z")

	A_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_r")

	A_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_h")



	timesteps = embeded_captions_shape[1]
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_captions = tf.transpose(embeded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(embeded_captions)
	else:
		input_embedded_words = input_embedded_words.unpack(embeded_captions)	



	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')

	train_loss = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_loss')


	axis = [1,0]
	print(mask.get_shape().as_list())
	mask = tf.transpose(mask,perm=axis)
	axis = [1,0,2]
	labels = tf.transpose(labels,perm=axis)

	input_mask = tf.TensorArray(
		dtype=mask.dtype,
		size=timesteps,
		tensor_array_name='input_mask')

	if hasattr(input_mask, 'unstack'):
		input_mask = input_mask.unstack(mask)
	else:
		input_mask = input_mask.unpack(mask)	
	
	input_labels = tf.TensorArray(
		dtype=labels.dtype,
		size=timesteps,
		tensor_array_name='input_labels')

	if hasattr(input_mask, 'unstack'):
		input_labels = input_labels.unstack(labels)
	else:
		input_labels = input_labels.unpack(labels)	


	def train_step(time, train_hidden_state, h_tm1, ori_input_feature, train_loss):
		x_t = input_embedded_words.read(time) # batch_size * dim

		mask = input_mask.read(time) 
		label = input_labels.read(time) 

		ori_feature = tf.reshape(ori_input_feature,(-1,input_dims))
		attend_wx = tf.reshape(matmul_wx(ori_feature, W_a, b_a, attention_dim),(-1,timesteps_v,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(matmul_uh(U_a, h_tm1),dim=1),[1,timesteps_v,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

		attend_fea = ori_input_feature * tf.tile(attend_e,[1,1,input_dims])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1) + matmul_uh(A_r,attend_fea))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1) + matmul_uh(A_z,attend_fea))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1) + matmul_uh(A_h,attend_fea))

		
		h = (1-z)*hh + z*h_tm1


		train_hidden_state = train_hidden_state.write(time, h)

		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict_score_t)
		cross_entropy = tf.cast(mask,tf.float32)*cross_entropy
		current_loss = tf.reduce_sum(cross_entropy)

		train_loss = train_loss.write(time,current_loss)




		return (time+1,train_hidden_state,h, ori_input_feature,train_loss)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state,ori_input_feature,train_loss),
            parallel_iterations=32,
            swap_memory=True)



	train_loss = train_out[-1]
	if hasattr(train_loss, 'stack'):
		train_loss = train_loss.stack()
	else:
		train_loss = train_loss.pack()

	train_loss = tf.reduce_sum(train_loss)/tf.cast(tf.reduce_sum(mask),tf.float32)


	# train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	# train_outputs = tf.nn.dropout(train_outputs, 0.5)
	# predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	# predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# test phase


	test_input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps+1,
            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	test_input_embedded_words = test_input_embedded_words.write(0,embeded_captions[0])

	def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1, ori_input_feature):
		x_t = test_input_embedded_words.read(time) # batch_size * dim

		ori_feature = tf.reshape(ori_input_feature,(-1,input_dims))
		attend_wx = tf.reshape(matmul_wx(ori_feature, W_a, b_a, attention_dim),(-1,timesteps_v,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(matmul_uh(U_a, h_tm1),dim=1),[1,timesteps_v,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

		attend_fea = ori_input_feature * tf.tile(attend_e,[1,1,input_dims])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1) + matmul_uh(A_r,attend_fea))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1) + matmul_uh(A_z,attend_fea))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1) + matmul_uh(A_h,attend_fea))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)


		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)

		test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h, ori_input_feature)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state, ori_input_feature),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-3]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return train_loss, predict_words


def getDirectCaptionEncoderDecoderWithAttention(input_video, ori_video, embeded_captions, T_w2v, T_mask):
	ori_input_feature = tf.reduce_mean(ori_video,reduction_indices=[3,4])


	input_shape = input_video.get_shape().as_list()
	T_w2v_shape = T_w2v.get_shape().as_list()
	embeded_captions_shape = embeded_captions.get_shape().as_list()

	ori_video_shape = ori_video.get_shape().as_list()
	print('ori_video_shape:',ori_video_shape)
	ori_input_dim = ori_video_shape[2]


	input_dims = input_shape[-1]
	output_dims = embeded_captions_shape[-1]

	
	# visula encoder 
	W_v_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_r")
	W_v_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_z")
	W_v_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_h")

	U_v_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_r")
	U_v_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_z")
	U_v_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_h")

	b_v_r = InitUtil.init_bias_variable((output_dims,),name="b_v_r")
	b_v_z = InitUtil.init_bias_variable((output_dims,),name="b_v_z")
	b_v_h = InitUtil.init_bias_variable((output_dims,),name="b_v_h")

	timesteps_v = input_shape[1]
	feature_init_state = get_init_state(input_video, output_dims)

	axis = [1,0]+list(range(2,3))  
	input_video = tf.transpose(input_video, perm=axis)

	input_feature = tf.TensorArray(
            dtype=input_video.dtype,
            size=timesteps_v,
            tensor_array_name='input_feature')
	if hasattr(input_feature, 'unstack'):
		input_feature = input_feature.unstack(input_video)
	else:
		input_feature = input_feature.unpack(input_video)

	feature_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps_v,
            tensor_array_name='feature_hidden_state')

	def feature_step(time, feature_hidden_state, h_tm1):
		x_t = input_feature.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_v_r, b_v_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_v_z, b_v_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_v_h, b_v_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_v_r,h_tm1))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_v_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_v_h,r*h_tm1))
		
		h = (1-z)*hh + z*h_tm1

		feature_hidden_state = feature_hidden_state.write(time, h)

		return (time+1,feature_hidden_state,h)

	time = tf.constant(0, dtype='int32', name='time')

	feature_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps_v,
            body=feature_step,
            loop_vars=(time, feature_hidden_state, feature_init_state),
            parallel_iterations=32,
            swap_memory=True)

	initial_state = feature_out[-1]

	'''
		Dense layer
	'''

	W_c = InitUtil.init_weight_variable((output_dims,T_w2v_shape[0]),init_method='glorot_uniform',name='W_c')
	b_c = InitUtil.init_bias_variable((T_w2v_shape[0],),name="b_c")


	'''
		caption part code
	'''

	# visula encoder 
	W_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_r")
	W_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_z")
	W_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='glorot_uniform',name="W_c_h")

	U_c_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_r")
	U_c_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_z")
	U_c_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_c_h")

	b_c_r = InitUtil.init_bias_variable((output_dims,),name="b_c_r")
	b_c_z = InitUtil.init_bias_variable((output_dims,),name="b_c_z")
	b_c_h = InitUtil.init_bias_variable((output_dims,),name="b_c_h")

	# attention
	attention_dim = 100
	W_a = InitUtil.init_weight_variable((ori_input_dim,attention_dim),init_method='glorot_uniform',name="W_a")
	U_a = InitUtil.init_weight_variable((output_dims,attention_dim),init_method='orthogonal',name="U_a")
	b_a = InitUtil.init_bias_variable((attention_dim,),name="b_a")

	W = InitUtil.init_weight_variable((attention_dim,1),init_method='glorot_uniform',name="W")

	A_z = InitUtil.init_weight_variable((ori_input_dim,output_dims),init_method='orthogonal',name="A_z")

	A_r = InitUtil.init_weight_variable((ori_input_dim,output_dims),init_method='orthogonal',name="A_r")

	A_h = InitUtil.init_weight_variable((ori_input_dim,output_dims),init_method='orthogonal',name="A_h")



	timesteps = embeded_captions_shape[1]
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_captions = tf.transpose(embeded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps,
            tensor_array_name='input_embedded_words')


	if hasattr(input_embedded_words, 'unstack'):
		input_embedded_words = input_embedded_words.unstack(embeded_captions)
	else:
		input_embedded_words = input_embedded_words.unpack(embeded_captions)	



	train_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_hidden_state')

	train_predict_scores = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='train_predict_scores')

	


	def train_step(time, train_hidden_state, h_tm1, ori_input_feature, train_predict_scores):
		x_t = input_embedded_words.read(time) # batch_size * dim

		ori_feature = tf.reshape(ori_input_feature,(-1,ori_input_dim))
		attend_wx = tf.reshape(matmul_wx(ori_feature, W_a, b_a, attention_dim),(-1,timesteps_v,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(matmul_uh(U_a, h_tm1),dim=1),[1,timesteps_v,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

		attend_fea = ori_input_feature * tf.tile(attend_e,[1,1,ori_input_dim])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1) + matmul_uh(A_r,attend_fea))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1) + matmul_uh(A_z,attend_fea))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1) + matmul_uh(A_h,attend_fea))

		
		h = (1-z)*hh + z*h_tm1


		train_hidden_state = train_hidden_state.write(time, h)

		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
		train_predict_scores = train_predict_scores.write(time,predict_score_t)




		return (time+1,train_hidden_state,h, ori_input_feature,train_predict_scores)

	

	time = tf.constant(0, dtype='int32', name='time')


	train_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=train_step,
            loop_vars=(time, train_hidden_state, initial_state,ori_input_feature,train_predict_scores),
            parallel_iterations=32,
            swap_memory=True)


	# train_hidden_state = train_out[1]
	
	# train_last_output = train_out[-2] 
	
	# if hasattr(train_hidden_state, 'stack'):
	# 	train_outputs = train_hidden_state.stack()
	# else:
	# 	train_outputs = train_hidden_state.pack()

	# axis = [1,0] + list(range(2,3))
	# train_outputs = tf.transpose(train_outputs,perm=axis)
	train_predict_scores = train_out[-1]
	if hasattr(train_predict_scores, 'stack'):
		predict_score = train_predict_scores.stack()
	else:
		predict_score = train_predict_scores.pack()

	axis = [1,0] + list(range(2,3))
	predict_score = tf.transpose(predict_score,perm=axis)


	# train_outputs = tf.reshape(train_outputs,(-1,output_dims))
	# train_outputs = tf.nn.dropout(train_outputs, 0.5)
	# predict_score = tf.matmul(train_outputs,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))
	# predict_score = tf.reshape(predict_score,(-1,timesteps,T_w2v_shape[0]))
	# test phase


	test_input_embedded_words = tf.TensorArray(
            dtype=embeded_captions.dtype,
            size=timesteps+1,
            tensor_array_name='test_input_embedded_words')

	predict_words = tf.TensorArray(
            dtype=tf.int64,
            size=timesteps,
            tensor_array_name='predict_words')

	test_hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='test_hidden_state')
	test_input_embedded_words = test_input_embedded_words.write(0,embeded_captions[0])

	def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1, ori_input_feature):
		x_t = test_input_embedded_words.read(time) # batch_size * dim

		ori_feature = tf.reshape(ori_input_feature,(-1,ori_input_dim))
		attend_wx = tf.reshape(matmul_wx(ori_feature, W_a, b_a, attention_dim),(-1,timesteps_v,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(matmul_uh(U_a, h_tm1),dim=1),[1,timesteps_v,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

		attend_fea = ori_input_feature * tf.tile(attend_e,[1,1,ori_input_dim])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		preprocess_x_r = matmul_wx(x_t, W_c_r, b_c_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_c_z, b_c_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_c_h, b_c_h, output_dims)

		r = hard_sigmoid(preprocess_x_r+ matmul_uh(U_c_r,h_tm1) + matmul_uh(A_r,attend_fea))
		z = hard_sigmoid(preprocess_x_z+ matmul_uh(U_c_z,h_tm1) + matmul_uh(A_z,attend_fea))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_c_h,r*h_tm1) + matmul_uh(A_h,attend_fea))

		
		h = (1-z)*hh + z*h_tm1

		test_hidden_state = test_hidden_state.write(time, h)


		drop_h = tf.nn.dropout(h, 0.5)
		predict_score_t = tf.matmul(drop_h,W_c) + tf.reshape(b_c,(1,T_w2v_shape[0]))

		predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
		predict_word_t = tf.argmax(predict_score_t,-1)

		predict_words = predict_words.write(time, predict_word_t) # output


		predict_word_t = tf.gather(T_w2v,predict_word_t)*tf.gather(T_mask,predict_word_t)

		test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

		return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h, ori_input_feature)


	time = tf.constant(0, dtype='int32', name='time')


	test_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=test_step,
            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state, ori_input_feature),
            parallel_iterations=32,
            swap_memory=True)


	predict_words = test_out[-3]
	
	if hasattr(predict_words, 'stack'):
		predict_words = predict_words.stack()
	else:
		predict_words = predict_words.pack()

	axis = [1,0] + list(range(2,3))

	predict_words = tf.transpose(predict_words,perm=[1,0])
	predict_words = tf.reshape(predict_words,(-1,timesteps))

	return predict_score, predict_words