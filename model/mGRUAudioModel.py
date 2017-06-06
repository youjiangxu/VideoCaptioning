import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import tensorflow as tf

import numpy as np
import InitUtil
import math

rng = np.random
rng.seed(1234)

def hard_sigmoid(x):
	x = (0.2 * x) + 0.5
	x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),tf.cast(1., dtype=tf.float32))
	return x

class mGRUCAudioAttentionCaptionModel(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self, input_feature, input_captions, input_audio, voc_size, d_w2v, output_dim, 
		input_audio_dim=34, T_k=[1,3,6], attention_dim = 100, dropout=0.5,
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):
		self.input_audio_dim = input_audio_dim
		

		self.input_audio = tf.transpose(input_audio,perm=[0,2,1])

		self.audio_matrix = self.init_audio_parameters(input_feature.get_shape().as_list()[-1])
		print(self.input_audio.get_shape().as_list())
		self.input_audio = tf.matmul(tf.reshape(self.input_audio,[-1,self.input_audio_dim]),self.audio_matrix)
		self.input_audio = tf.reshape(self.input_audio,[-1,input_feature.get_shape().as_list()[-2],input_feature.get_shape().as_list()[-1]])

		self.input_feature = tf.concat([input_feature,self.input_audio],-1)
		print('input_feature.shape(),', self.input_feature.get_shape().as_list())
		self.input_captions = input_captions

		

		self.voc_size = voc_size
		self.d_w2v = d_w2v
		self.output_dim = output_dim

		self.T_k = T_k
		self.dropout = dropout

		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences
		self.attention_dim = attention_dim

		self.encoder_input_shape = self.input_feature.get_shape().as_list()
		self.decoder_input_shape = self.input_captions.get_shape().as_list()
	def init_parameters(self):
		print('init_parameters ...')

		# encoder parameters
		# print(self.encoder_input_shape)
		encoder_i2h_shape = (self.encoder_input_shape[-1],self.output_dim)
		encoder_h2h_shape = (self.output_dim,self.output_dim)
		self.W_e_r = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_r")
		self.W_e_z = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_z")
		self.W_e_h = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_h")

		self.U_e_r = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_r")
		self.U_e_z = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_z")
		self.U_e_h = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_h")

		self.b_e_r = InitUtil.init_bias_variable((self.output_dim,),name="b_e_r")
		self.b_e_z = InitUtil.init_bias_variable((self.output_dim,),name="b_e_z")
		self.b_e_h = InitUtil.init_bias_variable((self.output_dim,),name="b_e_h")


		# decoder parameters
		self.T_w2v, self.T_mask = self.init_embedding_matrix()
		

		decoder_i2h_shape = (self.d_w2v,self.output_dim)
		decoder_h2h_shape = (self.output_dim,self.output_dim)
		self.W_d_r = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_r")
		self.W_d_z = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_z")
		self.W_d_h = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_h")

		self.U_d_r = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_r")
		self.U_d_z = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_z")
		self.U_d_h = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_h")

		self.b_d_r = InitUtil.init_bias_variable((self.output_dim,),name="b_d_r")
		self.b_d_z = InitUtil.init_bias_variable((self.output_dim,),name="b_d_z")
		self.b_d_h = InitUtil.init_bias_variable((self.output_dim,),name="b_d_h")


		
		self.W_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.attention_dim),init_method='glorot_uniform',name="W_a")
		self.U_a = InitUtil.init_weight_variable((self.output_dim,self.attention_dim),init_method='orthogonal',name="U_a")
		self.b_a = InitUtil.init_bias_variable((self.attention_dim,),name="b_a")

		self.W = InitUtil.init_weight_variable((self.attention_dim,1),init_method='glorot_uniform',name="W")

		self.A_z = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_z")

		self.A_r = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_r")

		self.A_h = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_h")



		# multirate
		self.block_length = int(math.ceil(self.output_dim/len(self.T_k)))
		print('block_length:%d'%self.block_length)


		# classification parameters
		self.W_c = InitUtil.init_weight_variable((self.output_dim,self.voc_size),init_method='uniform',name='W_c')
		self.b_c = InitUtil.init_bias_variable((self.voc_size,),name="b_c")

	def init_embedding_matrix(self):
		'''init word embedding matrix
		'''
		voc_size = self.voc_size
		d_w2v = self.d_w2v	
		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
		for v in range(voc_size):
			LUT[v] = rng.randn(d_w2v)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

		# word 0 is blanked out, word 1 is 'UNK'
		LUT[0] = np.zeros((d_w2v))
		# setup LUT!
		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

		return T_w2v, T_mask 

	def init_audio_parameters(self, embedding_dim):
		'''init word embedding matrix
		'''
		input_audio_dim = self.input_audio_dim

		LUT = np.zeros((input_audio_dim, embedding_dim), dtype='float32')
		for v in range(input_audio_dim):
			LUT[v] = rng.randn(embedding_dim)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

		# setup LUT!
		audio_matrix = tf.Variable(LUT.astype('float32'),trainable=True)

		return audio_matrix 

	def encoder(self):
		'''
			visual feature part
		'''
		print('building encoder ... ...')
		def get_init_state(x, output_dims):
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,2])
			initial_state = tf.expand_dims(initial_state,dim=-1)
			initial_state = tf.tile(initial_state,[1,self.output_dim])
			return initial_state

		
		
		timesteps = self.encoder_input_shape[1]

		embedded_feature = self.input_feature

		initial_state = get_init_state(embedded_feature, self.output_dim)


		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
		embedded_feature = tf.transpose(embedded_feature, perm=axis)

		input_feature = tf.TensorArray(
	            dtype=embedded_feature.dtype,
	            size=timesteps,
	            tensor_array_name='input_feature')
		if hasattr(input_feature, 'unstack'):
			input_feature = input_feature.unstack(embedded_feature)
		else:
			input_feature = input_feature.unpack(embedded_feature)	


		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states')

		self.array_clock = []
		
		for idx, T_i in enumerate(self.T_k):
			self.array_clock.append(tf.ones_like(initial_state[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)],
				dtype=tf.int32)*T_i
				)
		self.array_clock = tf.concat(self.array_clock,axis=-1)	
		def feature_step(time, hidden_states, h_tm1):
			x_t = input_feature.read(time) # batch_size * dim

			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_e_r, self.b_e_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_e_z, self.b_e_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_e_h, self.b_e_h)

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_e_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_e_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_e_h))

			
			# h = (1-z)*hh + z*h_tm1
			h_t = (1-z)*hh + z*h_tm1

			h = tf.where(tf.equal(tf.mod(time,self.array_clock),0),
				h_t,
				h_tm1)
			
			# h = []
			# for idx, T_i in enumerate(self.T_k):
			# 	if time % T_i == 0:
			# 		h.append(h_t[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)])
			# 	else:
			# 		h.append(h_tm1[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)])
			# h = tf.concat(h,axis=-1)

			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states,h)

		

		time = tf.constant(0, dtype='int32', name='time')


		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=feature_step,
	            loop_vars=(time, hidden_states, initial_state),
	            parallel_iterations=32,
	            swap_memory=True)

		last_output = feature_out[-1] 
		hidden_states = feature_out[-2]
		if hasattr(hidden_states, 'stack'):
			encoder_output = hidden_states.stack()
		else:
			encoder_output = hidden_states.pack()

		encoder_output = tf.reshape(encoder_output,[-1,self.encoder_input_shape[1],self.output_dim])
		return last_output, encoder_output

	def decoder(self, initial_state):
		'''
			captions: (batch_size x timesteps) ,int32
			d_w2v: dimension of word 2 vector
		'''
		captions = self.input_captions

		print('building decoder ... ...')
		mask =  tf.not_equal(captions,0)


		loss_mask = tf.cast(mask,tf.float32)

		embedded_captions = tf.gather(self.T_w2v,captions)*tf.gather(self.T_mask,captions)

		timesteps = self.decoder_input_shape[1]


		# batch_size x timesteps x dim -> timesteps x batch_size x dim
		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
		embedded_captions = tf.transpose(embedded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



		input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
	            size=timesteps,
	            tensor_array_name='input_embedded_words')


		if hasattr(input_embedded_words, 'unstack'):
			input_embedded_words = input_embedded_words.unstack(embedded_captions)
		else:
			input_embedded_words = input_embedded_words.unpack(embedded_captions)	


		# preprocess mask
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

		def step(x_t,h_tm1):

			ori_feature = tf.reshape(self.input_feature,(-1,self.encoder_input_shape[-1]))

			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)

			attend_fea = self.input_feature * tf.tile(attend_e,[1,1,self.encoder_input_shape[-1]])
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + tf.matmul(attend_fea,self.A_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + tf.matmul(attend_fea,self.A_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + tf.matmul(attend_fea,self.A_h))

			
			h = (1-z)*hh + z*h_tm1


			return h

		def train_step(time, train_hidden_state, h_tm1):
			x_t = input_embedded_words.read(time) # batch_size * dim
			mask_t = input_mask.read(time)

			h = step(x_t,h_tm1)

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



		train_outputs = tf.reshape(train_outputs,(-1,self.output_dim))
		train_outputs = tf.nn.dropout(train_outputs, self.dropout)
		predict_score = tf.matmul(train_outputs,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
		predict_score = tf.reshape(predict_score,(-1,timesteps,self.voc_size))
		# predict_score = tf.nn.softmax(predict_score,-1)
		# test phase


		test_input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
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
		test_input_embedded_words = test_input_embedded_words.write(0,embedded_captions[0])

		def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
			x_t = test_input_embedded_words.read(time) # batch_size * dim

			h = step(x_t,h_tm1)

			test_hidden_state = test_hidden_state.write(time, h)


			# drop_h = tf.nn.dropout(h, 0.5)
			drop_h = h*self.dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			# predict_score_t = tf.matmul(normed_h,tf.transpose(T_w2v,perm=[1,0]))
			predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
			predict_word_t = tf.argmax(predict_score_t,-1)

			predict_words = predict_words.write(time, predict_word_t) # output


			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t)

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

		return predict_score, predict_words, loss_mask

	def build_model(self):
		print('building model ... ...')
		self.init_parameters()
		last_output, encoder_output = self.encoder()
		predict_score, predict_words , loss_mask= self.decoder(last_output)
		return predict_score, predict_words, loss_mask


class mGRUAudioAttentionBeamsearchCaptionModel(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self, input_feature, input_captions, input_audio, voc_size, d_w2v, output_dim, done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5, 
		input_audio_dim=34, T_k=[1,3,6], attention_dim = 100, dropout=0.5,
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):

		self.input_audio_dim = input_audio_dim
		

		self.input_audio = tf.transpose(input_audio,perm=[0,2,1])

		self.audio_matrix = self.init_audio_parameters(input_feature.get_shape().as_list()[-1])
		print(self.input_audio.get_shape().as_list())
		self.input_audio = tf.matmul(tf.reshape(self.input_audio,[-1,self.input_audio_dim]),self.audio_matrix)
		self.input_audio = tf.reshape(self.input_audio,[-1,input_feature.get_shape().as_list()[-2],input_feature.get_shape().as_list()[-1]])


		self.input_feature = tf.concat([input_feature,self.input_audio],-1)
		print('input_feature.shape(),', self.input_feature.get_shape().as_list())
		self.input_captions = input_captions

		self.voc_size = voc_size
		self.d_w2v = d_w2v
		self.output_dim = output_dim

		self.T_k = T_k
		self.dropout = dropout

		self.beam_size = beam_size

		assert(beamsearch_batchsize==1)
		self.batch_size = beamsearch_batchsize
		self.done_token = done_token
		self.max_len = max_len


		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences
		self.attention_dim = attention_dim

		self.encoder_input_shape = self.input_feature.get_shape().as_list()
		self.decoder_input_shape = self.input_captions.get_shape().as_list()


	def init_parameters(self):
		print('init_parameters ...')

		# encoder parameters
		# print(self.encoder_input_shape)
		encoder_i2h_shape = (self.encoder_input_shape[-1],self.output_dim)
		encoder_h2h_shape = (self.output_dim,self.output_dim)
		self.W_e_r = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_r")
		self.W_e_z = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_z")
		self.W_e_h = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_h")

		self.U_e_r = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_r")
		self.U_e_z = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_z")
		self.U_e_h = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_h")

		self.b_e_r = InitUtil.init_bias_variable((self.output_dim,),name="b_e_r")
		self.b_e_z = InitUtil.init_bias_variable((self.output_dim,),name="b_e_z")
		self.b_e_h = InitUtil.init_bias_variable((self.output_dim,),name="b_e_h")


		# decoder parameters
		self.T_w2v, self.T_mask = self.init_embedding_matrix()

		decoder_i2h_shape = (self.d_w2v,self.output_dim)
		decoder_h2h_shape = (self.output_dim,self.output_dim)
		self.W_d_r = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_r")
		self.W_d_z = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_z")
		self.W_d_h = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_h")

		self.U_d_r = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_r")
		self.U_d_z = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_z")
		self.U_d_h = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_h")

		self.b_d_r = InitUtil.init_bias_variable((self.output_dim,),name="b_d_r")
		self.b_d_z = InitUtil.init_bias_variable((self.output_dim,),name="b_d_z")
		self.b_d_h = InitUtil.init_bias_variable((self.output_dim,),name="b_d_h")


		
		self.W_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.attention_dim),init_method='glorot_uniform',name="W_a")
		self.U_a = InitUtil.init_weight_variable((self.output_dim,self.attention_dim),init_method='orthogonal',name="U_a")
		self.b_a = InitUtil.init_bias_variable((self.attention_dim,),name="b_a")

		self.W = InitUtil.init_weight_variable((self.attention_dim,1),init_method='glorot_uniform',name="W")

		self.A_z = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_z")

		self.A_r = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_r")

		self.A_h = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_h")



		# multirate
		self.block_length = int(math.ceil(self.output_dim/len(self.T_k)))
		print('block_length:%d'%self.block_length)


		# classification parameters
		self.W_c = InitUtil.init_weight_variable((self.output_dim,self.voc_size),init_method='uniform',name='W_c')
		self.b_c = InitUtil.init_bias_variable((self.voc_size,),name="b_c")

	def init_embedding_matrix(self):
		'''init word embedding matrix
		'''
		voc_size = self.voc_size
		d_w2v = self.d_w2v	
		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
		for v in range(voc_size):
			LUT[v] = rng.randn(d_w2v)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

		# word 0 is blanked out, word 1 is 'UNK'
		LUT[0] = np.zeros((d_w2v))
		# setup LUT!
		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

		return T_w2v, T_mask 

	def init_audio_parameters(self, embedding_dim):
		'''init word embedding matrix
		'''
		input_audio_dim = self.input_audio_dim

		LUT = np.zeros((input_audio_dim, embedding_dim), dtype='float32')
		for v in range(input_audio_dim):
			LUT[v] = rng.randn(embedding_dim)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

		# setup LUT!
		audio_matrix = tf.Variable(LUT.astype('float32'),trainable=True)

		return audio_matrix 
	def encoder(self):
		'''
			visual feature part
		'''
		print('building encoder ... ...')
		def get_init_state(x, output_dims):
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,2])
			initial_state = tf.expand_dims(initial_state,dim=-1)
			initial_state = tf.tile(initial_state,[1,self.output_dim])
			return initial_state

		
		
		timesteps = self.encoder_input_shape[1]

		embedded_feature = self.input_feature

		initial_state = get_init_state(embedded_feature, self.output_dim)


		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
		embedded_feature = tf.transpose(embedded_feature, perm=axis)

		input_feature = tf.TensorArray(
	            dtype=embedded_feature.dtype,
	            size=timesteps,
	            tensor_array_name='input_feature')
		if hasattr(input_feature, 'unstack'):
			input_feature = input_feature.unstack(embedded_feature)
		else:
			input_feature = input_feature.unpack(embedded_feature)	


		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states')

		self.array_clock = []
		
		for idx, T_i in enumerate(self.T_k):
			self.array_clock.append(tf.ones_like(initial_state[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)],
				dtype=tf.int32)*T_i
				)
		self.array_clock = tf.concat(self.array_clock,axis=-1)	

		def feature_step(time, hidden_states, h_tm1):
			x_t = input_feature.read(time) # batch_size * dim

			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_e_r, self.b_e_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_e_z, self.b_e_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_e_h, self.b_e_h)

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_e_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_e_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_e_h))

			
			# h = (1-z)*hh + z*h_tm1
			h_t = (1-z)*hh + z*h_tm1

			h = tf.where(tf.equal(tf.mod(time,self.array_clock),0),
				h_t,
				h_tm1)
			# h = []
			# for idx, T_i in enumerate(self.T_k):
			# 	if time % T_i == 0:
			# 		h.append(h_t[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)])
			# 	else:
			# 		h.append(h_tm1[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)])
			# h = tf.concat(h,axis=-1)

			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states,h)

		

		time = tf.constant(0, dtype='int32', name='time')


		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=feature_step,
	            loop_vars=(time, hidden_states, initial_state),
	            parallel_iterations=32,
	            swap_memory=True)

		last_output = feature_out[-1] 
		hidden_states = feature_out[-2]
		if hasattr(hidden_states, 'stack'):
			encoder_output = hidden_states.stack()
		else:
			encoder_output = hidden_states.pack()

		encoder_output = tf.reshape(encoder_output,[-1,self.encoder_input_shape[1],self.output_dim])
		return last_output, encoder_output

	def decoder(self, initial_state):
		'''
			captions: (batch_size x timesteps) ,int32
			d_w2v: dimension of word 2 vector
		'''
		captions = self.input_captions

		print('building decoder ... ...')
		mask =  tf.not_equal(captions,0)


		loss_mask = tf.cast(mask,tf.float32)

		embedded_captions = tf.gather(self.T_w2v,captions)*tf.gather(self.T_mask,captions)

		timesteps = self.decoder_input_shape[1]


		# batch_size x timesteps x dim -> timesteps x batch_size x dim
		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
		embedded_captions = tf.transpose(embedded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



		input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
	            size=timesteps,
	            tensor_array_name='input_embedded_words')


		if hasattr(input_embedded_words, 'unstack'):
			input_embedded_words = input_embedded_words.unstack(embedded_captions)
		else:
			input_embedded_words = input_embedded_words.unpack(embedded_captions)	


		# preprocess mask
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

		def step(x_t,h_tm1):

			ori_feature = tf.reshape(self.input_feature,(-1,self.encoder_input_shape[-1]))

			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)

			attend_fea = self.input_feature * tf.tile(attend_e,[1,1,self.encoder_input_shape[-1]])
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + tf.matmul(attend_fea,self.A_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + tf.matmul(attend_fea,self.A_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + tf.matmul(attend_fea,self.A_h))

			
			h = (1-z)*hh + z*h_tm1


			return h

		def train_step(time, train_hidden_state, h_tm1):
			x_t = input_embedded_words.read(time) # batch_size * dim
			mask_t = input_mask.read(time)

			h = step(x_t,h_tm1)

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



		train_outputs = tf.reshape(train_outputs,(-1,self.output_dim))
		train_outputs = tf.nn.dropout(train_outputs, self.dropout)
		predict_score = tf.matmul(train_outputs,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
		predict_score = tf.reshape(predict_score,(-1,timesteps,self.voc_size))
		# predict_score = tf.nn.softmax(predict_score,-1)
		# test phase


		test_input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
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
		test_input_embedded_words = test_input_embedded_words.write(0,embedded_captions[0])

		def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
			x_t = test_input_embedded_words.read(time) # batch_size * dim

			h = step(x_t,h_tm1)

			test_hidden_state = test_hidden_state.write(time, h)


			# drop_h = tf.nn.dropout(h, 0.5)
			drop_h = h*self.dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			# predict_score_t = tf.matmul(normed_h,tf.transpose(T_w2v,perm=[1,0]))
			predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
			predict_word_t = tf.argmax(predict_score_t,-1)

			predict_words = predict_words.write(time, predict_word_t) # output


			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t)

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

		return predict_score, predict_words, loss_mask
	def beamSearchDecoder(self, initial_state):
		'''
			captions: (batch_size x timesteps) ,int32
			d_w2v: dimension of word 2 vector
		'''

		# self.batch_size = self.input_captions.get_shape().as_list().eval()[0]
		def step(x_t,h_tm1):
			ori_feature = tf.tile(tf.expand_dims(self.input_feature,dim=1),[1,self.beam_size,1,1])
			ori_feature = tf.reshape(ori_feature,(-1,self.encoder_input_shape[-1]))

			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)
			print('attend_e.get_shape()',attend_e.get_shape().as_list())

			attend_fea = tf.multiply(tf.reshape(ori_feature,[self.batch_size,self.beam_size,self.encoder_input_shape[-2],self.encoder_input_shape[-1]]),
				tf.reshape(attend_e,[self.batch_size,self.beam_size,self.encoder_input_shape[-2],1]))
			attend_fea = tf.reshape(tf.reduce_sum(attend_fea,reduction_indices=2),[self.batch_size*self.beam_size,self.encoder_input_shape[-1]])

			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

			r = tf.nn.sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + tf.matmul(attend_fea,self.A_r))
			z = tf.nn.sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + tf.matmul(attend_fea,self.A_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + tf.matmul(attend_fea,self.A_h))

			
			h = (1-z)*hh + z*h_tm1
			return h
		def take_step_zero(x_0, h_0):

			x_0 = tf.gather(self.T_w2v,x_0)*tf.gather(self.T_mask,x_0)
			x_0 = tf.reshape(x_0,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_0,h_0)
			drop_h = h
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
			# logprobs = tf.log(tf.nn.softmax(predict_score_t))
			logprobs = tf.nn.log_softmax(predict_score_t)

			print('logrobs.get_shape().as_list():',logprobs.get_shape().as_list())

			logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.voc_size])

			
			past_logprobs, indices = tf.nn.top_k(
			        logprobs_batched[:,0,:],self.beam_size)

			symbols = indices % self.voc_size
			parent_refs = indices//self.voc_size
			h = tf.gather(h,  tf.reshape(parent_refs,[-1]))
			print('symbols.shape',symbols.get_shape().as_list())

			past_symbols = tf.concat([tf.expand_dims(symbols, 2), tf.zeros((self.batch_size, self.beam_size, self.max_len-1), dtype=tf.int32)],-1)
			return symbols, h, past_symbols, past_logprobs


		def test_step(time, x_t, h_tm1, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):

			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_t,h_tm1)

			print('h.shape()',h.get_shape().as_list())
			drop_h = h
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			logprobs = tf.nn.log_softmax(predict_score_t)
			# logprobs = tf.log(tf.nn.softmax(predict_score_t))
			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size])


			logprobs, indices = tf.nn.top_k(logprobs, self.beam_size, sorted=False)  # logprobs-->  [batch_size==1 , beam_size , beam_size]
			
			
			logprobs = logprobs+tf.expand_dims(past_logprobs, 2)

			
			past_logprobs, topk_indices = tf.nn.top_k(
			    tf.reshape(logprobs, [1, self.beam_size * self.beam_size]),
			    self.beam_size, 
			    sorted=False
			)       
			indices = tf.gather(tf.reshape(indices,[-1,1]),tf.reshape(topk_indices,[-1]))

			# For continuing to the next symbols
			symbols = indices % self.voc_size
			symbols = tf.reshape(symbols, [1,self.beam_size])


			parent_refs = topk_indices // self.beam_size
			h = tf.gather(h,  tf.reshape(parent_refs,[-1]))
			
			past_symbols_batch_major = tf.reshape(past_symbols[:,:,0:time], [-1, time])

			beam_past_symbols = tf.gather(past_symbols_batch_major,  parent_refs)
			

			past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2), tf.zeros((1, self.beam_size, self.max_len-time-1), dtype=tf.int32)],2)
			past_symbols = tf.reshape(past_symbols, [1,self.beam_size,self.max_len])
			
			# For finishing the beam here
			cond1 = tf.equal(symbols,tf.ones_like(symbols,tf.int32)*self.done_token) # condition on done sentence
			

			for_finished_logprobs = tf.where(cond1,past_logprobs,tf.ones_like(past_logprobs,tf.float32)* -1e5)

			done_indice_max = tf.cast(tf.argmax(for_finished_logprobs,axis=-1),tf.int32)
			logprobs_done_max = tf.reduce_max(for_finished_logprobs,reduction_indices=-1)

			
			done_past_symbols = tf.gather(tf.reshape(past_symbols,[self.beam_size,self.max_len]),done_indice_max)
			
			cond2 = tf.greater(logprobs_done_max,logprobs_finished_beams)
			cond3 = tf.equal(done_past_symbols[:,time],self.done_token)
			cond4 = tf.equal(time,self.max_len-1)

			finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
			                                done_past_symbols,
			                                finished_beams)
			logprobs_finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
											logprobs_done_max, 
											logprobs_finished_beams)

			

			return (time+1, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)



		captions = self.input_captions

		# past_logprobs = tf.ones((self.batch_size,), dtype=tf.float32) * -1e5
		# past_symbols = tf.zeros((self.batch_size, self.beam_size, self.max_len), dtype=tf.int32)

		finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32)
		logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')

		x_0 = captions[:,0]
		x_0 = tf.expand_dims(x_0,dim=-1)
		print('x_0',x_0.get_shape().as_list())
		x_0 = tf.tile(x_0,[1,self.beam_size])


		h_0 = tf.expand_dims(initial_state,dim=1)
		h_0 = tf.reshape(tf.tile(h_0,[1,self.beam_size,1]),[self.batch_size*self.beam_size,self.output_dim])
		symbols, h, past_symbols, past_logprobs = take_step_zero(x_0,h_0)
		time = tf.constant(1, dtype='int32', name='time')
		timesteps = self.max_len

		

		test_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=test_step,
	            loop_vars=(time, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams),
	            parallel_iterations=32,
	            swap_memory=True)

		

		


		out_finished_beams = test_out[-2]
		out_logprobs_finished_beams = test_out[-1]
		out_past_symbols = test_out[-4]

		return   out_finished_beams, out_logprobs_finished_beams, out_past_symbols


	def build_model(self):
		print('building model ... ...')
		self.init_parameters()
		last_output, encoder_output = self.encoder()
		predict_score, predict_words , loss_mask= self.decoder(last_output)
		finished_beam, logprobs_finished_beams, past_symbols = self.beamSearchDecoder(last_output)
		return predict_score, predict_words, loss_mask, finished_beam, logprobs_finished_beams, past_symbols