
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


class mGRUCaptionModel(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, T_k=[1,3,6], dropout=0.5,
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):
		self.input_feature = input_feature
		self.input_captions = input_captions

		self.voc_size = voc_size
		self.d_w2v = d_w2v
		self.output_dim = output_dim
		
		self.T_k = T_k

		self.dropout = dropout


		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences

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

		# self.array_clock = []
		
		# for idx, T_i in enumerate(self.T_k):
		# 	self.array_clock.append(tf.ones_like(initial_state[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)],
		# 		dtype=tf.int32)*T_i
		# 		)
		# self.array_clock = tf.concat(self.array_clock,axis=-1)	
		def feature_step(time, hidden_states, h_tm1):
			x_t = input_feature.read(time) # batch_size * dim

			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_e_r, self.b_e_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_e_z, self.b_e_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_e_h, self.b_e_h)

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_e_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_e_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_e_h))

			
			ht = (1-z)*hh + z*h_tm1

			# h = tf.where(tf.equal(tf.mod(time,self.array_clock),0),
			# 	ht,
			# 	h_tm1)
			h = []
			for idx, T_i in enumerate(self.T_k):
				if time % T_i == 0:
					h.append(ht[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)])
				else:
					h.append(h_tm1[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)])
			h = tf.concat(h,axis=-1)
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
		return last_output

	def decoder(self, initial_state):
		'''
			captions: (batch_size x timesteps) ,int32
			d_w2v: dimension of word 2 vector
		'''
		captions = self.input_captions

		print('building decoder ... ...')
		mask =  tf.not_equal(captions,0)
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
			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h))

			
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

		return predict_score, predict_words

	def build_model(self):
		print('building model ... ...')
		self.init_parameters()
		last_output = self.encoder()
		predict_score, predict_words = self.decoder(last_output)
		return predict_score, predict_words


class mGRUAttentionCaptionModel(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, T_k=[1,3,6], attention_dim = 100, dropout=0.5,
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):
		self.input_feature = input_feature
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
		self.block_length = self.output_dim/len(self.T_k)+1
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

class mLSTMAttentionCaptionModel(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, T_k=[1,3,6], attention_dim = 100, dropout=0.5,
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):
		self.input_feature = input_feature
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
		encoder_i2h_shape = (self.encoder_input_shape[-1],4*self.output_dim)
		encoder_h2h_shape = (self.output_dim,4*self.output_dim)


		self.W_e = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e")
		self.U_e = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e")
		self.b_e = InitUtil.init_bias_variable((4*self.output_dim,),name="b_e")

		# self.W_e_i = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_i")
		# self.W_e_f = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_f")
		# self.W_e_o = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_o")
		# self.W_e_c = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_c")

		# self.U_e_i = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_i")
		# self.U_e_f = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_f")
		# self.U_e_o = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_o")
		# self.U_e_c = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_c")

		# self.b_e_i = InitUtil.init_bias_variable((self.output_dim,),name="b_e_i")
		# self.b_e_f = InitUtil.init_bias_variable((self.output_dim,),name="b_e_f")
		# self.b_e_o = InitUtil.init_bias_variable((self.output_dim,),name="b_e_o")
		# self.b_e_c = InitUtil.init_bias_variable((self.output_dim,),name="b_e_c")


		# decoder parameters
		self.T_w2v, self.T_mask = self.init_embedding_matrix()

		decoder_i2h_shape = (self.d_w2v,4*self.output_dim)
		decoder_h2h_shape = (self.output_dim,4*self.output_dim)
		self.W_d = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d")
		self.U_d = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d")
		self.b_d = InitUtil.init_bias_variable((4*self.output_dim,),name="b_d")

		# self.W_d_i = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_i")
		# self.W_d_f = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_f")
		# self.W_d_o = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_o")
		# self.W_d_c = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_c")

		# self.U_d_i = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_i")
		# self.U_d_f = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_f")
		# self.U_d_o = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_o")
		# self.U_d_c = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_c")

		# self.b_d_i = InitUtil.init_bias_variable((self.output_dim,),name="b_d_i")
		# self.b_d_f = InitUtil.init_bias_variable((self.output_dim,),name="b_d_f")
		# self.b_d_o = InitUtil.init_bias_variable((self.output_dim,),name="b_d_o")
		# self.b_d_c = InitUtil.init_bias_variable((self.output_dim,),name="b_d_c")


		
		self.W_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.attention_dim),init_method='glorot_uniform',name="W_a")
		self.U_a = InitUtil.init_weight_variable((self.output_dim,self.attention_dim),init_method='orthogonal',name="U_a")
		self.b_a = InitUtil.init_bias_variable((self.attention_dim,),name="b_a")

		self.W = InitUtil.init_weight_variable((self.attention_dim,1),init_method='glorot_uniform',name="W")

		# self.A_i = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_i")
		# self.A_f = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_f")
		# self.A_o = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_o")
		# self.A_c = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='orthogonal',name="A_c")

		self.A = InitUtil.init_weight_variable((self.encoder_input_shape[-1],4*self.output_dim),init_method='orthogonal',name="A_i")

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

		initial_state_c = get_init_state(embedded_feature, self.output_dim)
		initial_state_h = get_init_state(embedded_feature, self.output_dim)


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


		hidden_states_h = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states_h')

		hidden_states_c = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states_c')

		self.array_clock = []
		
		for idx, T_i in enumerate(self.T_k):
			self.array_clock.append(tf.ones_like(initial_state_h[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)],
				dtype=tf.int32)*T_i
				)
		self.array_clock = tf.concat(self.array_clock,axis=-1)	

		def feature_step(time, hidden_states_h, hidden_states_c, h_tm1, c_tm1):
			x_t = input_feature.read(time) # batch_size * dim

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_e, self.b_e)
			preprocess_uh = tf.matmul(h_tm1, self.U_e)

			preprocess_x_i = preprocess_x[:,0:self.output_dim]
			preprocess_x_f = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_o = preprocess_x[:,2*self.output_dim:3*self.output_dim]
			preprocess_x_c = preprocess_x[:,3*self.output_dim::]

			preprocess_uh_i = preprocess_uh[:,0:self.output_dim]
			preprocess_uh_f = preprocess_uh[:,self.output_dim:2*self.output_dim]
			preprocess_uh_o = preprocess_uh[:,2*self.output_dim:3*self.output_dim]
			preprocess_uh_c = preprocess_uh[:,3*self.output_dim::]
			# preprocess_x_i = tf.nn.xw_plus_b(x_t, self.W_e_i, self.b_e_i)
			# preprocess_x_f = tf.nn.xw_plus_b(x_t, self.W_e_f, self.b_e_f)
			# preprocess_x_o = tf.nn.xw_plus_b(x_t, self.W_e_o, self.b_e_o)

			# preprocess_x_c = tf.nn.xw_plus_b(x_t, self.W_e_c, self.b_e_c)

			i = hard_sigmoid(preprocess_x_i + preprocess_uh_i)
			f = hard_sigmoid(preprocess_x_f + preprocess_uh_f)
			o = hard_sigmoid(preprocess_x_o + preprocess_uh_o)



			c_tt = tf.nn.tanh(preprocess_x_c + preprocess_uh_c) 
			c_t = i*c_tt + f*c_tm1
			h_t = o*tf.nn.tanh(c_t)


			h = tf.where(tf.equal(tf.mod(time,self.array_clock),0),
				h_t,
				h_tm1)

			c = tf.where(tf.equal(tf.mod(time,self.array_clock),0),
				c_t,
				c_tm1)
			


			hidden_states_h = hidden_states_h.write(time, h)
			hidden_states_c = hidden_states_c.write(time, c)

			return (time+1,hidden_states_h, hidden_states_c, h,c)

		

		time = tf.constant(0, dtype='int32', name='time')


		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=feature_step,
	            loop_vars=(time, hidden_states_h, hidden_states_c, initial_state_h, initial_state_c),
	            parallel_iterations=32,
	            swap_memory=True)

		last_output_h = feature_out[-2]
		last_output_c = feature_out[-1]

		hidden_states_h = feature_out[1]
		hidden_states_c = feature_out[2]

		if hasattr(hidden_states_h, 'stack'):
			hidden_states_h = hidden_states_h.stack()
		else:
			hidden_states_h = hidden_states_h.pack()

		if hasattr(hidden_states_c, 'stack'):
			hidden_states_c = hidden_states_c.stack()
		else:
			hidden_states_c = hidden_states_c.pack()

		hidden_states_h = tf.reshape(hidden_states_h,[-1,self.encoder_input_shape[1],self.output_dim])
		hidden_states_c = tf.reshape(hidden_states_c,[-1,self.encoder_input_shape[1],self.output_dim])

		return last_output_h, last_output_c, hidden_states_h, hidden_states_c

	def decoder(self, initial_state_h, initial_state_c):
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


		train_hidden_state_h = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='train_hidden_state_h')

		train_hidden_state_c = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='train_hidden_state_c')

		def step(x_t,h_tm1,c_tm1):

			ori_feature = tf.reshape(self.input_feature,(-1,self.encoder_input_shape[-1]))

			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)

			attend_fea = self.input_feature * tf.tile(attend_e,[1,1,self.encoder_input_shape[-1]])
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

			attend_fea = tf.matmul(attend_fea,self.A)

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_d, self.b_d)
			preprocess_uh = tf.matmul(h_tm1, self.U_d)

			preprocess_x_i = preprocess_x[:,0:self.output_dim]
			preprocess_x_f = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_o = preprocess_x[:,2*self.output_dim:3*self.output_dim]
			preprocess_x_c = preprocess_x[:,3*self.output_dim::]

			preprocess_uh_i = preprocess_uh[:,0:self.output_dim]
			preprocess_uh_f = preprocess_uh[:,self.output_dim:2*self.output_dim]
			preprocess_uh_o = preprocess_uh[:,2*self.output_dim:3*self.output_dim]
			preprocess_uh_c = preprocess_uh[:,3*self.output_dim::]
			
			attend_fea_i = attend_fea[:,0:self.output_dim]
			attend_fea_f = attend_fea[:,self.output_dim:2*self.output_dim]
			attend_fea_o = attend_fea[:,2*self.output_dim:3*self.output_dim]
			attend_fea_c = attend_fea[:,3*self.output_dim::]

			i = hard_sigmoid(preprocess_x_i + preprocess_uh_i + attend_fea_i)
			f = hard_sigmoid(preprocess_x_f + preprocess_uh_f + attend_fea_f)
			o = hard_sigmoid(preprocess_x_o + preprocess_uh_o + attend_fea_o)



			c_tt = tf.nn.tanh(preprocess_x_c + preprocess_uh_c + attend_fea_c) 
			c_t = i*c_tt + f*c_tm1
			h_t = o*tf.nn.tanh(c_t)


			return h_t, c_t

		def train_step(time, train_hidden_state_h, train_hidden_state_c, h_tm1, c_tm1):
			x_t = input_embedded_words.read(time) # batch_size * dim
			mask_t = input_mask.read(time)

			h,c = step(x_t,h_tm1,c_tm1)

			tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

			h = tf.where(tiled_mask_t, h, h_tm1) # (batch_size, output_dims)
			c = tf.where(tiled_mask_t, c, c_tm1)

			train_hidden_state_h = train_hidden_state_h.write(time, h)
			train_hidden_state_c = train_hidden_state_c.write(time, c)

			return (time+1,train_hidden_state_h, train_hidden_state_c ,h,c)

		

		time = tf.constant(0, dtype='int32', name='time')


		train_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=train_step,
	            loop_vars=(time, train_hidden_state_h, train_hidden_state_c, initial_state_h, initial_state_c),
	            parallel_iterations=32,
	            swap_memory=True)


		train_hidden_state_h = train_out[1]
		train_last_output = train_out[-2] 
		
		if hasattr(train_hidden_state_h, 'stack'):
			train_outputs = train_hidden_state_h.stack()
		else:
			train_outputs = train_hidden_state_h.pack()

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

		test_hidden_state_h = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='test_hidden_state_h')

		test_hidden_state_c = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='test_hidden_state_c')


		test_input_embedded_words = test_input_embedded_words.write(0,embedded_captions[0])

		def test_step(time, test_hidden_state_h, test_hidden_state_c, test_input_embedded_words, predict_words, h_tm1, c_tm1):
			x_t = test_input_embedded_words.read(time) # batch_size * dim

			h,c = step(x_t,h_tm1, c_tm1)

			test_hidden_state_h = test_hidden_state_h.write(time, h)
			test_hidden_state_c = test_hidden_state_c.write(time, c)

			# drop_h = tf.nn.dropout(h, 0.5)
			drop_h = h*self.dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			# predict_score_t = tf.matmul(normed_h,tf.transpose(T_w2v,perm=[1,0]))
			predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
			predict_word_t = tf.argmax(predict_score_t,-1)

			predict_words = predict_words.write(time, predict_word_t) # output


			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t)

			test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

			return (time+1,test_hidden_state_h, test_hidden_state_c, test_input_embedded_words, predict_words, h, c)


		time = tf.constant(0, dtype='int32', name='time')


		test_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=test_step,
	            loop_vars=(time, test_hidden_state_h, test_hidden_state_c, test_input_embedded_words, predict_words, initial_state_h, initial_state_c),
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

		return predict_score, predict_words, loss_mask

	def build_model(self):
		print('building model ... ...')
		self.init_parameters()
		last_h, last_c, hidden_h, hidden_c = self.encoder()
		predict_score, predict_words, loss_mask= self.decoder(last_h, last_c)
		return predict_score, predict_words, loss_mask



class mGRUAttentionBeamsearchCaptionModel(object):
	'''
		caption model for ablation studying
	'''
	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5, 
		T_k=[1,3,6], attention_dim = 100, dropout=0.5,
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):
		self.input_feature = input_feature
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


		# self.past_logprobs = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')
		# self.past_symbols = tf.zeros((self.batch_size, self.beam_size, self.max_len), dtype=tf.int32)

		# self.finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32)
		# self.logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')

	def init_parameters(self):
		print('init_parameters ...')

		# encoder parameters
		# print(self.encoder_input_shape)
		encoder_i2h_shape = (self.encoder_input_shape[-1],3*self.output_dim)
		encoder_h2h_shape = (self.output_dim,self.output_dim)
		self.W_e = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e")
		

		self.U_e_r = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_r")
		self.U_e_z = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_z")
		self.U_e_h = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_h")

		self.b_e = InitUtil.init_bias_variable((3*self.output_dim,),name="b_e")
		


		# decoder parameters
		self.T_w2v, self.T_mask = self.init_embedding_matrix()

		decoder_i2h_shape = (self.d_w2v,3*self.output_dim)
		decoder_h2h_shape = (self.output_dim,self.output_dim)
		self.W_d = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d")
		

		self.U_d_r = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_r")
		self.U_d_z = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_z")
		self.U_d_h = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_h")

		self.b_d = InitUtil.init_bias_variable((3*self.output_dim,),name="b_d_r")
		


		
		self.W_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.attention_dim),init_method='glorot_uniform',name="W_a")
		self.U_a = InitUtil.init_weight_variable((self.output_dim,self.attention_dim),init_method='orthogonal',name="U_a")
		self.b_a = InitUtil.init_bias_variable((self.attention_dim,),name="b_a")

		self.W = InitUtil.init_weight_variable((self.attention_dim,1),init_method='glorot_uniform',name="W")

		self.A = InitUtil.init_weight_variable((self.encoder_input_shape[-1],3*self.output_dim),init_method='orthogonal',name="A")

		


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

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_e, self.b_e)

			preprocess_x_r = preprocess_x[:,0:self.output_dim]
			preprocess_x_z = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_h = preprocess_x[:,2*self.output_dim::]

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_e_r))
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_e_z))
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_e_h))


			h_t = (1-z)*hh + z*h_tm1

			h = tf.where(tf.equal(tf.mod(time,self.array_clock),0),
				h_t,
				h_tm1)


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

			attend_fea = tf.matmul(attend_fea,self.A)

			attend_fea_r = attend_fea[:,0:self.output_dim]
			attend_fea_z = attend_fea[:,self.output_dim:2*self.output_dim]
			attend_fea_h = attend_fea[:,2*self.output_dim::]

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_d, self.b_d)

			preprocess_x_r = preprocess_x[:,0:self.output_dim]
			preprocess_x_z = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_h = preprocess_x[:,2*self.output_dim::]

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + attend_fea_r)
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + attend_fea_z)
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + attend_fea_h)

			
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

			attend_fea = tf.matmul(attend_fea,self.A)

			attend_fea_r = attend_fea[:,0:self.output_dim]
			attend_fea_z = attend_fea[:,self.output_dim:2*self.output_dim]
			attend_fea_h = attend_fea[:,2*self.output_dim::]

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_d, self.b_d)

			preprocess_x_r = preprocess_x[:,0:self.output_dim]
			preprocess_x_z = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_h = preprocess_x[:,2*self.output_dim::]

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + attend_fea_r)
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + attend_fea_z)
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + attend_fea_h)

			

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
			# beam_hidden_state, past_symbols_states, finished_beams_states,
			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_t,h_tm1)

			print('h.shape()',h.get_shape().as_list())
			drop_h = h
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			logprobs = tf.nn.log_softmax(predict_score_t)
			# logprobs = tf.log(tf.nn.softmax(predict_score_t))
			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size])


			# logprobs, indices = tf.nn.top_k(logprobs, self.beam_size, sorted=False)  # logprobs-->  [batch_size==1 , beam_size , beam_size]
			# logprobs = logprobs+tf.expand_dims(past_logprobs, 2)
			# past_logprobs, topk_indices = tf.nn.top_k(
			#     tf.reshape(logprobs, [1, self.beam_size * self.beam_size]),
			#     self.beam_size, 
			#     sorted=False
			# )       
			# indices = tf.gather(tf.reshape(indices,[-1,1]),tf.reshape(topk_indices,[-1]))
			# # For continuing to the next symbols
			# symbols = indices % self.voc_size
			# symbols = tf.reshape(symbols, [1,self.beam_size])
			# parent_refs = topk_indices // self.beam_size

			logprobs = logprobs+tf.expand_dims(past_logprobs, 2)
			past_logprobs, topk_indices = tf.nn.top_k(
			    tf.reshape(logprobs, [1, self.beam_size * self.voc_size]),
			    self.beam_size, 
			    sorted=False
			)       

			symbols = topk_indices % self.voc_size
			symbols = tf.reshape(symbols, [1,self.beam_size])
			parent_refs = topk_indices // self.voc_size


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
			
			logprobs_done_max = tf.div(-logprobs_done_max,tf.cast(time,tf.float32))
			cond2 = tf.greater(logprobs_finished_beams,logprobs_done_max)

			# cond2 = tf.greater(logprobs_done_max,logprobs_finished_beams)
			
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
		logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * float('inf')

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


# class mGRUBidirectionalAttentionCaptionModel(mGRUAttentionCaptionModel):
# 	'''
# 		caption model for ablation studying
# 	'''
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, T_k=[1,3,6], attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):
# 		self.input_feature = tf.concat([input_feature,input_feature[:,::-1,:]],-2)
# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v
# 		self.output_dim = output_dim

# 		self.T_k = T_k
# 		self.dropout = dropout

# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim

# 		self.encoder_input_shape = self.input_feature.get_shape().as_list()
# 		print('encoder_input_shape:',self.encoder_input_shape)
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()



# class mGRUAttentionBeamsearchCaptionMergedFeaureModel(mGRUAttentionBeamsearchCaptionModel):
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5, 
# 		T_k=[1,3,6], attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):
# 		# self.input_feature = tf.concat([tf.nn.l2_normalize(input_feature[:,:,0:2048],-1),tf.nn.l2_normalize(input_feature[:,:,2048::],-1)],-1)
# 		self.input_feature = input_feature

# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v
# 		self.output_dim = output_dim

# 		self.T_k = T_k
# 		self.dropout = dropout

# 		self.beam_size = beam_size

# 		assert(beamsearch_batchsize==1)
# 		self.batch_size = beamsearch_batchsize
# 		self.done_token = done_token
# 		self.max_len = max_len


# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim

# 		self.encoder_input_shape = self.input_feature.get_shape().as_list()
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()


	
# class mGRUPaperModel(object):
# 	'''
# 		caption model for ablation studying
# 	'''
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v=512, output_dim=512, cell_dimension=512, T_k=[1,2,4,8], attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):
# 		self.input_feature = input_feature
# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v
# 		self.output_dim = output_dim

# 		self.cell_dimension = cell_dimension

# 		self.T_k = T_k
# 		self.dropout = dropout

# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim


# 		self.encoder_input_shape = self.input_feature.get_shape().as_list()
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()
# 	def init_parameters(self):
# 		print('init_parameters ...')

# 		# encoder parameters
# 		# print(self.encoder_input_shape)
# 		encoder_i2h_shape = (self.encoder_input_shape[-1],self.cell_dimension)
# 		encoder_h2h_shape = (self.cell_dimension,self.cell_dimension)
# 		self.W_e_r = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_r")
# 		self.W_e_z = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_z")
# 		self.W_e_h = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e_h")

# 		self.U_e_r = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_r")
# 		self.U_e_z = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_z")
# 		self.U_e_h = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='orthogonal',name="U_e_h")

# 		self.b_e_r = InitUtil.init_bias_variable((self.cell_dimension,),name="b_e_r")
# 		self.b_e_z = InitUtil.init_bias_variable((self.cell_dimension,),name="b_e_z")
# 		self.b_e_h = InitUtil.init_bias_variable((self.cell_dimension,),name="b_e_h")

# 		self.W_e_o = InitUtil.init_weight_variable((self.cell_dimension,self.output_dim),init_method='glorot_uniform',name='W_e_o')
# 		self.b_e_o = InitUtil.init_bias_variable((self.output_dim,),name='b_e_o')
		

# 		# decoder parameters
# 		self.T_w2v, self.T_mask = self.init_embedding_matrix()

# 		decoder_i2h_shape = (self.d_w2v,self.cell_dimension)
# 		decoder_h2h_shape = (self.cell_dimension,self.cell_dimension)
# 		self.W_d_r = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_r")
# 		self.W_d_z = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_z")
# 		self.W_d_h = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_h")

# 		self.U_d_r = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_r")
# 		self.U_d_z = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_z")
# 		self.U_d_h = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='orthogonal',name="U_d_h")

# 		self.b_d_r = InitUtil.init_bias_variable((self.cell_dimension,),name="b_d_r")
# 		self.b_d_z = InitUtil.init_bias_variable((self.cell_dimension,),name="b_d_z")
# 		self.b_d_h = InitUtil.init_bias_variable((self.cell_dimension,),name="b_d_h")

# 		self.W_d_o = InitUtil.init_weight_variable((self.cell_dimension,self.output_dim),init_method='glorot_uniform',name='W_d_o')
# 		self.b_d_o = InitUtil.init_bias_variable((self.output_dim,),name='b_d_o')
# 		# Linear parameters

# 		self.W_l_i = InitUtil.init_weight_variable((self.d_w2v,self.d_w2v),init_method='glorot_uniform',name='W_l_i')
# 		self.W_l_i_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.d_w2v),init_method='glorot_uniform',name='W_l_i_a')
# 		self.b_l_i = InitUtil.init_bias_variable((self.d_w2v,),name='b_l_i')

# 		self.W_l_o = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='glorot_uniform',name='W_l_o')
# 		self.W_l_o_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.output_dim),init_method='glorot_uniform',name='W_l_o_a')
# 		self.b_l_o = InitUtil.init_bias_variable((self.output_dim,),name='b_l_o')

		
# 		self.W_a = InitUtil.init_weight_variable((self.encoder_input_shape[-1],self.attention_dim),init_method='glorot_uniform',name="W_a")
# 		self.U_a = InitUtil.init_weight_variable((self.cell_dimension,self.attention_dim),init_method='glorot_uniform',name="U_a")
# 		self.b_a = InitUtil.init_bias_variable((self.attention_dim,),name="b_a")

# 		self.W = InitUtil.init_weight_variable((self.attention_dim,1),init_method='glorot_uniform',name="W")

# 		# self.A_z = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='orthogonal',name="A_z")

# 		# self.A_r = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='orthogonal',name="A_r")

# 		# self.A_h = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='orthogonal',name="A_h")



# 		# multirate
# 		self.block_length = self.cell_dimension/len(self.T_k)+1
# 		print('block_length:%d'%self.block_length)


# 		# #classification parameters
# 		self.W_c = InitUtil.init_weight_variable((self.output_dim,self.voc_size),init_method='uniform',name='W_c')
# 		self.b_c = InitUtil.init_bias_variable((self.voc_size,),name="b_c")

# 	def init_embedding_matrix(self):
# 		'''init word embedding matrix
# 		'''
# 		voc_size = self.voc_size
# 		d_w2v = self.d_w2v	
# 		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
# 		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

# 		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
# 		for v in range(voc_size):
# 			LUT[v] = rng.randn(d_w2v)
# 			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

# 		# word 0 is blanked out, word 1 is 'UNK'
# 		LUT[0] = np.zeros((d_w2v))
# 		# setup LUT!
# 		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

# 		return T_w2v, T_mask 

# 	def encoder(self):
# 		'''
# 			visual feature part
# 		'''
# 		print('building encoder ... ...')
# 		def get_init_state(x, output_dim):
# 			initial_state = tf.zeros_like(x)
# 			initial_state = tf.reduce_sum(initial_state,axis=[1,2])
# 			initial_state = tf.expand_dims(initial_state,dim=-1)
# 			initial_state = tf.tile(initial_state,[1,output_dim])
# 			return initial_state

		
		
# 		timesteps = self.encoder_input_shape[1]

# 		embedded_feature = self.input_feature

# 		initial_state = get_init_state(embedded_feature, self.cell_dimension)


# 		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
# 		embedded_feature = tf.transpose(embedded_feature, perm=axis)

# 		input_feature = tf.TensorArray(
# 	            dtype=embedded_feature.dtype,
# 	            size=timesteps,
# 	            tensor_array_name='input_feature')
# 		if hasattr(input_feature, 'unstack'):
# 			input_feature = input_feature.unstack(embedded_feature)
# 		else:
# 			input_feature = input_feature.unpack(embedded_feature)	


# 		hidden_states = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='hidden_states')


# 		def feature_step(time, hidden_states, h_tm1):
# 			x_t = input_feature.read(time) # batch_size * dim

# 			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_e_r, self.b_e_r)
# 			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_e_z, self.b_e_z)
# 			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_e_h, self.b_e_h)

# 			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_e_r))
# 			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_e_z))
# 			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_e_h))

			
# 			# h = (1-z)*hh + z*h_tm1
# 			h_t = (1-z)*hh + z*h_tm1
# 			h = []
# 			# def fn1(h_t, idx):
# 			# 	return h_t[:,idx*self.block_length:min((idx+1)*self.block_length,self.cell_dimension)]
# 			# def fn2(h_tm1, idx):
# 			# 	return h_tm1[:,idx*self.block_length:min((idx+1)*self.block_length,self.cell_dimension)] 
# 			for idx, T_i in enumerate(self.T_k):
# 				# h.append(tf.cond(tf.equal(tf.mod(time,T_i),0),
# 				# 	fn1(h_t, idx),
# 				# 	fn2(h_tm1, idx))
# 				#  )
# 				if time % T_i == 0:
# 					# h[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)] = h_tm1[:,idx*self.block_length:min((idx+1)*self.block_length,self.output_dim)]
# 					h.append(h_t[:,idx*self.block_length:min((idx+1)*self.block_length,self.cell_dimension)])
# 				else:
# 					h.append(h_tm1[:,idx*self.block_length:min((idx+1)*self.block_length,self.cell_dimension)])
			

			
# 			h = tf.concat(h,axis=-1)
# 			print('h.get_shape():',h.get_shape().as_list())
# 			o = tf.matmul(h,self.W_e_o)+tf.reshape(self.b_e_o,[1,self.output_dim])

# 			hidden_states = hidden_states.write(time, o)

# 			return (time+1,hidden_states,h)

		

# 		time = tf.constant(0, dtype='int32', name='time')


# 		feature_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=feature_step,
# 	            loop_vars=(time, hidden_states, initial_state),
# 	            parallel_iterations=32,
# 	            swap_memory=True)

# 		last_output = feature_out[-1] 
# 		hidden_states = feature_out[-2]
# 		if hasattr(hidden_states, 'stack'):
# 			encoder_output = hidden_states.stack()
# 		else:
# 			encoder_output = hidden_states.pack()

# 		encoder_output = tf.reshape(encoder_output,[-1,self.encoder_input_shape[1],self.output_dim])
# 		return last_output, encoder_output

# 	def decoder(self, initial_state, encoder_output):
# 		'''
# 			captions: (batch_size x timesteps) ,int32
# 			d_w2v: dimension of word 2 vector
# 		'''
# 		print('building decoder ... ...')
		
# 		attend_encoder_output = tf.reshape(encoder_output,(-1,self.output_dim))


# 		captions = self.input_captions
# 		mask =  tf.not_equal(captions,0)
# 		loss_mask = tf.cast(mask,tf.float32)

# 		embedded_captions = tf.gather(self.T_w2v,captions)*tf.gather(self.T_mask,captions)

# 		timesteps = self.decoder_input_shape[1]


# 		# batch_size x timesteps x dim -> timesteps x batch_size x dim
# 		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
# 		embedded_captions = tf.transpose(embedded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



# 		input_embedded_words = tf.TensorArray(
# 	            dtype=embedded_captions.dtype,
# 	            size=timesteps,
# 	            tensor_array_name='input_embedded_words')


# 		if hasattr(input_embedded_words, 'unstack'):
# 			input_embedded_words = input_embedded_words.unstack(embedded_captions)
# 		else:
# 			input_embedded_words = input_embedded_words.unpack(embedded_captions)	


# 		# preprocess mask
# 		mask = tf.expand_dims(mask,dim=-1)
		
# 		mask = tf.transpose(mask,perm=axis)

# 		input_mask = tf.TensorArray(
# 			dtype=mask.dtype,
# 			size=timesteps,
# 			tensor_array_name='input_mask'
# 			)

# 		if hasattr(input_mask, 'unstack'):
# 			input_mask = input_mask.unstack(mask)
# 		else:
# 			input_mask = input_mask.unpack(mask)


# 		train_hidden_state = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='train_hidden_state')

# 		def step(x_t,h_tm1):

# 			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
# 			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
# 			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

# 			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r))
# 			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z))
# 			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h))

			
# 			h = (1-z)*hh + z*h_tm1

# 			o = tf.matmul(h,self.W_d_o)+tf.reshape(self.b_d_o,[1,self.output_dim])
			

# 			return h, o

# 		def train_step(time, train_hidden_state, h_tm1, a_tm1):
# 			x_t = input_embedded_words.read(time) # batch_size * dim
# 			mask_t = input_mask.read(time)

# 			x_t = tf.matmul(x_t,self.W_l_i)+tf.matmul(a_tm1,self.W_l_i_a)+tf.reshape(self.b_l_i,[1,self.d_w2v])

# 			# x_t = tf.nn.dropout(x_t,self.dropout) # input dropout

# 			h, o_attn = step(x_t,h_tm1)


# 			tiled_mask_h = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))
# 			h = tf.where(tiled_mask_h, h, h_tm1) # (batch_size, output_dims)

			
# 			ori_feature = tf.reshape(self.input_feature,(-1,self.encoder_input_shape[-1]))
# 			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
# 			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

# 			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
# 			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
# 			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
# 			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)

# 			a_t = self.input_feature * tf.tile(attend_e,[1,1,self.encoder_input_shape[-1]])
# 			a_t = tf.reduce_sum(a_t,reduction_indices=1)

# 			tiled_mask_a = tf.tile(mask_t, tf.stack([1, a_t.get_shape().as_list()[1]]))
# 			a_t = tf.where(tiled_mask_a, a_t, a_tm1) # (batch_size, output_dims)

# 			# classification with attention
# 			# output dropout
# 			# drop_o_attn = tf.nn.dropout(o_attn,self.dropout)
# 			# drop_a_t = tf.nn.dropout(a_t,self.dropout)
# 			# o_dec = tf.matmul(o_attn,self.W_l_o)+tf.matmul(a_t,self.W_l_o_a)+tf.reshape(self.b_l_o,[1,self.output_dim])
# 			o_dec = o_attn+tf.matmul(a_t,self.W_l_o_a)+tf.reshape(self.b_l_o,[1,self.output_dim])
# 			train_hidden_state = train_hidden_state.write(time, o_dec)

# 			return (time+1,train_hidden_state,h,a_t)

		

		 
# 		def get_attention_init_state(x, output_dim):
# 			attn_initial_state = tf.zeros_like(x)
# 			attn_initial_state = tf.reduce_sum(attn_initial_state,axis=[1,2])
# 			attn_initial_state = tf.expand_dims(attn_initial_state,dim=-1)
# 			attn_initial_state = tf.tile(attn_initial_state,[1,output_dim])
# 			return attn_initial_state

# 		time = tf.constant(0, dtype='int32', name='time')

# 		a_0 = get_attention_init_state(encoder_output, self.encoder_input_shape[-1])

# 		train_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=train_step,
# 	            loop_vars=(time, train_hidden_state, initial_state, a_0),
# 	            parallel_iterations=32,
# 	            swap_memory=True)


# 		train_hidden_state = train_out[1]
# 		train_last_output = train_out[-2] 
		
# 		if hasattr(train_hidden_state, 'stack'):
# 			train_outputs = train_hidden_state.stack()
# 		else:
# 			train_outputs = train_hidden_state.pack()

# 		axis = [1,0] + list(range(2,3))
# 		train_outputs = tf.transpose(train_outputs,perm=axis)



# 		train_outputs = tf.reshape(train_outputs,(-1,self.output_dim))
# 		train_outputs = tf.nn.dropout(train_outputs, self.dropout)
# 		predict_score = tf.matmul(train_outputs,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

# 		predict_score = tf.reshape(predict_score,(-1,timesteps,self.voc_size))
# 		# predict_score = tf.nn.softmax(predict_score,-1)


# 		# test phase


# 		test_input_embedded_words = tf.TensorArray(
# 	            dtype=embedded_captions.dtype,
# 	            size=timesteps+1,
# 	            tensor_array_name='test_input_embedded_words')

# 		predict_words = tf.TensorArray(
# 	            dtype=tf.int64,
# 	            size=timesteps,
# 	            tensor_array_name='predict_words')

		
# 		test_input_embedded_words = test_input_embedded_words.write(0,embedded_captions[0])

# 		def test_step(time, test_input_embedded_words, predict_words, h_tm1, a_tm1):
# 			x_t = test_input_embedded_words.read(time) # batch_size * dim

# 			x_t = tf.matmul(x_t,self.W_l_i)+tf.matmul(a_tm1,self.W_l_i_a)+tf.reshape(self.b_l_i,[1,self.d_w2v])
# 			# x_t = x_t*self.dropout

# 			h, o_attn = step(x_t,h_tm1)

# 			ori_feature = tf.reshape(self.input_feature,(-1,self.encoder_input_shape[-1]))
# 			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
# 			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

# 			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
# 			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
# 			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
# 			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)

# 			a_t = self.input_feature * tf.tile(attend_e,[1,1,self.encoder_input_shape[-1]])
# 			a_t = tf.reduce_sum(a_t,reduction_indices=1)

# 			# classification with attention
# 			# output dropout
# 			# drop_o_attn = o_attn*self.dropout
# 			# drop_a_t = a_t*self.dropout
# 			# o_dec = tf.matmul(o_attn,self.W_l_o)+tf.matmul(a_t,self.W_l_o_a)+tf.reshape(self.b_l_o,[1,self.output_dim])
# 			o_dec = o_attn+tf.matmul(a_t,self.W_l_o_a)+tf.reshape(self.b_l_o,[1,self.output_dim])

# 			o_dec = tf.reshape(o_dec,(-1,self.output_dim))
# 			o_dec = o_dec*self.dropout
# 			o_dec = tf.matmul(o_dec,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

# 			o_dec = tf.nn.softmax(o_dec,dim=-1)
# 			predict_word_t = tf.argmax(o_dec,-1)

# 			predict_words = predict_words.write(time, predict_word_t) # output

# 			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t)

# 			test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

			

# 			return (time+1, test_input_embedded_words, predict_words, h, a_t)


# 		time = tf.constant(0, dtype='int32', name='time')

# 		a_0 = get_attention_init_state(encoder_output, self.encoder_input_shape[-1])

# 		test_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=test_step,
# 	            loop_vars=(time, test_input_embedded_words, predict_words, initial_state, a_0),
# 	            parallel_iterations=32,
# 	            swap_memory=True)


# 		predict_words = test_out[-3]
		
# 		if hasattr(predict_words, 'stack'):
# 			predict_words = predict_words.stack()
# 		else:
# 			predict_words = predict_words.pack()

# 		axis = [1,0] + list(range(2,3))

# 		predict_words = tf.transpose(predict_words,perm=[1,0])
# 		predict_words = tf.reshape(predict_words,(-1,timesteps))

# 		return predict_score, predict_words, loss_mask

# 	def build_model(self):
# 		print('building model ... ...')
# 		self.init_parameters()
# 		last_output, encoder_output = self.encoder()
# 		predict_score, predict_words , loss_mask= self.decoder(last_output, encoder_output)
# 		return predict_score, predict_words, loss_mask