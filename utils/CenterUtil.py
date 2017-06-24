import numpy as np 
import math

from sklearn.cluster import KMeans
from six.moves import cPickle
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from numpy import linalg as LA
import sys, os
import h5py

def get_centers(center_file, hf, sample_dim, k_clusters):
	

	if(not os.path.isfile(center_file)):


		max_iter = 100
		random_state = 47		
		n_samples = 1300*200
		print(n_samples)
		sampled_descriptor = np.zeros((n_samples,sample_dim))

		for idx, vid in enumerate(range(1,1300)):
			vid='vid'+str(vid)
			print('%d, vid: %s' %(idx,vid))

			loaded_video_feature = hf[vid]
			loaded_video_feature = np.asarray(loaded_video_feature).reshape(-1,sample_dim,7,7)

			loaded_video_feature = np.swapaxes(loaded_video_feature,1,3)
			descriptors = loaded_video_feature.reshape(-1,sample_dim)

			des_num = len(descriptors)
			des_indexs = np.random.randint(0,des_num,size=200)
			sampled_descriptor[idx*200:(idx+1)*200] = descriptors[des_indexs]

		

		kmeans = KMeans(n_clusters=k_clusters, random_state=random_state, n_jobs=2, max_iter=max_iter ).fit(sampled_descriptor)
		centers = kmeans.cluster_centers_
		nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(centers)

		distances, indices = nbrs.kneighbors(sampled_descriptor)
		ave_dis = np.mean(distances[:,1]-distances[:,0])+sys.float_info.epsilon
		
		output = open(center_file, 'wb')
		cPickle.dump((centers,ave_dis),output,protocol=2)
		output.close()
	else:
		f = open(center_file, 'rb')
		(centers,ave_dis) = cPickle.load(f)
		f.close()
	
	
	alpha = -math.log(0.01)/ave_dis



	init_w = alpha*centers
	init_b = -alpha*np.mean(centers**2,axis=1)

	init_centers = centers
	
	print(init_w.shape)
	print(init_b.shape)
	print(init_centers.shape)

	return (init_w,init_b,init_centers)








