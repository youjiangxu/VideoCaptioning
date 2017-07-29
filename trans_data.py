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
		

if __name__ == '__main__':


	
	feature_path = '/data/msrvtt/ResNet200-res5c-relu-f40.h5'
	rf = h5py.File(feature_path,'r')

	output_path = '/data/msrvtt/ResNet200-res5c-relu-f10.h5'

	wf = h5py.File(output_path,'w')
	feature_shape = (10,2048,7,7)
	for idx, vid in enumerate(rf.keys()):
		print(idx,vid)
		det = wf.create_dataset(vid,feature_shape,dtype='f')
		det[:]=rf[vid][0::4]


	
	

	
	
	
	


	
