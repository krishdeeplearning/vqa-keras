import numpy as np
import json
import h5py
import os
from constants import *

from keras.utils.np_utils import to_categorical

def padding_for_RNN(seq, lengths):
	v = np.zeros(np.shape(seq))
	N = np.shape(seq)[1]
	for i in range(np.shape(seq)[0]):
	    v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
	return v

def prepare_train(limit):
	print "Starting to prepare training data"
	img_data = h5py.File(data_img)
	ques_data = h5py.File(data_prepo)

	# Lets look at the keys
	print img_data.keys()
	print ques_data.keys()

	# setting up images
	img_data = np.array(img_data['images_train'])
	img_pos_train = ques_data['img_pos_train'][:limit]
	train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])

	tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
	train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))
	
	ques_train = np.array(ques_data['ques_train'])[:limit, :]
	ques_length_train = np.array(ques_data['ques_length_train'])[:limit]
	
	ques_train = padding_for_RNN(ques_train, ques_length_train)

	train_X = [train_img_data, ques_train]
	train_y = to_categorical(ques_data['answers'])[:limit, :]
	return train_X, train_y

def get_val_data():
	img_data = h5py.File(data_img)
	ques_data = h5py.File(data_prepo)
	meta_data = metadata()
	with open(val_annotations_path, 'r') as an_file:
	    annotations = json.loads(an_file.read())

	img_data = np.array(img_data['images_test'])
	img_pos_train = ques_data['img_pos_test']
	train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
	tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
	train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))

	ques_train = np.array(ques_data['ques_test'])
	ques_length_train = np.array(ques_data['ques_length_test'])
	ques_train = right_align(ques_train, ques_length_train)

	# Convert all last index to 0, coz embeddings were made that way :/
	for _ in ques_train:
	    if 12602 in _:
	        _[_==12602] = 0

	val_X = [train_img_data, ques_train]

	ans_to_ix = {str(ans):int(i) for i,ans in meta_data['ix_to_ans'].items()}
	ques_annotations = {}
	for _ in annotations['annotations']:
	    idx = ans_to_ix.get(_['multiple_choice_answer'].lower())
	    _['multiple_choice_answer_idx'] = 1 if idx in [None, 1000] else idx
	    ques_annotations[_['question_id']] = _

	abs_val_y = [ques_annotations[ques_id]['multiple_choice_answer_idx'] for ques_id in ques_data['question_id_test']]
	abs_val_y = to_categorical(np.array(abs_val_y))

	multi_val_y = [list(set([ans_to_ix.get(_['answer'].lower()) for _ in ques_annotations[ques_id]['answers']])) for ques_id in ques_data['question_id_test']]
	for i,_ in enumerate(multi_val_y):
	    multi_val_y[i] = [1 if ans in [None, 1000] else ans for ans in _]

	return val_X, abs_val_y, multi_val_y


def metadata():
	meta_data = json.load(open(data_prepo_meta, 'r'))
	meta_data['word_to_ix'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
	return meta_data

def prepare_embeddings():
	if os.path.exists(embedding_matrix_filename):
	    with h5py.File(embedding_matrix_filename) as f:
	        return np.array(f['embedding_matrix'])
	
