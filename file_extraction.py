import pandas as pd
import torch
import torch.nn as nn
import dictionary_corpus
from dictionary_corpus import Corpus
import numpy as np
from sklearn.decomposition import PCA, SparsePCA, KernelPCA, IncrementalPCA
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import os

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

matplotlib.use('webagg')
# load the model and the corpus
model = torch.load('hidden650_batch128_dropout0.2_lr20.0.pt',map_location=torch.device('cpu'))
corpus = Corpus('')
# print("Vocab size %d", ntokens)

def get_init_emb(file, model=model, corpus=corpus):
	# def initial emb feature extraction function
	tmp = file[5:]
	target_word = tmp.split('_')[0]
	file_type = tmp.split('_')[1]
	with torch.no_grad():
		wordid = corpus.dictionary.word2idx[target_word]
	
		# initailize the hidden state of the model
		hidden = model.init_hidden(1)
		_, free_hid, emb = model(torch.as_tensor(wordid).reshape(1,1), hidden)
			
		# the init embedding and the free_hid is the embedding we wanted.
		emb = emb.view(650).numpy()
		free_hid = free_hid[0][1].view(650).numpy()
		df = {'labels':['_init '+ target_word+file_type,'_freeh '+ target_word+file_type],'tensors':[emb,free_hid]}
		return df

def sent_feature_extraction(sent,target_word,file_type,model=model, corpus=corpus): 

	with torch.no_grad():
		words = sent.split()
		tokenized_sent=[]

		for word in words:
			if word in corpus.dictionary.word2idx.keys():
				wordidx = corpus.dictionary.word2idx[word]
			else:
				print(f'{word} is to be deleted')
				wordidx = corpus.dictionary.word2idx['<unk>']
			tokenized_sent.append(wordidx)

		# initailize the hidden state of the model
		hidden = model.init_hidden(1)

		# for printing
		print(sent, '\ntokenized as',tokenized_sent)

		# iterate through the whole sentence
		for i, wordid in enumerate(tokenized_sent):
			word = corpus.dictionary.idx2word[wordid]
			# we are not gonna use the output
			_,hidden,_= model(torch.as_tensor(wordid).reshape(1,1), hidden)
			# the hidden embedding is the embedding we wanted.
			hidden_embedding = hidden[0][1].view(650).numpy()
			
			if word == target_word:
				df = {'label':words[i-1]+' '+target_word+file_type,'tensor':hidden_embedding}
				break
		return df

def get_target_features(file,model=model):
	if '_' not in file:
		print('wrong file name, fucked up')
		exit()
	# file name must be in the form of chicken_animal.txt
	# because 'data/' are literally 5 charactes 
	tmp = file[5:]

	target_word = tmp.split('_')[0]
	print(target_word)
	file_type = tmp.split('_')[1]
	# get tar word features from sentences in a file containinglines of sentences
	init_df = get_init_emb(file)
	df = {'tensors':[],'labels':[]}
	with open(file,'r') as f:
		for i,line in enumerate(f):
			if len(line)<3:
				continue
			else:
			# extract the features of the sentence return df
				sent_feature = sent_feature_extraction(line,target_word,file_type)
				df['tensors'].append(sent_feature['tensor'])
				df['labels'].append(sent_feature['label'])

	mean = np.mean(df['tensors'],axis=0)
	return df,mean,init_df

# import required module
directory = 'data'

df = pd.DataFrame()
# iterate over files in that directory
for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	# checking if it is a file
	if os.path.isfile(f):
		a,b,c = get_target_features(f)
		df1 = pd.DataFrame.from_dict(a)
		df2 = pd.DataFrame.from_dict(c)
		df3 = pd.concat([df1,df2])
		df = pd.concat([df,df3]).reset_index().drop(['index'],axis=1)

print(df)