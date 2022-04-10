import pandas as pd
import torch
import torch.nn as nn
import dictionary_corpus
from dictionary_corpus import Corpus
import numpy as np
# from sklearn.decomposition import PCA, SparsePCA, KernelPCA, IncrementalPCA
import matplotlib.pyplot as plt
import matplotlib


# matplotlib.use('webagg')
# load the model and the corpus
model = torch.load('hidden650_batch128_dropout0.2_lr20.0.pt',map_location=torch.device('cpu'))
corpus = Corpus('')
# print("Vocab size %d", ntokens)

def get_init_emb(words, model, corpus):
	# def feature extraction function
	with torch.no_grad():
		tokenized_words = []
		for word in words:
			if word in corpus.dictionary.word2idx.keys():
				wordidx = corpus.dictionary.word2idx[word]
			else:
				print(f'{word} is to be deleted')
				wordidx = corpus.dictionary.word2idx['<unk>']
			tokenized_words.append(wordidx)

		# initailize the hidden state of the model
		hidden = model.init_hidden(1)
		# for printing
		print(words, '\ntokenized as',tokenized_words)

		# previous_inputs = []
		words_features = {}
		# iterate through the list
		for i, wordid in enumerate(tokenized_words):
			hidden = model.init_hidden(1)
			# print(hidden)
			word = corpus.dictionary.idx2word[wordid]

			# we are not gonna use the output
			_, _, emb = model(torch.as_tensor(wordid).reshape(1,1), hidden)
			
			# the init embedding is the embedding we wanted.
			emb = emb.view(650).numpy()
			
			words_features[str(99-i)+word] = emb
		df = pd.DataFrame(data=words_features).T
		print(df)
		return df

words = ['bank','potato',]
get_init_emb(words,model,corpus)