import pandas as pd
from dictionary_corpus import Corpus
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some words.')
parser.add_argument('w')

args = parser.parse_args()

# load the model and the corpus

corpus = Corpus('')
# print("Vocab size %d", ntokens)

def check_lexeme(wordlist):
	in_list = []
	out_list = []

	for word in wordlist:
		if word in corpus.dictionary.word2idx.keys():
			in_list.append(word)
		else:
			out_list.append(word)
	
	if len(in_list)>0:
		print(in_list, ' in the vocab.')	
	if len(out_list)>0:
		print(out_list, ' not in the vocab.')
	return
word_list=[]
# word_list = ['apple','banana','avocado','pepper','pear','pineapple','celery','onion',\
# 'cabbage','garlic','tomato','carrot','peanut','cotton','watermelon','grape','olive',\
# 'cucumber','melon','walnut','peach','cherry','topped','millet','chopped','localities','villages','studying']

word_list.append(args.w)
check_lexeme(word_list)