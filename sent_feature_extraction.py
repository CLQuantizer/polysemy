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
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

matplotlib.use('webagg')
# load the model and the corpus
model = torch.load('hidden650_batch128_dropout0.2_lr20.0.pt',map_location=torch.device('cpu'))
corpus = Corpus('')
# print("Vocab size %d", ntokens)

def get_init_emb(words, model, corpus):
	# def initial emb feature extraction function
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
		print(words, 'for inital embed, \ntokenized as',tokenized_words)

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
			# 99 adds an identifier for pandas
			words_features[str(99-i)+word] = emb

		df = pd.DataFrame(data=words_features).T
		return df

def sent_feature_extraction(sent, model, corpus):
	# def feature extraction function
	# input = sentence
	with torch.no_grad():
		# tokenize the input sentence
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

		# previous_inputs = []
		sent_features = {}
		# iterate through the whole sentence

		for i, wordid in enumerate(tokenized_sent):
			word = corpus.dictionary.idx2word[wordid]

			# we are not gonna use the output
			output, hidden, emb = model(torch.as_tensor(wordid).reshape(1,1), hidden)
			
			# the hidden embedding is the embedding we wanted.
			hidden_embedding = hidden[0][1].view(650).numpy()

			# declare Softmax
			# m = nn.Softmax(dim=1)
			# apply softmax to output 
			# t = m(output.view(-1,50001))
			# specify dim1
			# t = t.reshape(1,50001)
			# _ , id= torch.max(t,1)
			# previous_inputs.append(word)
			# print('With input = ', previous_inputs ,'\nModel predicts next:', corpus.dictionary.idx2word[id])

			sent_features[str(99-i)+word] = hidden_embedding

		df = pd.DataFrame(data=sent_features).T
		# print(df)
		return df

# extract feature out of sentences
# sent_feature_extraction('I kicked some ducks', model, corpus)	

def get_target_features(file, target_word,model=model):
	# get features of a sentence (including target words) from a file containing lines of sentences
	# with open('chicken_meat.txt','r') as f:

	with open(file,'r') as f:
		target_features = []
		contexts = []
		for i,line in enumerate(f):
			print(f'Sentence {i}:')

			# extract the features of the sentence return df
			sent_feature = sent_feature_extraction(line,model,corpus)
			# find the feature of the target word

			# for plot labelling
			sentence = [word[2:] for word in list(sent_feature.index)]
			# print(sentence)
			
			# capture the context
			context = ''
			for i, word in enumerate(sentence):
				if word == target_word:
					context = sentence[i-1] + ' ' + word[0] + word[-1]
					target_vector = np.array(sent_feature.loc[str(99-i)+word])
				# else:
				# 	print(word, ' not in ', target_words, ' FATAL WARNING!')
			contexts.append(context)
			target_features.append(target_vector)

	# for each in target_features:
	# 	print(f'there are {len(each[each<0])} numbers < 0 in there')
	# 	print(f'there are {len(each[each<-1])} numbers < -1 there')
	# 	print(f'there are {len(each[each>1])} numbers > 1 there')

	target_features = np.array(target_features)
	mean = np.mean(target_features,axis=0)

	return target_features, contexts, mean

def plot_a_regularity(text_files,target_words,ax,corpus=corpus,model=model):
	# ax is a subplot of the plot
	target_and_label_list = []
	target_feature_list = []
	label_list = []
	# mean embedding for each file
	mean_emb_list = []
	# initial embbeding for target word in each file 
	init_emb_list = get_init_emb(target_words,model,corpus)

	df = init_emb_list.to_numpy()
	# for each in df:
		# print(f'ggg there are {len(each[each<0])} numbers < 0 in there')
		# print(f'ggg there are {len(each[each==0])} numbers = 0 in there')
		# print(f'there are {len(each[each<-1])} numbers < -1 there')
		# print(f'there are {len(each[each>1])} numbers > 1 there')

	# list of tuple contating the word and distance of each mean embedding and initial embedding
	distance_list = []

	# control the number of sentences processed is correct
	for file,keys in zip(text_files,target_words):
		assert len(text_files) == len(target_words)
		# get target returns target_features (list), contexts (string), and mean (np.array)
		target_and_label_list.append(get_target_features(file,keys,model))

	# create target feature list, correspinding to the label as well
	for each in target_and_label_list:
		# label is just a list of strings
		label_list+=each[1]
		mean_emb_list.append(each[2])
		# target_featue_list is a list of file number * list, each file contains a number of vectors
		target_feature_list.append(each[0])

	# print('the lenth of initial embedding is: ',len(init_emb_list))
	# print('length of target and label list is: ', len(target_and_label_list))

	# calculate the mean vector bewtween the mean and the init_emb 
	# every .txt file generates a vector, forming a list
	# also get a copy of initial embediing
	
	init_emb_and_label = []
	for i in range(len(mean_emb_list)):
		init_emb_vec = np.array(init_emb_list.iloc[i])
		label = str('init_'+target_words[i])
		if label not in [x[1] for x in init_emb_and_label]:
			init_emb_and_label.append((np.array([init_emb_vec]),label))
		distance = mean_emb_list[i] - init_emb_vec
		distance_list.append((target_words[i],distance))

	# print('the target feature list length is: ' , label_list )
	# update the label_list and the target_feature_list to include the central shit
	for each in init_emb_and_label:
		label_list.append(each[1])
		target_feature_list.append(each[0])
	# print('the updated target feature list length is: ' , label_list )

	target_feature_tuple = tuple(target_feature_list)
	# stack all the features
	tar = np.vstack(target_feature_tuple)
	# print(tar.shape)

	# Apply PCA to the features
	pca = PCA(n_components=2)
	pca.fit(tar)
	tar = pca.transform(tar)
	# print('The shape of the bundled embeddings are: ',tar.shape)

	# get the sentence groupping marks
	demarcations = [0]
	for i, each in enumerate(target_feature_list):
		if i == 0:
			# implicitly target_feature_list[i] is a numpy array, fuck
			mark =  target_feature_list[i].shape[0]
		else:
			mark+= target_feature_list[i].shape[0]
		demarcations.append(mark)

	# print('demarcations are: ', demarcations)
	# print('size of label list is ', len(label_list))

	for i, mark in enumerate(demarcations):
		if i == 0:
			continue
		else:
			# print(f'the demarcations are: {mark} and {demarcations[i-1]}.' )
			for j in range(demarcations[i-1],mark):
				x,y = tar[j]
				label = label_list[j]
				# annotate the data points with contexts for intepretation
				ax.annotate(label, (x,y))
				if label[:5] == 'init_':
					ax.scatter(x,y, color='magenta')
				elif i%2==0:
					ax.scatter(x,y, color='darkcyan')
				else:
					ax.scatter(x,y, color='lightpink')
	return distance_list
# Below is the expriment

animal_files = ['chicken_meat.txt','chicken_animal.txt','fish_meat.txt',\
'fish_animal.txt','salmon_meat.txt','salmon_animal.txt','lamb_meat.txt',\
'lamb_animal.txt',]
animal_words =[ 'chicken','chicken','fish','fish','salmon', 'salmon','lamb','lamb',]

Switzerland_files = ['Switzerland_1.txt','Switzerland_2.txt','transistor_1.txt','transistor_2.txt']
Switzerland_words = ['Switzerland','Switzerland','transistor','transistor',]

plant_files = ['tomato_plant.txt', 'tomato_dish.txt','potato_plant.txt','potato_dish.txt']
plant_words = ['tomato','tomato','potato','potato']

power_files = ['power_poli.txt','power_J.txt']
power_words = ['power','power',]

bank_files = ['bank_ins.txt','bank_river.txt',]
bank_words = ['bank','bank',]

pupil_files = ['pupil_student.txt','pupil_eye.txt',]
pupil_words = ['pupil','pupil']

watch_files = ['watch_look.txt','watch_time.txt',]
watch_words = ['watch','watch',]

can_files = ['can_soda.txt','can_cannot.txt',]
can_words = ['can','can',]

computer_files = ['computer_1.txt','computer_2.txt',]
computer_words = ['computer','computer',]


exp_files = ['exp_1.txt','exp_2.txt']
exp_words = ['line','line',]

# for small sub plots
fig = plt.figure(figsize=(12, 6.5))
ax1 = plt.subplot(3,3,1)
ax2 = plt.subplot(3,3,2)
ax3 = plt.subplot(3,3,3)
ax4 = plt.subplot(3,3,4)
ax5 = plt.subplot(3,3,5)
ax6 = plt.subplot(3,3,6)
ax7 = plt.subplot(3,3,7)
ax8 = plt.subplot(3,3,8)
ax9 = plt.subplot(3,3,9)

# for big subplots
# fig = plt.figure(figsize=(12, 6.5))
# ax1 = plt.subplot(1,2,1)
# ax2 = plt.subplot(1,2,2)
# ax3 = plt.subplot(1,3,3)

# for small plots
new = []
new += plot_a_regularity(Switzerland_files,Switzerland_words,ax1)
new += plot_a_regularity(computer_files,computer_words,ax2)
new += plot_a_regularity(animal_files,animal_words,ax3)
new += plot_a_regularity(plant_files,plant_words,ax4)
new += plot_a_regularity(bank_files,bank_words,ax5)
new += plot_a_regularity(power_files, power_words,ax6)
new += plot_a_regularity(pupil_files,pupil_words,ax7)
new += plot_a_regularity(watch_files,watch_words,ax8)
new += plot_a_regularity(can_files,can_words,ax9)


# for big plots
# new = []
# new += plot_a_regularity(exp_files,exp_words, ax1)
# new += plot_a_regularity(power_files, power_words,ax1)
# new += plot_a_regularity(pupil_files,pupil_words,ax2)
# new += plot_a_regularity(watch_files,watch_words,ax1)
# new += plot_a_regularity(bank_files,bank_words,ax2)

# print the first 2 numbers in 650 dim vectors
# for each in new:
# 	print(each[0],each[1][:2])

# used means used vector
# new is a list of tuples contating 2 elements, first is the word, second is the distance vector

cos_dict = defaultdict()
eclid1_dict = defaultdict()
eclid2_dict = defaultdict()
eclid_dict = defaultdict()

for i in range(len(new)):
	if new[i][0] not in cos_dict.keys():
		for j in range(i+1,len(new)):
			if new[j][0]==new[i][0]:
				cos_sim = np.dot(new[i][1],new[j][1])/(np.linalg.norm(new[i][1]) * np.linalg.norm(new[j][1]))
				cos_sim = np.arccos(cos_sim)/3.1415926*180
				euclidean_dist = np.linalg.norm(new[i][1] - new[j][1])
				path1euc = np.linalg.norm(new[i][1])
				path2euc = np.linalg.norm(new[j][1])

				cos_dict[new[i][0]] = cos_sim
				eclid_dict[new[i][0]] = euclidean_dist
				eclid1_dict[new[i][0]] = path1euc
				eclid2_dict[new[i][0]] = path2euc
			else:
				pass
			# row_dict[new[j][0]] = cos_sim
			# print(f'angle between path1 and path2 of {new[i][0]} is {cos_sim} degree')
			# print(f'euclidean distance between path1 and path2 of {new[i][0]} is {euclidean_dist}')
			# print(f'euclidean distance between path1 and root of {new[i][0]} is {path1euc}')
			# print(f'euclidean distance between path2 and root of {new[i][0]} is {path2euc}')
			# print()
		attribute_matrix = defaultdict(defaultdict)

		for key in cos_dict.keys():
			attribute_matrix[key]['cos'] = cos_dict[key]
			attribute_matrix[key]['dist'] = eclid_dict[key]
			attribute_matrix[key]['path1'] = eclid1_dict[key]
			attribute_matrix[key]['path2'] = eclid2_dict[key]

df = pd.DataFrame.from_dict(attribute_matrix).round(2)
df.T.to_csv('df.csv')
print(df)
plt.show()





