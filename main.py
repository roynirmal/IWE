import helper, dictionary, json, itertools, torch, training, math
import numpy as np
from model import WordEmbIWE


''' Load Data and Features '''

## input document from WPQA
docs = json.load(open('./data/document_passages.json'))
example_doc = docs['624']
stop = json.load(open('./data/SMARTstop.json'))
doc = []
for value in example_doc.values():
    doc.append(helper.split_into_sentences(value))
text = list(itertools.chain.from_iterable(doc))
textLower = [t.lower() for t in text]

## input the rootlist as mentioned in https://github.com/ShelsonCao/IWE/blob/master/Data/features/rootList.txt
rootList = open("./features/rootList.txt", 'r')
roots = []
for root in rootList:
    roots.append(root[:-1])
roots[0] = 'age'
# print(roots)

word2Root = open("./features/word2Roots.txt", 'r')
word2RootMap = {}
for item in word2Root:
    word2RootMap[item.split()[0]] = item.split()[1].split(",")

data = dictionary.Corpus()

train = list(data.read_doc(textLower, stop)) 
i2w = {v: k for k, v in data.w2i.items()} ##stores index to words


'''Build the input representation using the features '''
root_dictionary = dictionary.Root_Dictionary(word2RootMap)
root_dictionary.build_dict(textLower,stop)
root_rep = root_dictionary.build_input_feature(data.w2i)

tri_dictionary = dictionary.Tri_Dictionary()
tri_dictionary.build_dict(textLower, stop)

tri_rep = tri_dictionary.build_input_feature(data.w2i)
input_rep = dict([(k, tri_rep[k].tolist()+root_rep[k].tolist()) for k in tri_rep])

''' Build the model'''

K = 20 # number of negative samples per target word
N = 5 # length of context on the left side, do i have to try with context on both sides, ik denk zo? 
emb_dim = 100
feature_dim= len(input_rep[0]) # feature size of input representation; for one hot encoding feature_dim = vocab_size
window_size = 3
nwords = len(data.w2i)

model = WordEmbIWE(feature_dim, emb_dim, window_size, feature_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.02) 

type = torch.FloatTensor
use_cuda = torch.cuda.is_available()


''' Send model and data for training '''

train_epoch = training.Train(model, optimizer, train, data.w2i, K, N, nwords, input_rep)
train_epoch.train_model(1)

''' Write output '''

helper.writeEmbedding("./embeddings/624_emb.txt", data.w2i, train_epoch.word_embeddings)
helper.writeWords("./embeddings/624_word.txt", data.w2i, i2w)