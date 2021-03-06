import helper,string
import numpy as np
from collections import Counter, defaultdict
from random import sample

## 
class Corpus(object):
	def __init__(self):
		self.w2i = defaultdict()

	def read_doc(self, document,stop):
		self.w2i = defaultdict(lambda: len(self.w2i))
		S = self.w2i["<s>"]
		UNK = self.w2i["<unk>"]
		c = 0
		for sentence in document:
			c += 1
			if(c%100000 ==0):
				print(c, " sentences has been read and turned into index")
			# for words in sentence.strip().split(" "):
			#     if not words in stop:
			#         word_counts[w2i[words]] +=1 
			table = str.maketrans('', '', string.punctuation)
			tokens = [x.translate(table) for x in sentence.split(" ") if not x in stop]
			# print(tokens)
			tokens = sample(tokens, len(tokens)) 
			# print("after shuffle", tokens		)
			# tokens = [x.translate(table) for x in sentence.split(" ")]
			yield[self.w2i[y] for y in tokens if y.isalpha()]

	def one_hot(self, w2i, nwords):
		ohv = {}
		for word in w2i:
			onehotvec = np.zeros(nwords)
			onehotvec[w2i[word]] = 1
			ohv[w2i[word]] = onehotvec
		return ohv


## Dictionary of Root Afflixes Features
class Root_Dictionary(object):
	def __init__(self, word2RootMap):
		self.root2idx = {}
		self.idx2root = []
		self.word_rep_root = {}
		self.root_map = word2RootMap
	
	def build_dict(self, document,stop):
		
		roots_in_doc = []
		for s in document:
			table = str.maketrans('', '', string.punctuation)
			# content_term = [x.translate(table) for x in s.split(" ") if not x in stop]
			content_term = [x.translate(table) for x in s.split(" ")]
			for i in range(len(content_term)):
				# if not content_term[i] in stop:
				## just pick up roots of the words in the document
				if content_term[i] in self.root_map.keys():
					for j in self.root_map[content_term[i]]:
						roots_in_doc.append(j)
				else:
					self.root_map[content_term[i]] = [content_term[i]]
					roots_in_doc.append(content_term[i])
		
		root_count = Counter()
		root_count.update(roots_in_doc)

		most_common_root = root_count.most_common()
		for (index, w) in enumerate(most_common_root):
			self.idx2root.append(w[0])
			self.root2idx[w[0]] = len(self.idx2root) - 1        
	
	
	def build_input_feature(self, w2i):
		
		for word in w2i:
			one_word = np.zeros(len(self.root2idx))
			if word in self.root_map.keys():
				for root in self.root_map[word]:
					one_word[self.root2idx[root]] = 1
				self.word_rep_root[w2i[word]] = one_word
			else:
				self.word_rep_root[w2i[word]] = one_word

		return self.word_rep_root


## Dictionary of letter trigram features

class Tri_Dictionary(object):
	def __init__(self):
		self.tri2idx = {}
		self.idx2tri = []
		self.tri_lookup = {}
		self.word_rep = {}
	
	def build_dict(self, document,stop):
		all_terms = []
		
		for s in document:    
			content_terms = s.split(" ")
			for i in range(len(content_terms)):
				# if not content_terms[i] in stop:
				term = '#' + content_terms[i] + '#'
				tmp = []
				for j in range(0, len(term) - 2):
					tmp.append(term[j:j + 3])
					all_terms.append(term[j:j + 3])
				self.tri_lookup[content_terms[i]] = tmp


		word_count = Counter()
		max_words  = 0
		word_count.update(all_terms)

		most_common = word_count.most_common(max_words) if max_words > 0 else word_count.most_common()
		for (index, w) in enumerate(most_common):
			self.idx2tri.append(w[0])
			self.tri2idx[w[0]] = len(self.idx2tri) - 1
			
	def build_input_feature(self, w2i):
		# print(len(self.tri2idx))
		# print(len(w2i))
		for word in w2i:
			one_word = np.zeros(len(self.tri2idx))
			if word in self.tri_lookup:
				for tri in self.tri_lookup[word]:
					one_word[self.tri2idx[tri]] = 1
				self.word_rep[w2i[word]] = one_word
			else:
				self.word_rep[w2i[word]] = one_word

		return self.word_rep


''' To Do - Add Inflictional Afflixes Feature '''
