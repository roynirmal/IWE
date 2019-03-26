

import torch


class WordEmbIWE(torch.nn.Module):
    def __init__(self, feature_dim, emb_dim, window_size, hidden_size):
        super(WordEmbIWE, self).__init__()

        """ word embeddings """
        
        # self.generate_word_embedding = torch.nn.Sequential(torch.nn.Linear(feature_dim, hidden_size), torch.nn.Tanh(), 
                                                           # torch.nn.Linear(hidden_size, emb_dim), torch.nn.Tanh())
        self.generate_word_embedding = torch.nn.Sequential(torch.nn.Linear(feature_dim, emb_dim), torch.nn.Tanh())
        """ context embeddings"""
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=feature_dim, out_channels=emb_dim, kernel_size=window_size,
                                        stride=1, padding=0, dilation=1, groups=1, bias=True)
        


    # useful ref: https://arxiv.org/abs/1402.3722
    def forward(self, sample_word, context_words):
        ## context_words dimension = no_of_context_words x feature_dim
        # print(sample_word)
        con = context_words.unsqueeze(0).permute(0, 2, 1) # 1 x feature_dim x no_of_context_words 
        context_conv = self.conv_1d(con) # 1 x emb_dim x no_of_context_words 
        
        context_rep = context_conv.max(dim=2)[0] # 1 x emb_dim 
        # print(context_rep)
        ## sample_words_dim = (target_word + neg_words) x feature_dim
        word_rep = self.generate_word_embedding(sample_word)# (target_word + neg_words) x emb_dim
        # print(word_rep)
        # context_rep = context_rep.cuda(0)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08) # eps = to avoid divisibility by zero
        sim_layer = 100*cos(context_rep, word_rep) 
        # print(word_rep[0])
        # print(word_rep[1])


        # print(cos(context_rep, word_rep[0]))
        # print(cos(context_rep, word_rep[1]))
        # print(sim_layer)

        # print(sim_layer.sum())

        # similarity of each word with the context representation 1 x (target_word + neg_words)
        softmax = torch.nn.LogSoftmax(dim = 0)
        # softmax1 = torch.nn.LogSoftmax(dim=0)
        obj = softmax(sim_layer)
        # obj1 = softmax1(sim_layer)
        # print(obj)
        # print(obj1)        

        
        return  -obj[0], word_rep	