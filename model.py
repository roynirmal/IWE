

import torch


class WordEmbIWE(torch.nn.Module):
    def __init__(self, feature_dim, emb_dim, window_size, hidden_size):
        super(WordEmbIWE, self).__init__()

        """ word embeddings """
        
        self.generate_word_embedding = torch.nn.Sequential(torch.nn.Linear(feature_dim, hidden_size), torch.nn.Tanh(), 
                                                           torch.nn.Linear(hidden_size, emb_dim), torch.nn.Tanh()).cuda(0)
        
        """ context embeddings"""
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=feature_dim, out_channels=emb_dim, kernel_size=window_size,
                                        stride=1, padding=int((window_size-1)/2), dilation=1, groups=1, bias=True).cuda(1)
        


    # useful ref: https://arxiv.org/abs/1402.3722
    def forward(self, sample_word, context_words):
        ## context_words dimension = no_of_context_words x feature_dim
        
        con = context_words.unsqueeze(0).permute(0, 2, 1) # 1 x feature_dim x no_of_context_words 
        context_conv = self.conv_1d(con) # 1 x emb_dim x no_of_context_words 
        
        context_rep = context_conv.max(dim=2)[0] # 1 x emb_dim 1 x 100
        ## target_word_dim = 1 x feature_dim
        
        word_rep = self.generate_word_embedding(sample_word)# (target_word + neg_words) x emb_dim
        context_rep = context_rep.cuda(0)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08) # eps = to avoid divisibility by zero
        sim_layer = cos(context_rep, word_rep) # similarity of each word with the context representation 1 x (target_word + neg_words)
        softmax = torch.nn.LogSoftmax(dim = 0)
        obj = softmax(sim_layer)
        
        
        return  -obj[0], word_rep