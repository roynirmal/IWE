import time, torch, helper, math
import numpy as np
class Train:
    def __init__(self, model, optimizer, train, w2i, K, N, nwords, input_rep, cuda):
        self.model = model
        self.optimizer = optimizer
        self.train = list(filter(None, train))
        self.w2i = w2i
        self.word_embeddings = {v: np.random.uniform(-1,1,100) for k, v in self.w2i.items()}
        self.K = K
        self.N = N
        self.nwords = nwords
        self.input_rep = input_rep
        self.cudadevice = cuda
        self.filters = None
        self.type = torch.FloatTensor
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            self.type = torch.cuda.FloatTensor
            self.model.cuda(self.cudadevice)
        ## Function to Calculate Sentence Loss
    def calc_sent_loss(self, sent,  BothSides, LastEpoch):
        #check if we are taking context from both sides
        if BothSides:
        # add padding to the sentence beginning and end equal to the size of the window
            padded_sent = [0] * self.N + sent + [0] * self.N



            # Step through the sentence
            losses = []
            for i in range(self.N, len(sent) + self.N+1):
                neg_words = np.random.choice(self.nwords, size=self.K, replace=True)

                context_words = padded_sent[i-self.N:i] + padded_sent[i + 1:i + self.N + 1]
                context_words_tensor= torch.tensor([self.input_rep[x] for x in context_words]).cuda(self.cudadevice).FloatTensor

                target_word = padded_sent[i]
        #         neg_words = all_neg_words[(i-N)*K:(i-N+1)*K]

                sample_words_tensor= torch.tensor([self.input_rep[target_word]]+ 
                                                  [self.input_rep[x] for x in neg_words]).cuda(self.cudadevice).FloatTensor

                loss, word_emb, self.filters = self.model(sample_words_tensor, context_words_tensor, LastEpoch)

                # save the word embedding at the last epoch
                if LastEpoch:
                    # print(self.filters.shape)

                    for c, word in enumerate([target_word]+[x for x in neg_words]):
                        self.word_embeddings[word] = word_emb.data.cpu().numpy()[c]

                losses.append(loss)
        else:
                # add padding to the sentence beginning equal to the size of the window
            padded_sent = [0] * self.N + sent + [0]



            # Step through the sentence
            losses = []
            for i in range(self.N, len(sent) +self. N+1):
                neg_words = np.random.choice(self.nwords, size=self.K, replace=True)

                context_words = padded_sent[i-self.N:i]
                context_words_tensor= torch.tensor([self.input_rep[x] for x in context_words]).cuda(self.cudadevice)

                target_word = padded_sent[i]
        #         neg_words = all_neg_words[(i-N)*K:(i-N+1)*K]

                sample_words_tensor= torch.tensor([self.input_rep[target_word]]+ 
                                                  [self.input_rep[x] for x in neg_words]).cuda(self.cudadevice)

                loss, word_emb, self.filters = self.model(sample_words_tensor, context_words_tensor, LastEpoch)
                # save the word embedding at the last epoch
                if LastEpoch:
                    for c, word in enumerate([target_word]+[x for x in neg_words]):
                        self.word_embeddings[word] = word_emb.data.cpu().numpy()[c]
                        

                losses.append(loss)

        return torch.stack(losses).sum()

    def train_model(self, epoch, doc):


        for ITER in range(epoch):
            print("started iter %r of document %r" % (ITER, doc))
            # Perform training
        #     random.shuffle(train)
            train_words, train_loss = 0, 0.0
            start = time.time()
            for sent_id, sent in enumerate(self.train):
                if(sent_id%100000 ==0):
                    print("Finished %r sentences for iteration %r" %(sent_id, ITER))

                if ITER == epoch-1:
                    my_loss = self.calc_sent_loss(sent,  BothSides=True, LastEpoch=True)
                else:
                    my_loss = self.calc_sent_loss(sent, BothSides=True, LastEpoch=False)
                train_loss += my_loss.item()
                train_words += len(sent)
                # Taking the step after calculating loss for all the words in the sentence
                self.optimizer.zero_grad()
                my_loss.backward()
                self.optimizer.step()
                # if (sent_id + 1) % 50 == 0:
                #     print("--finished %r sentences" % (sent_id + 1))
        #     print("iter %r: train loss %.4f, train words %.4f" % (ITER, train_loss, train_words))
        #     break
            print("finished iter %r of document %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
            ITER, doc, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))