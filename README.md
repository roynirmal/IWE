# IWE

This is the Py-torch version of the embedding model **IWE** proposed by Shaosheng Cao and Wei Lu. 
- [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14724/14187)
- [Original Github Repo](https://github.com/ShelsonCao/IWE)

In this model, words are represented by three sets of subword features - Letter Trigram features, Root Afflixes and Inflectional Affixes.

For our work, we use the [WikiPassageQA](https://arxiv.org/pdf/1805.03797.pdf) dataset to train the embeddings. We use the local embeddings as proposed by [Diaz et al.](http://www.aclweb.org/anthology/P16-1035) where we train the embeddings separately on each document.
