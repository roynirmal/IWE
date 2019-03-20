# IWE

This is the Py-torch version of the embedding model **IWE** proposed by Shaosheng Cao and Wei Lu. 
- [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14724/14187)
- [Original Github Repo](https://github.com/ShelsonCao/IWE)

In this model, words are represented by three sets of subword features - Letter Trigram features, Root Afflixes and Inflectional Affixes. We obtain the last two set of features from the author's original repo and perform the letter trigram hasing ourselves.

For our work, we use the [WikiPassageQA](https://arxiv.org/pdf/1805.03797.pdf) dataset to train the embeddings. We use the local embeddings as proposed by [Diaz et al.](http://www.aclweb.org/anthology/P16-1035) where we train the embeddings separately on each document.

To get your own IWE embeddings on the WikiPassageQA data, run the follwoing code in your favorite terminal:
```shell
git clone https://github.com/roynirmal/IWE.git
python main.py 
```
The word embeddings and the word list will be in the `./Embeddings/` folder. If you want to use your own data put it inside the `./data/` folder.
