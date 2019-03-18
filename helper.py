import re
import pandas as pd
import numpy as np

## Function to split document passages into sentence
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "e.g." in text: text = text.replace("e.g.","e<prd>g<prd>")
    if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
    text = text.replace(",","")
    text = text.replace("-"," ")
    text = text.replace("'"," '")
    text = text.replace('"', "")
    text = text.replace('(','')
    text = text.replace(')','')
    text = text.replace('[',' ')
    text = text.replace(']',' ')
    text = text.replace('\\','')
    text = text.replace('{',' ')
    text = text.replace('}',' ') 
    text = text.replace('&',' and ')
    text = text.replace(':','')
    text = text.replace(".","<stop>")
    text = text.replace(";","<stop>")
    text = text.replace("?","<stop>")
    text = text.replace("!","<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

    

## Function to remove stop words from a text
def removeStopWords(questionText):
    return [w for w in questionText.split()  if not w in stop]

## Function which returns an array with 1 if a question word is present in the vocabulary and 0 otherwise
def binaryEncoding(wordList, qText):
    questionList = qText
    binaryEncoded =np.array([ 1 if i in questionList else 0 for i in wordList])
    return binaryEncoded

## Normalize query expansion weights
def normalise(weights):
    sumW = np.sum(weights)
    normW = weights/sumW
    return normW

## Function to find query expansion terms to a given query/text
def expandQuery(qtext):
    # qText='unemployment'
    filtered_text = removeStopWords(qText)
    q = binaryEncoding(wordList, filtered_text)
    # # candidate = UU'q
    tmp = np.matmul(U.T,q)
    candidate = np.matmul(U,tmp)
    index = np.argsort(-candidate)[:100]
    weights = -np.sort(-candidate)[:100]
    normW = normalise(weights)
    expansionTerms = wordList[index]
    pqPlus = dict(zip(expansionTerms,normW))


## Functions to write embeddings and word list to disk
    
def writeEmbedding(embfile):
    with open(embfile, 'w') as embeddings_file:
        for i in range(len(w2i)):
            ith_embedding = '\t'.join(map(str, word_embeddings[i]))
            embeddings_file.write(ith_embedding + '\n')

def writeWords(wordfile):             
    with open(wordfile, "w") as words_file:
        for i in range(len(w2i)):
            ith_word = i2w[i]
            words_file.write(ith_word + '\n')


