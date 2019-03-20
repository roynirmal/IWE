import re
import pandas as pd
import numpy as np

## Function to split document passages into sentence

def split_into_sentences(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"

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
    text = text.replace("'"," ")
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




def prepare_question_text(question_text):
    question_text = question_text.replace('?','') #remove ?
    question_text = question_text.replace("'",'')
    question_text = question_text.replace('"','')
    question_text = question_text.replace('-',' ')
    question_text = question_text.replace('(','')
    question_text = question_text.replace(')','')
    question_text = question_text.replace(',','')
    question_text = question_text.replace('.','')
    question_text = question_text.replace('&',' and ')
    question_text = question_text.replace(':','')
    question_text = question_text.replace('>','')#error in dataset

    # if "[" in question_text:
    #     #print q_id,question_text
    #     if q_id == "3340": #remove contents
    #         question_text = re.sub(r'\[[^\(]*?\]', r'', question_text)
    #     else: #keep contents
    #         question_text = re.sub(r'\[(?:[^\]|]*\|)?([^\]|]*)\]', r'\1', question_text)
    # if "/" in question_text:
    #     #print q_id,question_text
    #     if q_id == "104" or q_id == "857":
    #         question_text = question_text.replace('/','')
    #     else:
    #         question_text = question_text.replace('/',' or ')

    return question_text.lower()

    

## Function to remove stop words from a text
def removeStopWords(questionText, stop):
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
def expandQuery(question, U, terms, wordList, stop):
    # qText='unemployment'
    qText = prepare_question_text(question)
    filtered_text = removeStopWords(qText, stop)
    q = binaryEncoding(wordList, filtered_text)

    tmp = np.matmul(U.T,q)
    candidate = np.matmul(U,tmp)
    index = np.argsort(-candidate)[:terms]
    weights = -np.sort(-candidate)[:terms]
    normW = normalise(weights)
    expansionTerms = wordList[index]

    pqPlus = dict(zip(expansionTerms,normW))
    return expansionTerms, pqPlus


## Functions to write embeddings and word list to disk
    
def writeEmbedding(embfile, w2i, word_embeddings):
    with open(embfile, 'w') as embeddings_file:
        for i in range(len(w2i)):
            ith_embedding = '\t'.join(map(str, word_embeddings[i]))
            embeddings_file.write(ith_embedding + '\n')

def writeWords(wordfile, w2i, i2w):             
    with open(wordfile, "w") as words_file:
        for i in range(len(w2i)):
            ith_word = i2w[i]
            words_file.write(ith_word + '\n')


