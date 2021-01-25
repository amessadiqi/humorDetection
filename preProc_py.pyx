# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from glob import glob
import json
from nltk.corpus import wordnet as wn
from nltk import pos_tag,FreqDist
from nltk.corpus import sentiwordnet as swn
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
import tensorflow_text as text
from nltk.corpus import stopwords
from re import search,compile,findall
from emojis import encode , decode
from emojis.db import get_emoji_by_alias
import time
import Cython
from numba import jit,njit,vectorize
import pickle


# %%
# haha , happy , smile , pleased , laugh , joy , evil , devil , danger , dead

#path = "joke-dataset-master\\reddit_jokes.json"

selectedTags = [    ("haha",3) , ("happy",3) , ("smile",3) , ("pleased",2) , ("laugh",3) , ("joy",3) ,
                    ("evil",1) , ("devil",1) , ("danger",1) , ("dead",1),("goofy",3), ("wow",2), ("lol",3),
                    ("perfect",1), ("proud",1), ("praise",1) ]

wcRatiosDf = pd.read_csv("ANC-token-count.csv", sep=',',encoding='ISO-8859-1',header=0).set_index("word")
writtenDf = pd.read_csv("ANC-written-count.csv", sep=',', encoding='ISO-8859-1',header=0)
writtenDf = writtenDf.set_index("Word")
writtenDf.drop(columns=["POS","Lemma"],inplace=True)
writtenDf = writtenDf.groupby("Word").sum()
spokenDf = pd.read_csv("ANC-written-count.csv", sep=',', encoding='ISO-8859-1',header=0)
spokenDf = spokenDf.set_index("Word")
spokenDf.drop(columns=["POS","Lemma"],inplace=True)
spokenDf = spokenDf.groupby("Word").sum()
totalWrittenWord = writtenDf["Count"].sum()
totalSpokenWord = spokenDf["Count"].sum()
lenVectorized = np.vectorize(len)



def cleaner(df):
    df.drop(columns=["text","textSeq"],inplace=True)
    return df
# %%
cpdef int getMeanWordLength(list array):
    return int(lenVectorized(array).mean())

cpdef (int,int,int,int) getRatios(freqDict):
    noun,adj,verb,adv = 0,0,0,0
    all = 0
    cdef str tag
    cdef int occ
    for (tag,occ) in freqDict.items():
        if not tag == "'.'":
            all += occ
    for (tag,occ) in freqDict.items():
        if tag == 'NOUN':
            noun = occ / all
        if tag == 'ADJ':
            adj = occ / all
        if tag == 'VERB':
            verb = occ / all
        if tag == 'ADV':
            adv = occ / all
    return adj,adv,noun,verb

cpdef (int,int,int,int) getPuncCount(list sequence):
    return np.char.count(sequence,sub=",").sum(),np.char.count(sequence,sub=".").sum(),np.char.count(sequence,sub="!").sum(),np.char.count(sequence,sub="?").sum()

cpdef int emojiScorer(sentence):
    cdef int score = 0
    for tagArray in [get_emoji_by_alias(alias[1:-1])[2] for alias in compile(':[A-z]*:').findall(sentence)]:
        for selectedTag in selectedTags:
            if selectedTag[0] in tagArray:
                score += selectedTag[1]
                break
        continue
    return score

def getLaughingExprCounter(sentence):
    return len(findall( r"\b(?:a*(?:ha*)+h?|h*ha+h[ha]*|(?:l+o+)+l+|o?l+o+l+[ol]*)\b", sentence))

cpdef double getwcRatio( word):
    if word in wcRatiosDf.index:
        return float(wcRatiosDf.at[word,"frequency"])
    else:
        return np.NaN

getwcRatioVectorized = np.vectorize(getwcRatio)
cpdef (double,double) getWordsFreq(list sequence):
    wcRatios = getwcRatioVectorized(sequence)
    wcRatios = wcRatios[np.logical_not(np.isnan(wcRatios))] 
    return wcRatios.mean(),wcRatios.min()

cpdef double getWritten( word):
    if word in writtenDf.index:
        return float(writtenDf.at[word,"Count"]/totalWrittenWord)
    else:
        return np.NaN  

cpdef double getSpoken( word):
    if word in spokenDf.index:
        return float(spokenDf.at[word,"Count"]/totalSpokenWord)
    else:
        return np.NaN

getWrittenVectorized = np.vectorize(getWritten)
getSpokenVectorized = np.vectorize(getSpoken)

cpdef (double,double) getWrittenSeq(list sequence):
    written = getWrittenVectorized(sequence)
    written = written[np.logical_not(np.isnan(written))] 
    return written.mean(),written.min()


cpdef (double,double) getSpokenSeq(sequence):
    spoken = getSpokenVectorized(sequence)
    spoken = spoken[np.logical_not(np.isnan(spoken))] 
    return spoken.mean(),spoken.min()


"""Get synonyms' features"""
## Split by "." vectorized
def splitSyno(word):
    return word.split(".")[0]

splitVectorized = np.vectorize(splitSyno)

def splitSynos(synoSequence):
    return splitVectorized(synoSequence)

## Filter synonyms vectorized
def filter(word):
    if word in wcRatiosDf.index:
        return word
filterVectorized = np.vectorize(filter)

def filterSynosSequence(sequence):
    filteredSeq = filterVectorized(sequence) 
    return filteredSeq[filteredSeq != 'None'] 

## Get frequency vectorized
def getFreq(word):
    try:
        return wcRatiosDf.at[word,"frequency"]
    except:
        return -1
getFreqVectorized = np.vectorize(getFreq)

def getFreqSeq(sequence):
    frequencies = getFreqVectorized(sequence)
    return frequencies[frequencies != -1]

## Get synsets vectorized
def getSynsets(word):
    return wn.synsets(word)
    
getSynsetsVectorized = np.vectorize(getSynsets)

def getSynsetsNumber(word):
    return len(wn.synsets(word))
    
getSynsetsNumberVectorized = np.vectorize(getSynsetsNumber)

def getSynsetsMeanMaxGap(sequence):
    try:
        synNumbers = getSynsetsNumberVectorized(sequence)
        synsetMean = synNumbers.mean()
        synsetMax = synNumbers.max()
        synsetGap = synsetMax - synsetMean
        return synsetMean,synsetMax,synsetGap
    except:
        return 0,0,0
#print(getSynsetsMeanMaxGap(["dog","cat","father"]))



def getSynsetsSeq(sequence):
    return getSynsetsVectorized(sequence)

## get sysnonyms from lemmas vectorized
def getSynonymFromLemma(lemma):
    return lemma.name()

getSynonymFromLemmaVectorized = np.vectorize(getSynonymFromLemma)

def getSynonymFromSeqLemmas(lemmasSequence):
    return getSynonymFromLemmaVectorized(lemmasSequence)

## get synonyms from word vectorized
def getSynosFromWord(word):
    synos = filterSynosSequence(splitSynos(getSynonymFromSeqLemmas(getSynsets(word))))
    return synos,getFreqSeq(synos)

## get features vectorized
def getSynoFeatures(word):
    try:
        synonyms,frequencies = getSynosFromWord(word)
        synoWithFreq = np.array((synonyms,frequencies)).T
        synoLowerWithFreq = sorted(synoWithFreq, key=lambda x: x[1],reverse = True)
        synoGreaterWithFreq = sorted(synoWithFreq, key=lambda x: x[1],reverse = False)
        return len(synoLowerWithFreq[([x for x, y in enumerate(synoLowerWithFreq) if y[0] == word])[0]:]), len(synoGreaterWithFreq[([x for x, y in enumerate(synoGreaterWithFreq) if y[0] == word])[0]:])
    except:
        return 0,0

getSynoFeaturesVectorized = np.vectorize(getSynoFeatures)

def  getSynoFeaturesSeq(sequence):
    try:
        array = np.array(getSynoFeaturesVectorized(sequence))
        return np.array(array[0]).mean(),np.array(array[1]).mean(),np.array(array[0]).max(),np.array(array[1]).max()
    except:
        return 0,0,0,0

# LEMMATIZER
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatizeWord(word):
    try:
        return lemmatizer.lemmatize(word)
    except:
        return word

lemmatizeVectorized = np.vectorize(lemmatizeWord)
def lemmatizeSeq(sequence):
    try:
        return lemmatizeVectorized(sequence).tolist()
    except:
        return sequence


# tag renamer
def tagRenamer(wordTagTuple):
    if wordTagTuple[1] == "NOUN":
        return wordTagTuple[0],'n'
    
    elif wordTagTuple[1] == "VERB":
        return wordTagTuple[0],'v'
    
    elif wordTagTuple[1] == "ADJ":
        return wordTagTuple[0],'a'

    elif wordTagTuple[1] == "ADV":
        return wordTagTuple[0],'r'

    else:
        return wordTagTuple
# sentiment
def sentimentFeatures(wordTagTuple):
    try:
        mainSyn = next(swn.senti_synsets(wordTagTuple[0],wordTagTuple[1]))
        return mainSyn.pos_score(),mainSyn.neg_score(),mainSyn.obj_score()
    except:
        return 0,0,0
# %%
class ReddittextsDataset:
    def __init__(self,path,sentence = None):
        #JSON to dataframe
        #self.df = pd.read_json(path).iloc[:-1,:]
        if sentence == None:
            self.df = pd.read_csv(path, sep=',',encoding='ISO-8859-1',header=0)
        else:
            self.df = pd.DataFrame(data = {'text':[sentence],'humor':[np.NaN]})
            print(self.df)
        #self.df["text"] = self.df["title"] + " " + self.df["body"]
        self.df["text"] = self.df["text"].str.replace("!"," !",regex = False)
        self.df["text"] = self.df["text"].str.replace("?"," ?",regex = False)
        self.df["text"] = self.df["text"].str.replace("."," .",regex = False)
        self.df["text"] = self.df["text"].str.replace(","," ,",regex = False)
        #self.df.drop(columns = ["title","body","id"], inplace= True)
        self.df['textSeq'] = self.df["text"].apply(lambda ind:text_to_word_sequence(ind,filters='%\n\t01245679',lower=False, split=' '))
        self.df['textSeq'] = self.df['textSeq'].apply(lambda ind:[word for word in ind if not word in stopwords.words()])
        self.df['textSeq'] = self.df['textSeq'].apply(lambda ind:lemmatizeSeq(ind))
        self.df['lenSeq'] = self.df["textSeq"].apply(lambda ind: len(ind))
        #meanLen = int(self.df['lenSeq'].mean(axis = 0)) + 1
        #self.df.drop(columns= ["score"],inplace=True)

    ## Structure
    def getStructure(self):
        self.df['nbOfWords'] = self.df["textSeq"].apply(lambda ind:len(np.unique(ind)))
        self.df['meanWordLength'] = self.df["textSeq"].apply(lambda ind:getMeanWordLength(ind))
        self.df['tags'] = self.df["textSeq"].apply(lambda ind: np.array(pos_tag(ind,tagset='universal')))
        self.df['tagsNameChange'] = self.df['tags'].apply(lambda tagSeq: np.apply_along_axis(tagRenamer,1,tagSeq) )
        self.df['tagged'] = self.df["textSeq"].apply(lambda ind: FreqDist(tag for (word,tag) in pos_tag(ind,tagset='universal')))
        self.df[['Adj ratio','Adv ratio','Noun ratio','Verb ratio']] = pd.DataFrame(self.df['tagged'].apply(getRatios).tolist(), index= self.df.index)
        self.df[['N. commas','N. fullStops','N. exclamation','N. qstMark']] = pd.DataFrame(self.df['textSeq'].apply(getPuncCount).tolist(),index = self.df.index)
        self.df["EmojisScore"] = self.df['text'].apply(emojiScorer)
        self.df['laughingExpr'] = self.df['text'].apply(getLaughingExprCounter)
        return self


    ## Frequency
    def frequency(self):
        self.df[["freqMean","freqMin"]] = pd.DataFrame(self.df['textSeq'].apply(getWordsFreq).tolist(),index = self.df.index)
        self.df["freqGap"] = self.df['freqMean'] - self.df['freqMin']
        return self

    ## Written - Spoken
    def getWrittenSpoken(self):
        self.df[["freqSpokenMean", "freqSpokenMin"]] = pd.DataFrame(self.df['textSeq'].apply(getSpokenSeq).tolist(),index = self.df.index)
        self.df[["freqWrittenMean", "freqWrittenMin"]] = pd.DataFrame(self.df['textSeq'].apply(getWrittenSeq).tolist(),index = self.df.index)
        self.df["freqWrittenGap"] = self.df['freqWrittenMean'] - self.df['freqWrittenMin']
        self.df["freqSpokenGap"] = self.df['freqSpokenMean'] - self.df['freqSpokenMin']
        return self

    ## Synonyms
    def getSynonyms(self):
        self.df[["synoLower","synoGreater","wordLowestSyno","wordGreatestSyno"]] = pd.DataFrame(self.df["textSeq"].apply(getSynoFeaturesSeq).tolist(),index = self.df.index)
        self.df["synoLowerGap"] = self.df["wordLowestSyno"] - self.df["synoLower"]
        self.df["synoGreaterGap"] = self.df["wordGreatestSyno"] - self.df["synoGreater"]
        return self

    ## Synsets
    def getSynsetsFeatures(self):
        self.df[["synsetMean","synsetMax","synsetGap"]] = pd.DataFrame(self.df["textSeq"].apply(getSynsetsMeanMaxGap).tolist(),index = self.df.index)
        return self

    ## Sentiment
    def getSentiment(self):
        self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
        self.df["posNegObjSentiSum"] = self.df['posNegObjSenti'].apply(lambda ind:ind.sum(axis=0) )
        self.df['posSentiSum'] = self.df["posNegObjSentiSum"].apply(lambda ind:ind[0] )
        self.df['negSentiSum'] = self.df["posNegObjSentiSum"].apply(lambda ind:ind[1] )
        self.df['objSentiSum'] = self.df["posNegObjSentiSum"].apply(lambda ind:ind[2] )
        self.df["posNegObjSentiMean"] = self.df['posNegObjSenti'].apply(lambda ind:ind.mean(axis=0) )
        self.df['posSentiMean'] = self.df["posNegObjSentiMean"].apply(lambda ind:ind[0] )
        self.df['negSentiMean'] = self.df["posNegObjSentiMean"].apply(lambda ind:ind[1] )
        self.df['objSentiMean'] = self.df["posNegObjSentiMean"].apply(lambda ind:ind[2] )
        self.df['posNegGap'] = self.df["posSentiSum"] + self.df["negSentiSum"]
        self.df.drop(columns=["posNegObjSenti","posNegObjSentiSum","tags","tagged",'tagsNameChange'],inplace=True)
        #print(self.df[["posSentiSum","negSentiSum","objSentiSum","posSentiMean","negSentiMean","objSentiMean","posNegGap"]])
        return self

    ## Predictions
    def getPredictions(self):
        print(cleaner(self.df))
        loaded_model = pickle.load(open('weights\\XGBoost.sav', 'rb'))
        predictions = loaded_model.predict(self.df.iloc[:,1:].to_numpy())
        print(predictions)
        #s.fillna(fill_value)
        #self.df["humor"] = 
        #self.df["humor"].fillna(lambda )
        #return self.df["humor"].transform(loaded_model.predict(cleaner(self.df).iloc[:,1:].to_numpy()))

"""
strating_time = time.time()
ReddittextsDataset().getStructure().frequency().getWrittenSpoken().getSynonyms().df.to_csv(path+"")
print(time.time() - strating_time)
"""
ReddittextsDataset(path="C:\\Users\\Saad\\Desktop\\Humor_Detection\\dataset_201.csv").getStructure().getSentiment()
# %%


