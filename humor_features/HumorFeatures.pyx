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
from emojis.db import get_emoji_by_alias
import Cython
import numpy as np


class HumorFeatures:
    def __init__(self):
        self.selectedTags = [ ("haha",3) , ("happy",3) , ("smile",3) , ("pleased",2) , ("laugh",3) , ("joy",3) ,
                            ("evil",1) , ("devil",1) , ("danger",1) , ("dead",1),("goofy",3), ("wow",2), ("lol",3),
                            ("perfect",1), ("proud",1), ("praise",1) ]

        self.wcRatiosDf = pd.read_csv("ANC-token-count.csv", sep=',',encoding='ISO-8859-1',header=0).set_index("word")
        self.writtenDf = pd.read_csv("ANC-written-count.csv", sep=',', encoding='ISO-8859-1',header=0)
        self.writtenDf = writtenDf.set_index("Word")
        self.writtenDf.drop(columns=["POS","Lemma"],inplace=True)
        self.writtenDf = self.writtenDf.groupby("Word").sum()
        self.spokenDf = pd.read_csv("ANC-written-count.csv", sep=',', encoding='ISO-8859-1',header=0)
        self.spokenDf = self.spokenDf.set_index("Word")
        self.spokenDf.drop(columns=["POS","Lemma"],inplace=True)
        self.spokenDf = self.spokenDf.groupby("Word").sum()
        self.totalWrittenWord = self.writtenDf["Count"].sum()
        self.totalSpokenWord = self.spokenDf["Count"].sum()
        self.lenVectorized = np.vectorize(len)


    def cleaner(df):
        df.drop(columns=["text","textSeq"],inplace=True)
        return df


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


    def getSynsetsSeq(sequence):
        return getSynsetsVectorized(sequence)



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


    def getSynoFeaturesSeq(sequence):
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


if __name__=='__main__':
    features = HumorFeatures()

    f1 = features.lemmatize(sentence, seq=False)
    f2 = features.lemmatizeWord(sentence)

    f = features.getFeatures(features=['f1', 'f2', 'f3'])
