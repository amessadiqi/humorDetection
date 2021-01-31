import pandas as pd
from humor_features.utils import *


class HumorFeatures:
    def __init__(self, dataset = None):
        self.df = dataset
        self.df["text"] = self.df["text"].str.replace("!"," !",regex = False)
        self.df["text"] = self.df["text"].str.replace("?"," ?",regex = False)
        self.df["text"] = self.df["text"].str.replace("."," .",regex = False)
        self.df["text"] = self.df["text"].str.replace(","," ,",regex = False)
        self.df['textSeq'] = self.df["text"].apply(lambda ind:text_to_word_sequence(ind,filters='%\n\t01245679',lower=False, split=' '))
        self.df['textSeq'] = self.df['textSeq'].apply(lambda ind:[word for word in ind if not word in stopwords.words()])
        self.df['textSeq'] = self.df['textSeq'].apply(lambda ind:lemmatizeSeq(ind))
        #self.df['lenSeq'] = self.df["textSeq"].apply(lambda ind: len(ind))
    
    ## Structure
    def getNumWords(self):
        self.df['nbOfWords'] = self.df["textSeq"].apply(lambda ind:len(np.unique(ind)))
        return self
        
    def getMeanWordLength(self):
        self.df['meanWordLength'] = self.df["textSeq"].apply(lambda ind:getMeanWordLength(ind))
        return self
        
    def getTags(self):
        self.df['tags'] = self.df["textSeq"].apply(lambda ind: np.array(pos_tag(ind,tagset='universal')))
        self.df['tagsNameChange'] = self.df['tags'].apply(lambda tagSeq: np.apply_along_axis(tagRenamer,1,tagSeq) )
        self.df['tagged'] = self.df["textSeq"].apply(lambda ind: FreqDist(tag for (word,tag) in pos_tag(ind,tagset='universal')))
        return self
    
    def getGrammarRatios(self):
        self.df[['Adj ratio','Adv ratio','Noun ratio','Verb ratio']] = pd.DataFrame(self.df['tagged'].apply(getRatios).tolist(), index= self.df.index)
        return self
    
    def getPuncCount(self):
        self.df[['N. commas','N. fullStops','N. exclamation','N. qstMark']] = pd.DataFrame(self.df['textSeq'].apply(getPuncCount).tolist(),index = self.df.index)
        return self
    
    def getEmojiScore(self):
        self.df["EmojisScore"] = self.df['text'].apply(emojiScorer)
        return self
    
    def getLaughExprCount(self):
        self.df['laughingExpr'] = self.df['text'].apply(getLaughingExprCounter)
        return self
    
    def getStructure(self):
        return self.getNumWords().getMeanWordLength().getTags().getGrammarRatios().getPuncCount().getEmojiScore().getLaughExprCount()
    
    ## Frequency
    
    def getFreqMeanMin(self):
        self.df[["freqMean","freqMin"]] = pd.DataFrame(self.df['textSeq'].apply(getWordsFreq).tolist(),index = self.df.index)
        return self
    
    def getFreqGap(self):
        self.df["freqGap"] = self.df['freqMean'] - self.df['freqMin']
        return self
    
    def getFreq(self):
        return self.getFreqMeanMin().getFreqGap()

    ## Written - Spoken
    
    def getSpokenFreqs(self):
        self.df[["freqSpokenMean", "freqSpokenMin"]] = pd.DataFrame(self.df['textSeq'].apply(getSpokenSeq).tolist(),index = self.df.index)
        return self
    
    def getWrittenFreqs(self):
        self.df[["freqWrittenMean", "freqWrittenMin"]] = pd.DataFrame(self.df['textSeq'].apply(getWrittenSeq).tolist(),index = self.df.index)
        return self
    
    def getWrittenFreqGap(self):
        self.df["freqWrittenGap"] = self.df['freqWrittenMean'] - self.df['freqWrittenMin']
        return self

    
    def getSpokenFreqGap(self):
        self.df["freqSpokenGap"] = self.df['freqSpokenMean'] - self.df['freqSpokenMin']
        return self
    
    def getWrittenSpoken(self):
        return self.getSpokenFreqs().getWrittenFreqs().getWrittenFreqGap().getSpokenFreqGap()
    
    ## Synonyms
    def getSynoLowerGreater(self):
        self.df[["synoLower","synoGreater","wordLowestSyno","wordGreatestSyno"]] = pd.DataFrame(self.df["textSeq"].apply(getSynoFeaturesSeq).tolist(),index = self.df.index)
        return self
    
    def getSynoGaps(self):
        self.df["synoLowerGap"] = self.df["wordLowestSyno"] - self.df["synoLower"]
        self.df["synoGreaterGap"] = self.df["wordGreatestSyno"] - self.df["synoGreater"]
        return self

    def getSyno(self):
        return self.getSynoLowerGreater().getSynoGaps()
    
    ## Sentiment
    def getPSentiSum(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns or not 'posNegObjSentiSum' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiSum"] = self.df['posNegObjSenti'].apply(lambda ind:ind.sum(axis=0) )
        self.df['posSentiSum'] = self.df["posNegObjSentiSum"].apply(lambda ind:ind[0] )
        return self
    
    def getNSentiSum(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns or not 'posNegObjSentiSum' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiSum"] = self.df['posNegObjSenti'].apply(lambda ind:ind.sum(axis=0) )
        self.df['negSentiSum'] = self.df["posNegObjSentiSum"].apply(lambda ind:ind[1] )
        return self
    
    def getObjSentiSum(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns or not 'posNegObjSentiSum' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiSum"] = self.df['posNegObjSenti'].apply(lambda ind:ind.sum(axis=0) )
        self.df['objSentiSum'] = self.df["posNegObjSentiSum"].apply(lambda ind:ind[2] )
        return self
    
    def getPSentiMean(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns or not 'posNegObjSentiMean' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiMean"] = self.df['posNegObjSenti'].apply(lambda ind:ind.mean(axis=0) )
        self.df['posSentiMean'] = self.df["posNegObjSentiMean"].apply(lambda ind:ind[0] )
        return self
    
    def getNSentiMean(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns or not 'posNegObjSentiMean' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiMean"] = self.df['posNegObjSenti'].apply(lambda ind:ind.mean(axis=0) )
        self.df['negSentiMean'] = self.df["posNegObjSentiMean"].apply(lambda ind:ind[1] )
        return self
    
    def getObjSentiMean(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns or not 'posNegObjSentiMean' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiMean"] = self.df['posNegObjSenti'].apply(lambda ind:ind.mean(axis=0) )
        self.df['objSentiMean'] = self.df["posNegObjSentiMean"].apply(lambda ind:ind[2] )
        return self
    
    def getPNSentiGap(self):
        if not 'tags' in self.df.columns:
            self.getTags()
        if not 'posNegObjSenti' in self.df.columns and not 'posNegObjSentiSum' in self.df.columns:
            self.df['posNegObjSenti'] = self.df['tagsNameChange'].apply(lambda tagSeq: np.apply_along_axis(sentimentFeatures,1,tagSeq) )
            self.df["posNegObjSentiSum"] = self.df['posNegObjSenti'].apply(lambda ind:ind.sum(axis=0) )
        
        if not 'posSentiSum' in self.df.columns:
            getPSentiSum()
        
        if not 'negSentiSum' in self.df.columns:
            getNSentiSum()
        self.df['posNegGap'] = self.df["posSentiSum"] + self.df["negSentiSum"]
        return self
    
    def getSentiment(self):
        return self.getPSentiSum().getNSentiSum().getObjSentiSum().getPSentiMean().getNSentiMean().getObjSentiMean().getPNSentiGap()


    ## Synsets
    def getSynsets(self):
        self.df[["synsetMean","synsetMax","synsetGap"]] = pd.DataFrame(self.df["textSeq"].apply(getSynsetsMeanMaxGap).tolist(),index = self.df.index)
        return self


    def getAllFeatures(self):
        return self.getStructure().getFreq().getWrittenSpoken().getSyno().getSynsets().getSentiment().df
