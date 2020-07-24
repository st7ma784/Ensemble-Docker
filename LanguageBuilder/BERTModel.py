
# or run truncated Singular Value Decomposition (SVD) on the streamed corpus
from gensim.models import Word2Vec
import gensim
import os
from gensim import *
from time import time
from gensim import utils
from unidecode import unidecode
from itertools import repeat
from multiprocessing import Pool, freeze_support
from gensim.summarization.textcleaner import tokenize_by_word
import copy
from langdetect import detect, lang_detect_exception
import concurrent.futures as cf 
class WordTrainer(object):
    def __init__(self, dir_name=".",language='en'):
        self.dir_name = dir_name
        self.filenames= list(filter(lambda fname: fname.endswith('.txt'), os.listdir(dir_name)))
        print("Found Files: \n " + "\n".join(self.filenames))
        self.files={}
        self.language=language
        self.knownlanguages=[]
    def setLanguage(self,language):
        self.language=language
    def testLanguage(self,text):
        try:
            language=detect(text)#.lang#
        except lang_detect_exception.LangDetectException as e:
            language='en'
        if language==self.language:
             return True
        else:
            if language not in self.knownlanguages:
                self.knownlanguages.append(language)
            return False
    def openfile(self, file_name):
        if self.files.get(file_name,None) is None:

            with open(os.path.join(self.dir_name, file_name),'r',encoding="utf",errors="ignore") as text:
            #print(text.read())
                sentences= gensim.summarization.textcleaner.split_sentences(text.read())
                self.files[file_name]=sentences
        sentences=filter(lambda sentence:self.testLanguage(sentence) and len(sentence.split())>4,self.files[file_name])
        return [sentence.split() for sentence in sentences]
    def __iter__(self):
        with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as tp:
            fl = [tp.submit(self.openfile, fn) for fn in self.filenames]
        for sentencelist in cf.as_completed(fl):
            for sentence in sentencelist.result():
                yield sentence
texts=None
def make_model(language):
    print("Reading Vocab")
    t = time()
    print("Reading sentences from files...")
    if texts is None:
    	texts=WordTrainer()
    	WriteLanguages(text.knownlanguages)
    
    texts.setLanguage(language)
    #Copora=gensim.corpora.textcorpus.TextCorpus(input="./test/")

    print('Time to train model on everything: {} mins'.format(round((time() - t) / 60, 2)))
    return text.knownlanguages, texts

from collections import Counter
from functools import partial
import pymongo
def Connect(URL,language):

  mongo=pymongo.MongoClient(URL)
  db= mongo["Connects"] #Connect to DB
  col=db["Access"]
  col.insert_one({"time":datetime.now()})
  db=mongo[language]
  return db


URL= os.environ.get("MONGO_CLUSTERURI")
Words={}
languages=['en']
def WriteLanguages(languages):
	Connect(URL,"LANGUAGES")["CODES"].insertMany(languages)

def main():
    for language in languages:
        print("Beginning with language : " + language)
        languages, samples=make_model(language)
        print("Pushing to database:")
        LanguageDB=Connect(URL,language)
        Sentences=LanguageDB["Sentences"]
        Sentences.insertMany([sentence for sentence in samples)
if __name__=="__main__":
    freeze_support()
    main()
