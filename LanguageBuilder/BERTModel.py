
# or run truncated Singular Value Decomposition (SVD) on the streamed corpus
from gensim.models import Word2Vec
import gensim
import os
from gensim import *
from datetime import datetime
from time import time
from gensim import utils
from unidecode import unidecode
from itertools import repeat
from multiprocessing import Pool, freeze_support, cpu_count
from gensim.summarization.textcleaner import tokenize_by_word
import copy

from collections import Counter
from functools import partial
import pymongo
from langdetect import detect, lang_detect_exception
import concurrent.futures as cf 
from multiprocessing import Pool



def Connect(URL,collection="sentences"):

  mongo=pymongo.MongoClient(URL)
  db= mongo["Connects"] #Connect to DB
  col=db["Access"]
  col.insert_one({"time":datetime.now()})
  db=mongo[collection]
  return db


URL= os.environ.get("MONGO_CLUSTERURI")
knownlanguages=set(['en'])
def WriteLanguages():
    global knownlanguages
    Connect(URL,"LANGUAGES")["CODES"].insert_many(dict({"code":code}) for code in list(knownlanguages))

def testLanguage(text):
    global knownlanguages
    try:
        language=detect(text)#.lang#
    except lang_detect_exception.LangDetectException as e:
        language='UNK'
    if language not in knownlanguages:
        knownlanguages.add(language)
    return language
start=time()
def openfile(file_name):
    global start
    t=time()
    
    DB=Connect(URL)["Sentences"]
    with open(file_name,'r',encoding="utf",errors="ignore") as text:
        print("Opened : " + file_name)
        sentences= gensim.summarization.textcleaner.split_sentences(text.read())
        sentences=filter(lambda sentence:len(sentence.split())>4,sentences)
        DB.insert_many([{"text":sentence, "lang":testLanguage(sentence)} for sentence in sentences])
        #return [{"text":sentence, "lang":self.testLanguage(sentence)} for sentence in sentences]
        print('Time to upload{}: {} mins / {} mins'.format(file_name,round((time() - t) / 60, 2),round((time() - start) / 60, 2) ))
        return 1
def main():
    dir_name="textfiles"
    filenames=[os.path.join(dir_name,filename) for filename in filter(lambda fname: fname.endswith('.txt'), os.listdir(dir_name))]
    with Pool(cpu_count()) as p:
        p.map(openfile, filenames)
    WriteLanguages()
if __name__=="__main__":
    freeze_support()
    main()
    print("finished putting sentences in DB...")