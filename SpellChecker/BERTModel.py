
# or run truncated Singular Value Decomposition (SVD) on the streamed corpus
from gensim.models import Word2Vec
import gensim
import os
from gensim import *
from time import time
from datetime import datetime
from gensim import utils
from unidecode import unidecode
from itertools import repeat
from multiprocessing import Pool, freeze_support
from multiprocessing.pool import ThreadPool
from gensim.summarization.textcleaner import tokenize_by_word
from gensim.summarization import summarize
import copy
#from langdetect import detect, lang_detect_exception
import concurrent.futures as cf 
import pymongo
from functools import partial
import json
import langid
correctiondictionary=dict()
WordRanks={}
languages=['en']
models={}
URL= os.environ.get("MONGO_CLUSTERURI")
start=time()

def Connect(URL,collection="sentences"):

  mongo=pymongo.MongoClient(URL)
  db= mongo["Connects"] #Connect to DB
  col=db["Access"]
  col.insert_one({"time":datetime.now()})
  db=mongo[collection]
  return db

def testLanguage(text):
    try:
        language=langid.classify(text)
        #language=detect(text)#.lang#
    except Exception as e:
        language='UNK'
    return language

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

def words(text): return re.findall(r'\w+', text.lower())

def P(language,word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # get model
    # get sentence
    # how well does the word fit in the sentence. 
    # returns 0 if the word isn't in the dictionary
    return - WordRanks[language].get(word, 0)

def correction(word,language='en'): 
    global correctiondictionary
    "Most probable spelling correction for word."
    if language in correctiondictionary:
        if word not in correctiondictionary[language]:
            LangProb=partial(P,language)
            correctiondictionary[language][word]= max(candidates(word,language), key=LangProb)
        return word,correctiondictionary[language][word]
    else:
        return word, word

def candidates(word,language): 
    "Generate possible spelling corrections for word."
    return (known([word],language) or known(edits1(word),language) or known(edits2(word),language) or [word])

def known(words,language): 
    #"The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WordRanks[language])

def edits1(word):
    #"All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    #"All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def editsn(word,range):
    if range==1:
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    else:
        return (edit for word in editsn(word,range-1) for edit in editsn(word,1))


def FixText(text):
    sentences= gensim.summarization.textcleaner.split_sentences(text)
    #in line replace may be better. 
    
    with Pool() as p:
        newSentences=p.map(FixSentence,sentences)
 
    return "\n".join(newSentences)
    
def FixDocument(Document,dirname):
    outdir="correctedtexts"
    outputlocation=os.path.join(outdir,Document)
    #try:
    if not os.path.exists(outputlocation):
        print("Beginning : "+Document)

        t = time()
        with open(os.path.join(dirname,Document),'r',encoding="utf",errors="ignore") as doc:
            text=doc.read()
            sentences=FixText(text)
        with open(outputlocation,"w") as output:
            output.writelines(sentences)
        print('Time to correct {}: {} mins'.format(Document,round((time() - t) / 60, 2)))
    else:
        print("skipping {} as already found in {}".format(Document,outdir))
    '''
    except Exception as e:
        print("failed correction read for : " + Document)
        print(e)
    '''

def FixSentence(sentence):

    #print(sentence)
    listedsentence=list(set(gensim.summarization.textcleaner.tokenize_by_word(sentence)))
    #print(listedsentence)
    newsentence=copy.deepcopy(sentence)
    language=langid.classify(sentence)
    #print(b[0])
    #language=detect(sentence)
    #print(lookups)
    for word,fix in filter(lambda x: x[0]!=x[1],map(correction,listedsentence,repeat(language))):
        '''to do
        ignore non alpha text
        change edit distance by word length? 
        '''
        newsentence=newsentence.replace(word,fix)
    #Sentence=" ".join(newsentence)    
    return newsentence

def BuildSummary(Document,dirname):
    outdir="summaryoftexts"
    outputlocation=os.path.join(outdir,Document)
    #try:
    if not os.path.exists(outputlocation):
        print("Beginning : "+Document)
        t = time()
        with open(os.path.join(dirname,Document),'r',encoding="utf",errors="ignore") as doc:
            text=doc.read()
            summmary=summarize(text, ratio=0.01)
        with open(outputlocation,"w") as output:
            output.write(summmary)
        print('Time to correct {}: {} mins'.format(Document,round((time() - t) / 60, 2)))
    else:
        print("skipping {} as already found in {}".format(Document,outdir))


def GetLanguages():
    print("fetching languages")
    found=list(Connect(URL,"sentences")["Sentences"].find({}).distinct("lang"))
    print(found)
    return found
#Copora=gensim.corpora.textcorpus.TextCorpus(input="./test/")
def make_model(language):
    print("Creating word model")
    t = time()
    Stream=Connect(URL,"sentences")["Sentences"]
    cursor = Stream.find({"lang":language})
    word_vector_model = gensim.models.Word2Vec([sentence["text"].split() for sentence in cursor],size=200, window=8, min_count=10)
    print('Time to train model on everything: {} mins'.format(round((time() - t) / 60, 2)))
    return word_vector_model


def buildLanguage(language):
    global correctiondictionary
    print("Beginning with language : " + language)
    try:
        model = Word2Vec.load(os.path.join("models","".join([language,"word2vec.model"])))
    except:
        print("cant find models saved... lets make some")
        model=make_model(language)
        model.save(os.path.join("models","".join([language,"word2vec.model"])))
    finally:
        print("building word ranks")
        models[language]=model
        words = model.wv.vocab
        print(len(words))
        w_rank = {}
        for i,word in enumerate(words):
            w_rank[word] = i
        return w_rank
def fixAll(dirname):
    print("Begining document fixing")
    with Pool(os.cpu_count()) as p:
        p.starmap(FixDocument,zip(list(filter(lambda fname: fname.endswith('.txt'), os.listdir(dirname))),repeat(dirname)))
    
def main():
    global  WordRanks, correctiondictionary
    dir_name="textfiles"
    

    languages=list(GetLanguages())
    if languages==list():
        #if there arent any unique languages, we may need to repopulate database.
        print("languages not found so rebuilding vocab")
        filenames=[os.path.join(dir_name,filename) for filename in filter(lambda fname: fname.endswith('.txt'), os.listdir(dir_name))]
        with Pool(cpu_count()) as p:
            done=p.map(openfile, filenames)
            print("read from %s documents",str(sum(done)))
        languages=list(GetLanguages())
    lookupdir="dictionary"
    WordRankfile="WordRank.json"
    WordRankfilepath=os.path.join(lookupdir,WordRankfile)
    if not os.path.exists(WordRankfilepath):
        with Pool(os.cpu_count()) as p:
            WordRanks=dict(zip(languages,p.map(buildLanguage,list(languages))))   
        with open(WordRankfilepath,"w") as output:
            json.dump(WordRanks,output)
    else:
        with open(WordRankfilepath,"r") as input:
            WordRanks=json.load(input)
    
    print("models are ready and dictionaries locked and loaded.")
    
    correctionsfile="correctionsfile.json"
    correctionsfilepath=os.path.join(lookupdir,correctionsfile)
    if os.path.exists(correctionsfilepath):
        print("loading previous corrections...")
        with open(correctionsfilepath,"r") as input:
            correctiondictionary=json.load(input)
    with ThreadPool() as p:
        p.starmap(FixDocument,zip(list(filter(lambda fname: fname.endswith('.txt'), os.listdir(dir_name))),repeat(dir_name)))
    #for document in list(filter(lambda fname: fname.endswith('.txt'), os.listdir(dir_name))):
    #   FixDocument(document,dir_name)
    with open(correctionsfilepath,"w") as output:
        print("saving corrections")
        json.dump(correctiondictionary,output)
    with ThreadPool() as p:
        p.starmap(BuildSummary,zip(list(filter(lambda fname: fname.endswith('.txt'), os.listdir(correctedtexts))),repeat("correctedtexts")))

if __name__=="__main__":
    freeze_support()
    main()
