
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
from gensim.summarization.textcleaner import tokenize_by_word
import copy
from langdetect import detect, lang_detect_exception
import concurrent.futures as cf 


Words={}
languages=['en']
models={}
def words(text): return re.findall(r'\w+', text.lower())

def P(language,word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - Words[language].get(word, 0)

def correction(word,language='en'): 
    "Most probable spelling correction for word."
    LangProb=partial(P,language)
    return max(candidates(word,language), key=LangProb)

def candidates(word,language): 
    "Generate possible spelling corrections for word."
    return (known([word],language) or known(edits1(word),language) or known(edits2(word),language) or [word])

def known(words,language): 
    #"The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in Words[language])

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
    sentences= gensim.summarization.textcleaner.clean_text_by_sentences(text)
    
    newSentences=[]
    for sentence in sentences:
        #print(sentence)
        
        newSentences.append(FixSentence(sentence.token))
    
    return "\n".join(newSentences)
    
def FixDocument(Document,dirname):
    
    try:
        t = time()
        with open("".join([dirname,"/",Document]),'r',encoding="utf",errors="surrogateescape") as doc:
            text=doc.read()
            sentences=FixText(text)
        with open("".join(["CORRECTEDCORPORA/"+Document]),"w") as output:
            output.writelines(sentences)
        print('Time to correct {}: {} mins'.format(Document,round((time() - t) / 60, 2)))

    except Exception as e:
        print("failed correction read for : " + Document)
        print(e)
   
def FixSentence(sentence):
    print(sentence)
    listedsentence=set(gensim.summarization.textcleaner.tokenize_by_word(sentence))
    newsentence=copy.deepcopy(sentence)
    language=detect(sentence)

    for word in listedsentence:
        '''to do
        ignore non alpha text
        change edit distance by word length? 
        '''
        #print(word)
        
        if word not in correctiondictionary[language]:
            correctiondictionary[language][word]=correction(word)
        if correctiondictionary[language][word]!=word:
            #print( word + " :=>  " + correctiondictionary[word])
            newsentence=newsentence.replace(word,correctiondictionary[language][word])
    if sentence!=newsentence:
        print( sentence + " :=> " + newsentence)
    #Sentence=" ".join(newsentence)    
    return newsentence

def Connect(URL,language):

  mongo=pymongo.MongoClient(URL)
  db= mongo["Connects"] #Connect to DB
  col=db["Access"]
  col.insert_one({"time":datetime.now()})
  db=mongo[language]
  return db

URL= os.environ.get("MONGO_CLUSTERURI")

def GetLanguages():
	return list(Connect(URL,"LANGUAGES")["CODES"].find({}))
#Copora=gensim.corpora.textcorpus.TextCorpus(input="./test/")
def make_model(language):
    print("Creating word model")
    t = time()
    Stream=Connect(URL,language)[Sentences]
    cursor = Stream.find({})
    word_vector_model = gensim.models.Word2Vec([sentence["text"].split() for sentence in cursor],size=200, window=8, min_count=10)
    print('Time to train model on everything: {} mins'.format(round((time() - t) / 60, 2)))
    return word_vector_model
correctiondictionary=dict()
def main():
    with Pool(os.cpu_count()) as p:
        p.starmap(buildLanguage,GetLanguages())   

def buildLanguage(language):
	print("Beginning with language : " + language)
        try:
            model = Word2Vec.load("".join([language,"word2vec.model"]))
        except:
            print("cant find models saved... lets make some")
            model=make_model(language)
            model.save("".join([language,"word2vec.model"]))
        finally:
            print("building word ranks")
            models[language]=model
            words = model.wv.vocab
            print(len(words))
            w_rank = {}
            for i,word in enumerate(words):
                w_rank[word] = i
            Words[language] = w_rank
    	correctiondictionary[language]={}
def fixAll():
    print("Begining document fixing")
    with Pool(os.cpu_count()) as p:
        p.starmap(FixDocument,zip(list(filter(lambda fname: fname.endswith('.txt'), os.listdir(dirname))),repeat(dirname)))
    

if __name__=="__main__":
    freeze_support()
    main()
