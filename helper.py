# General
import torch
import pandas as pd
# Translation
from multiprocessing import Pool
from tqdm import *
from googletrans import Translator
# Shuffling
from sklearn.utils import shuffle
#HTML Removal and Tokenization
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
import pickle
import os


"""Translation Region"""
#---------------------------------------------------------#
ts = Translator()

def translate_now(x):
    try:
        return [x[0], ts.translate(x[1], dest='en').text, x[2]]
    except:
        return [x[0], None, x[2]]


def imap_unordered_bar(func, args, n_processes: int = 48):
    p = Pool(n_processes, maxtasksperchild=100)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def translateDf(CSV_PATH,output_path,columns):
    df = pd.read_csv(CSV_PATH)
    df[columns] = imap_unordered_bar(translate_now, df[columns].values)
    df.to_csv(f''+output_path)
    print('done. Check {} for the translated csv file'.format(output_path))    


"""Extraction with shuffling Region"""
#----------------------------------------------------------#
def prepare_imdb_data(data_train, data_valid, data_test):
    """Prepare training, validation and test sets."""
    
    data_train = pd.read_csv(data_train)
    data_valid = pd.read_csv(data_valid)
    data_test  = pd.read_csv(data_test)
    
    #Extract the necessary columns
    data_train, labels_train = data_train['comment_text'], data_train['toxic']
    data_valid, labels_valid, lang_valid = data_valid['comment_text'], data_valid['toxic'], data_valid['lang']
    data_test, id_test, lang_test = data_test['content'], data_test['id'], data_test['lang']
    
    #Shuffle comments and corresponding labels within training and validation sets
    data_train, labels_train= shuffle(data_train, labels_train)
    data_valid, labels_valid, lang_valid = shuffle(data_valid, labels_valid, lang_valid)
    
    # Return a unified training [2], validation[3], test [3] sets in this order { [data, labels, language, id] }
    return data_train,labels_train, data_valid,labels_valid,lang_valid, data_test,lang_test,id_test


"""Remove HTML Tags and Tokenize"""
#----------------------------------------------------------#
def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words




def preprocess_data(train_X,train_y, valid_X,valid_lan,valid_y, test_id,test_X,test_lan,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, train_X))
        #words_valid = list(map(review_to_words, valid_X))
        #words_test = list(map(review_to_words, test_X))
        words_train = [review_to_words(comment) for comment in train_X]
        words_valid = [review_to_words(comment) for comment in valid_X]
        words_test = [review_to_words(comment) for comment in test_X]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_valid=words_valid, words_test=words_test,
                              train_y=train_y, valid_y=valid_y, valid_lan=valid_lan, test_lan=test_lan,
                             test_id=test_id)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_valid, words_test, train_y, valid_y, valid_lan, test_lan, test_id = (cache_data['words_train'],
                cache_data['words_valid'], cache_data['words_test'], cache_data['train_y'], cache_data['valid_y'],
                cache_data['valid_lan'], cache_data['test_lan'], cache_data['test_id'])
    
    return words_train,train_y, words_valid,valid_lan,valid_y, test_id,words_test,test_lan

















