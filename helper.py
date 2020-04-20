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
import emoji

import numpy as np

"""Translation Region"""
#---------------------------------------------------------------------------------#
ts = Translator()

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)

"""Use the commented lines for translating test set while the lines below them is for translating validation set"""

def translate_now(x):
    #if x[3] == 'failure':
    if x[4] == 'failure':
        try:
            #return [x[0], ts.translate(remove_emoji(x[1]), dest='en').text, x[2], 'success']
            return [x[0], ts.translate(remove_emoji(x[1]), dest='en').text, x[2], x[3], 'success']
        except:
            try:
                #return [x[0], ts.translate(deEmojify(x[1]), dest='en').text, x[2], 'success']
                return [x[0], ts.translate(deEmojify(x[1]), dest='en').text, x[2], x[3], 'success']
            except:
                #return [x[0], x[1], x[2], x[3]]
                return [x[0], x[1], x[2], x[3], x[4]]
    #return [x[0], x[1], x[2], x[3]]
    return [x[0], x[1], x[2], x[3], x[4]]



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
    df.to_csv(f''+output_path,index=False)
    print('done. Check {} for the translated csv file'.format(output_path))    


"""Extraction with shuffling Region"""
#-----------------------------------------------------------------------------------------#
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
    #Swords = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words
#####################################################################################################
def word_list_train(x):
    try:
        return np.array([x[0], ['en'.upper()]+review_to_words(x[1]), x[2]], dtype=object)
    except:
        return np.array([x[0], None, x[2]], dtype=object)

def word_list_vaid(x):
    return np.array([x[0], [x[2].upper()]+review_to_words(x[1]), x[3]], dtype=object)

def word_list_test(x):
    return np.array([x[0], [x[2].upper()]+review_to_words(x[1]), x[2]], dtype=object)

def word_list_Df(CSV_PATH,output_path,columns):
    df = pd.read_csv(CSV_PATH)
    df[columns] = imap_unordered_bar(word_list_train, df[columns].values)
    df[columns].to_csv(f''+output_path,index=False)
    print('done. Check {} for the word_list csv file'.format(output_path))
    #result = set(df['comment_text'].flatten())
##########################################################################################################




def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    for i in data:
        for j in i:
            if j not in word_count:
                word_count[j] = 1
            else:
                word_count[j] += 1
    sorted_words = list({k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}.keys())
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict


def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)













def preprocess_data(train_X,train_y, valid_X,valid_lan,valid_y, test_id,test_X,test_lan,
                    cache_dir="./cache", cache_file="preprocessed_data.pkl"):
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
        words_train = list(map(review_to_words, train_X))
        print('train done')
        words_valid = list(map(review_to_words, valid_X))
        print('valid done')
        words_test = list(map(review_to_words, test_X))
        print('test done')
        #words_train = [review_to_words(comment) for comment in train_X]
        #words_valid = [review_to_words(comment) for comment in valid_X]
        #words_test = [review_to_words(comment) for comment in test_X]
        
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

















