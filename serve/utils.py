import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

import pickle

import os
import glob

from googletrans import Translator
import emoji


ts = Translator()

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def translate_now(x):
    
    if type(x) == bytes:
        x = x.decode("utf-8")
        if x[:3].lower() != 'en ':
            try:
                return str(x[:2].upper() +" "+ts.translate(remove_emoji(x[2:]), dest='en').text).encode('utf-8')
            except:
                try:
                    return str(x[:2].upper() +" "+ts.translate(deEmojify(x[2:]), dest='en').text).encode('utf-8')
                except:
                    return str(x[:2].upper() +" "+ x[2:]).encode('utf-8')
        return x
    else:
        if x[:3].lower() != 'en ':
            try:
                return x[:2].upper() +" "+ts.translate(remove_emoji(x[2:]), dest='en').text
            except:
                try:
                    return x[:2].upper() +" "+ts.translate(deEmojify(x[2:]), dest='en').text
                except:
                    return x[:2].upper() +" "+ x[2:]
        return x


def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words

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