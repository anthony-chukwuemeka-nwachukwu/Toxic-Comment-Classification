# Import packages
import torch
import pandas as pd
from sklearn.utils import shuffle
from multiprocessing import Pool
from tqdm import *
from googletrans import Translator

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