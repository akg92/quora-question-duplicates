"""
This file contains the preprocessing steps. Inline description is provided.
"""

import re
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import  pandas as pd
from autocorrect import spell
import os
import gc

"""
All the global configuration goes below.
"""
ENABLE_SPELL_CORRECT = True


SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}
COUNTER = 0

"""
This is  refered code from https://github.com/susanli2016/NLP-with-Python/blob/master/BOW_TFIDF_Xgboost_update.ipynb
Add more regex if you think required.
"""
def clean(text, stem_words=True):

    def pad_str(s):
        return ' ' + s + ' '

    if pd.isnull(text):
        return ''

    #    stops = set(stopwords.words("english"))
    # Clean the text, with the option to stem words.

    # Empty question

    if type(text) != str or text == '':
        return ''

    # Clean the text
    text = re.sub("\'s", " ",
                  text)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    # remove comma between numbers, i.e. 15,000 -> 15000

    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    #     # all numbers should separate from words, this is too aggressive

    #     def pad_number(pattern):
    #         matched_string = pattern.group(0)
    #         return pad_str(matched_string)
    #     text = re.sub('[0-9]+', pad_number, text)

    # add padding to punctuations and special chars, we still need them later

    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)

    #    def pad_pattern(pattern):
    #        matched_string = pattern.group(0)
    #       return pad_str(matched_string)
    #    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text)

    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']),
                  text)  # replace non-ascii word with special word

    # indian dollar

    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE)
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)

    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"

    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
    # Return a list of words

    ## get the auto-correct
    if ENABLE_SPELL_CORRECT:
        text = do_auto_correct(text)
    ## for debugging. It is quite slow
    global  COUNTER
    COUNTER += 1
    if not COUNTER%10000:
        print('completed {}'.format(COUNTER))
    ##
    return text


"""
    Do the spell correct. This has great impact in the encoding.
"""
def do_auto_correct(text):
    final_str = ""
    for x in text.split():
        corrected = spell(x)
        final_str += corrected+" "
    if len(final_str)>2:
        final_str += final_str[:len(final_str)-1]
    return final_str



"""
Clean and store the processed data.
"""
def clean_and_save(src_filename,dest_filename,enable_autocorrect=False):
    ## The global is set for compactability
    global ENABLE_SPELL_CORRECT

    if enable_autocorrect != ENABLE_SPELL_CORRECT:
        ENABLE_SPELL_CORRECT = enable_autocorrect

    df = pd.read_csv(src_filename)
    df['question1'] = df['question1'].apply(clean)
    gc.collect()
    print("Finished Question1")
    df['question2'] = df['question2'].apply(clean)
    print("Finished Question2")
    gc.collect()
    ## remove the index
    df.to_csv(dest_filename,index=False)
    print('Finished processing file {}'.format(src_filename))
    return  df


"""
    Too much run time. Doing multi--processing optionally
"""
def do_multiprocessing(train_file,train_processed_file,test_file,test_processed_file):
    from multiprocessing import  Process
    p1 = Process(target=clean_and_save,args=(train_file,train_processed_file))
    p2 = Process(target=clean_and_save, args=(test_file, test_processed_file))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


"""
Main method does initial processing.
"""
if __name__=='__main__':
    data_dir = '../data'
    test_file = os.path.join(data_dir,'test.csv')
    train_file = os.path.join(data_dir,'train.csv')
    test_processed_file = os.path.join(data_dir,'test_processed.csv')
    train_processed_file  = os.path.join(data_dir,'train_processed.csv')
    #print(test_file)
    ## train and save the files
    ## to avoid spell coreect comment the below line
    ENABLE_SPELL_CORRECT = False
    """ for single process uncomment below"""
    clean_and_save(test_file,test_processed_file)
    clean_and_save(train_file,train_processed_file)
    #do_multiprocessing(train_file,train_processed_file,test_file,train_processed_file)
