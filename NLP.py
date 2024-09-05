import numpy as np
import nltk 
# nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()

def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words),dtype=np.float32)
    
    for idx , w in enumerate(words):
        if w in sentence_word:
            bag[idx] = 1
    return bag
    
    