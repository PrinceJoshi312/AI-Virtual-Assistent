
# Importing necessary libraries
import random  # to choose a random response from provided responses in intents.json
import json  # to operate on intents.json
import pickle  # for serialization
import numpy as np  # for array operations

import nltk  # Natural Language Toolkit
from nltk.stem import WordNetLemmatizer  # to identify different words with the same meaning as a single word

import tensorflow as tf
from keras import layers, models, optimizers, Sequential

# Initializing the lemmatizer
lemmatizer = WordNetLemmatizer()

# Loading intents from the intents.json file
intents = json.loads(open('intents.json').read())

# Initializing empty lists and variables
words = []
classes = []
documents = []
ignoreLetters = []

# Extracting words, classes, and documents from intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)  # Tokenizing words in the pattern
        words.extend(wordList)  # Extending words list with tokenized words
        documents.append((wordList, intent['tags']))  # Appending (wordList, tag) tuple to documents
        if intent['tags'] not in classes:
            classes.append(intent['tags'])  # Appending unique tags to classes

# Lemmatizing words and removing ignored letters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # Sorting and removing duplicates from words list
classes = sorted(set(classes))  # Sorting and removing duplicates from classes list

# Dumping words and classes into pickle files for serialization
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initializing training data
training = []
outputEmpty = [0] * len(classes)  # Creating an empty list of length equal to the number of classes

# Processing documents to create training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)  # Creating bag of words representation
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1  # Marking the position of the class with 1 in output row
    training.append((bag, outputRow))  # Appending (bag, outputRow) tuple to training data

# Shuffling the training data
random.shuffle(training)

# Extracting bags and output rows separately
train_x = np.array([bag for bag, _ in training])
train_y = np.array([outputRow for _, outputRow in training])

# Creating a Sequential model
model = Sequential()
# Adding layers to the model
model.add(layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

# Configuring optimizer
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting the model to the training data
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)
# Saving the model
model.save('chatbot.h5')
