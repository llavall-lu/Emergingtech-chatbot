import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

#initilises the lemmatizer
lemmatizer = WordNetLemmatizer()

with open('intents.json', 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreLetters = ["?", '!', '.', ',',"'"]

#
for intent in intents['intents']:
    for pattern in intent['patterns']: #These refer to the JSON file
        wordList = nltk.word_tokenize(pattern) #This tokenizes the sentences
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes: #Checks to see if the intent is already present
            classes.append(intent['tag'])

#lemmatize and filter words
words = [lemmatizer.lemmatize(word.lower()) for word in words if not any(char in ignoreLetters for char in word)]
words = sorted(set(words))
classes = sorted(set(classes))

#this saves the words and classes and pkl files using the pickle import
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

#this uses the bag of words function to turn text into numberical values
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

#This shuffles the data
random.shuffle(training)

training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

#This creates a sequential model using the Keras API from tensor flow
model = tf.keras.Sequential() #this initialises the model
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu')) #adds the dense layer to the model
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu')) #this adds a smaller dense layer (64 instead of the above one which is 128)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax')) #This adds an output to the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5') #this saves it as a h5 file that can be used by chatbot.py