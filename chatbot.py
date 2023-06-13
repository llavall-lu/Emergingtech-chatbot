import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras.models
#Downloading required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

#Initialising lemmatizer
lemmatizer = WordNetLemmatizer()

#Loading intents words classes and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbotmodel.h5')

#Prints classes (For testing purpose can be removed)
print(classes)

#this tokenizes and lemmatizes the words and cleans up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#this converts sentences into bag of words (Turns text into numerical values)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#This takes the sentence inputed and uses it to predict the intent of the user
def predict_class(sentence):
    bow = bag_of_words(sentence)
    resu = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(resu) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

#This uses the prediction from the function above to give a response from the responses in the JSON file
def get_response(user_input, intents_json):
    list_of_intents = intents_json['intents']
    matched_intents = []
#This part checks if what was inputed matches any of the intent patterns in the JSON file
    for intent in list_of_intents:
        for pattern in intent['patterns']:
            if pattern.lower() in user_input.lower():
                matched_intents.append(intent)
                break
#In the event that no matches were found (Likely with current dataset) then it will print "I'm sorry, but I'm not sure what you're asking."
    if not matched_intents:
        return "I'm sorry, but I'm not sure what you're asking."
#Due to the limited datasets and limited capability of the bot, it currently chooses from the responses randomely
    selected_intent = random.choice(matched_intents)
    responses = selected_intent['responses']
    result = random.choice(responses)

    return result

#This last section allows interaction with the program
print('Running')

while True:
    message = input("")

    if message.strip() == "": #This was added to stop the spacebar from interacting with the chatbot
        continue
#Get a response on the predicted intetnt
    ints = predict_class(message)
    resu = get_response(message, intents)
    print(resu)
