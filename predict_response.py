import nltk
from nltk.stem import PorterStemmer
import numpy as np
import random
import tensorflow as tf
import json
import pickle



# Load the model and data
model = tf.keras.models.load_model('./chatbot_model.h5')

# Load data files
with open('./intents.json', 'r') as file:
    intents = json.load(file)

with open('./words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('./classes.pkl', 'rb') as file:
    classes = pickle.load(file)

porter_stemmer = PorterStemmer()

def preprocess_user_input(user_input, words):
    bag_of_words = []

    # Tokenize user input
    tokens = nltk.word_tokenize(user_input)
    
    # Stem tokens
    stemmed_tokens = [porter_stemmer.stem(word.lower()) for word in tokens if word.isalpha()]

    # Remove duplicacy and sort tokens
    unique_sorted_tokens = sorted(set(stemmed_tokens))

    # Create bag of words (BOW) representation
    for w in words:
        bag_of_words.append(1) if w in unique_sorted_tokens else bag_of_words.append(0)

    return np.array([bag_of_words])

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input, words)
    inp = np.expand_dims(inp, axis=0)  # Ensure input is shaped correctly
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    
    # Extract the class from the predicted_class_label
    predicted_class = classes[predicted_class_label]
    
    # Select a random response from intents
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            bot_response = random.choice(intent['responses'])
            return bot_response
    
    return "I'm sorry, I didn't understand that."

# Example usage
print("Hi, I am Stella. How can I help you?")

while True:
    user_input = input('You : ')
    response = bot_response(user_input)
    print("Bot : ", response)
