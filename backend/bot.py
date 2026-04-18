import os
import json
import random
import pickle
import numpy as np
import torch
from keras.models import load_model
from dotenv import load_dotenv
from say import say
from listen import Listen
from tasks import Noninputfun
import google.generativeai as genai
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")
chat = gemini.start_chat(history=[])

lemmatizer = WordNetLemmatizer()
last_response_text = ""
is_sleeping = False

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

def clean_response_text(text):
    # Remove markdown **bold** and *italic*
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Optional: Trim to 2-3 lines if response is too long
    sentences = text.strip().split('. ')
    return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    # Return empty list if no confident prediction
    if not results:
        return []

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if not intents_list:
        return None, None

    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tags'] == tag:
            if tag in ["time", "date", "google_search", "music", "alarm", "weather", "todo", "close tab"]:
                return "", tag
            return random.choice(i['responses']), tag
    return None, None

def generate_gemini_reply_with_context(query):
    try:
        response = chat.send_message(query)
        global last_response_text
        last_response_text = response.text
        return response.text
    except Exception:
        return "Sorry, I couldn't get a response from Gemini."


def Main():
    global is_sleeping

    try:
        sentence = Listen()
        if sentence is None or sentence.strip().lower() == "none":
            return

        sentence_lower = sentence.lower()

        if "shutdown" in sentence_lower:
            say("Shutting down. Goodbye!")
            exit()

        if is_sleeping and not any(wake in sentence_lower for wake in ["bugg", "wake up", "hey bugg"]):
            return

        if any(bye in sentence_lower for bye in ["bye", "exit", "goodbye", "sleep", "see you"]):
            say("Okay, going to sleep.")
            is_sleeping = True
            return

        if any(wake in sentence_lower for wake in ["bugg", "wake up", "hey bugg"]):
            say("Yes, I’m listening.")
            is_sleeping = False
            return

        intents_list = predict_class(sentence)
        response, tag = get_response(intents_list, intents)

        if response and tag not in ["google_search", "music", "alarm", "weather", "todo", "close tab"]:
            say(response)
        elif tag in ["time", "date", "google_search", "music", "alarm", "weather", "todo", "close tab"]:
            Noninputfun(tag, sentence)
        else:
            response = generate_gemini_reply_with_context(sentence)
            cleaned = clean_response_text(response)
            say(cleaned)



    except KeyboardInterrupt:
        say("Shutting down due to keyboard interrupt. Goodbye!")
        exit()


while True:
    Main()
