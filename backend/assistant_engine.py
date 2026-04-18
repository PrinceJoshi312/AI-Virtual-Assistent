import os
# Suppress TensorFlow warnings and oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import random
import pickle
import numpy as np
import torch
from keras.models import load_model
from dotenv import load_dotenv
from google import genai
from nltk.stem import WordNetLemmatizer
import nltk
import re
import datetime
import webbrowser
import requests
import pyautogui
import psutil
import wikipedia
import subprocess
import ctypes
import os
import sys

# Windows-specific imports
try:
    import winshell
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

load_dotenv()

class AssistantEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('intents.json').read())
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        
        # Use compile=False to avoid the warning, then compile manually
        self.model = load_model('chatbot.h5', compile=False)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        self.OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
        self.OLLAMA_URL = "http://localhost:11434/api/generate"

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        ERROR_THRESHOLD = 0.5 # Lowered threshold to pick up more intents
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]

    def get_response(self, intents_list):
        if not intents_list:
            return None, None
        tag = intents_list[0]['intent']
        for i in self.intents['intents']:
            if i['tags'] == tag:
                return random.choice(i['responses']), tag
        return None, None

    def generate_ollama_reply(self, query):
        try:
            payload = {
                "model": "llama3", # Defaulting to llama3, can be changed
                "prompt": query,
                "stream": False
            }
            response = requests.post(self.OLLAMA_URL, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json().get("response")
            return None
        except Exception:
            return None

    def generate_gemini_reply(self, query):
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=query
            )
            return response.text
        except Exception as e:
            return f"Sorry, I couldn't get a response from my AI core."

    def execute_task(self, tag, user_input):
        user_input_lower = user_input.lower()
        
        if tag == "time":
            return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}"
            
        elif tag == "date":
            return f"Today is {datetime.date.today().strftime('%A, %B %d, %Y')}"
            
        elif tag == "music":
            import pywhatkit
            song = user_input_lower.replace("play", "").replace("on youtube", "").replace("music", "").strip()
            if not song or song == "music":
                return "WHAT_TO_PLAY" # Special signal for frontend
            pywhatkit.playonyt(song)
            return f"Opening YouTube to play '{song}'."

        elif tag == "weather":
            city = user_input_lower.split("in")[-1].strip() if "in" in user_input_lower else None
            if not city or city == user_input_lower:
                return "Which city's weather should I check?"
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.OPENWEATHER_API_KEY}&units=metric"
            try:
                res = requests.get(url).json()
                temp, desc = res["main"]["temp"], res["weather"][0]["description"]
                return f"The weather in {city.capitalize()} is {desc} with {temp}°C."
            except:
                return f"Could not fetch weather for {city}."

        elif tag == "screenshot":
            try:
                # Ensure directory exists
                screenshot_dir = os.path.join("backend", "screenshots")
                if not os.path.exists(screenshot_dir):
                    os.makedirs(screenshot_dir)
                
                filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                path = os.path.join(screenshot_dir, filename)
                
                screenshot = pyautogui.screenshot()
                screenshot.save(path)
                return f"Screenshot saved successfully in {screenshot_dir} as {filename}."
            except Exception as e:
                return f"Failed to take screenshot: {str(e)}"

        elif tag == "volume_up":
            if not IS_WINDOWS: return "Volume control is only supported on Windows."
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)
            current_vol = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(min(1.0, current_vol + 0.1), None)
            return "Volume increased by 10%."

        elif tag == "volume_down":
            if not IS_WINDOWS: return "Volume control is only supported on Windows."
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)
            current_vol = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(max(0.0, current_vol - 0.1), None)
            return "Volume decreased by 10%."

        elif tag == "mute":
            if not IS_WINDOWS: return "Mute control is only supported on Windows."
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = interface.QueryInterface(IAudioEndpointVolume)
                is_muted = volume.GetMute()
                volume.SetMute(1 if not is_muted else 0, None)
                return "Audio muted." if not is_muted else "Audio unmuted."
            except Exception as e:
                return f"Error toggling mute: {str(e)}"

        elif tag == "google_search":
            query = user_input_lower.replace("google", "").replace("search", "").replace("find", "").replace("search for", "").strip()
            if not query or query == "search":
                return "WHAT_TO_SEARCH" # Special signal for frontend
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Searching Google for '{query}'."

        elif tag == "close_tab":
            try:
                pyautogui.hotkey('ctrl', 'w')
                return "Closing the current tab."
            except Exception as e:
                return f"Error closing tab: {str(e)}"

        elif tag == "open_github":
            webbrowser.open("https://github.com")
            return "Opening GitHub."

        elif tag == "open_linkedin":
            webbrowser.open("https://linkedin.com")
            return "Opening LinkedIn."

        elif tag == "open_gmail":
            webbrowser.open("https://mail.google.com")
            return "Opening Gmail."

        elif tag == "lock_pc":
            if not IS_WINDOWS: return "Locking PC is only supported on Windows."
            ctypes.windll.user32.LockWorkStation()
            return "Locking your PC now."

        elif tag == "empty_recycle_bin":
            if not IS_WINDOWS: return "Emptying recycle bin is only supported on Windows."
            try:
                winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=True)
                return "Recycle bin emptied."
            except:
                return "Recycle bin is already empty or an error occurred."

        elif tag == "wikipedia":
            query = user_input_lower.replace("wikipedia", "").replace("search", "").replace("who is", "").replace("what is", "").strip()
            if query:
                try:
                    summary = wikipedia.summary(query, sentences=2)
                    return f"According to Wikipedia: {summary}"
                except:
                    return f"Could not find any Wikipedia info for '{query}'."
            return "What should I look up on Wikipedia?"

        elif tag == "open_app":
            if not IS_WINDOWS: return "Opening local apps via this method is only supported on Windows."
            if "notepad" in user_input_lower:
                subprocess.Popen(["notepad.exe"])
                return "Opening Notepad."
            elif "chrome" in user_input_lower or "browser" in user_input_lower:
                subprocess.Popen(["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"])
                return "Opening Google Chrome."
            elif "calculator" in user_input_lower:
                subprocess.Popen(["calc.exe"])
                return "Opening Calculator."
            return "Which app should I open?"

        elif tag == "open_folder":
            if not IS_WINDOWS: return "Opening folders is only supported on Windows in this version."
            if "downloads" in user_input_lower:
                os.startfile(os.path.join(os.path.expanduser('~'), 'Downloads'))
                return "Opening Downloads folder."
            elif "documents" in user_input_lower:
                os.startfile(os.path.join(os.path.expanduser('~'), 'Documents'))
                return "Opening Documents folder."
            return "Which folder should I open?"

        elif tag == "todo":
            task = user_input_lower.replace("add to todo", "").replace("remember task", "").replace("save task", "").replace("todo", "").replace("add", "").strip()
            if not task:
                return "What task should I add to your to-do list?"
            with open("todo.txt", "a") as f:
                f.write(task + "\n")
            return f"Added '{task}' to your to-do list."

        elif tag == "alarm":
            # For simplicity, we'll just acknowledge the request for now 
            # as a background timer might need a separate thread
            return "Alarm functionality is being integrated with your system clock."

        elif tag == "ip_address":
            ip = requests.get('https://api.ipify.org').text
            return f"Your public IP address is {ip}."

        elif tag == "system_info":
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            batt = psutil.sensors_battery()
            msg = f"System Status: CPU at {cpu}%, RAM at {ram}%."
            if batt:
                msg += f" Battery at {batt.percent}% ({'Charging' if batt.power_plugged else 'Discharging'})."
            return msg

        elif tag == "joke":
            try:
                joke_res = requests.get("https://official-joke-api.appspot.com/random_joke").json()
                return f"{joke_res['setup']} ... {joke_res['punchline']}"
            except:
                return "Why did the AI cross the road? To get to the other side of the data stream!"

        elif tag == "news":
            try:
                # Using a public RSS feed or simple news API might be better, 
                # but for simplicity let's use a generic fetch if possible or just a nice message
                return "I'm currently fetching the latest headlines for you. You can find more details on Google News: https://news.google.com"
            except:
                return "I couldn't fetch the news at the moment."

        elif tag == "thanks":
            return random.choice(["You're very welcome!", "Anytime!", "Glad I could help!", "No problem at all!"])

        return None

    def process_query(self, query):
        intents_list = self.predict_class(query)
        res_text, tag = self.get_response(intents_list)
        
        task_res = self.execute_task(tag, query)
        if task_res: return task_res
        
        if res_text and tag not in ["wikipedia", "calculator", "google_search", "music"]:
            return res_text
            
        # Fallback 1: Ollama (Offline LLM)
        ollama_res = self.generate_ollama_reply(query)
        if ollama_res:
            return re.sub(r"[\*\#]", "", ollama_res)
            
        # Fallback 2: Gemini
        return re.sub(r"[\*\#]", "", self.generate_gemini_reply(query))

