import datetime
import webbrowser
import os
import requests
import time
import pyautogui
from say import say
from listen import Listen
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def Time():
    time_str = datetime.datetime.now().strftime("%H:%M")
    say(f"The time is {time_str}")

def Date():
    date = datetime.date.today()
    say(f"Today's date is {date}")

def Gsearch(query=None):
    if not query or query.strip().lower() == "none":
        say("What should I search for?")
        query = Listen()
    if query and query.strip().lower() != "none":
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        say(f"Searching Google for {query}.")
    else:
        say("I couldn't understand the search query.")

def CloseTab():
    say("Closing the last opened browser tab.")
    pyautogui.hotkey('ctrl', 'w')

def PlayYouTubeMusic():
    import pywhatkit
    say("What should I play?")
    song = Listen()
    if song:
        pywhatkit.playonyt(song)
        say(f"Playing {song} on YouTube.")
    else:
        say("Song not recognized.")

def Weather():
    say("Tell me the city name.")
    city = Listen()
    if not city or city.lower() == "none":
        say("City not recognized.")
        return
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url).json()
        if res.get("cod") != 200:
            raise ValueError("Invalid city or API error")
        weather = res["weather"][0]["description"]
        temp = res["main"]["temp"]
        say(f"The weather in {city} is {weather} with temperature {temp}°C")
    except:
        say("Sorry, I couldn't fetch the weather.")

def SetAlarm():
    say("In how many seconds should I set the alarm?")
    seconds = Listen()
    try:
        sec = int(''.join(filter(str.isdigit, seconds)))
        say(f"Alarm set for {sec} seconds from now.")
        time.sleep(sec)
        say("Time's up!")
    except:
        say("Invalid time.")

def AddToDo():
    say("What task should I add to your to-do list?")
    task = Listen()
    if task:
        with open("todo.txt", "a") as f:
            f.write(task + "\n")
        say("Added to your to-do list.")
    else:
        say("I couldn't understand the task.")

def Noninputfun(tag, user_input=None):
    if tag == "date":
        Date()
    elif tag == "time":
        Time()
    elif tag == "alarm":
        SetAlarm()
    elif tag == "todo":
        AddToDo()
    elif tag == "google_search":
        query = ""
        if user_input:
            triggers = ["search", "search this", "google this", "search for", "find", "look up"]
            lower_input = user_input.lower()
            for trigger in triggers:
                if trigger in lower_input:
                    query = lower_input.replace(trigger, "").strip()
                    break
            if not query:
                query = lower_input.strip()
        Gsearch(query)
    elif tag == "close tab":
        CloseTab()
    elif tag == "music":
        PlayYouTubeMusic()
