import pyttsx3

engine=pyttsx3.init("sapi5")
voices=engine.getProperty('voices')
engine.setProperty('voices', voices[1].id)
engine.setProperty('rate',170)

def say(Text):
    
    engine.say(text=Text)
    engine.runAndWait()
    print(f"A.I : {Text}")

    print("    ")
