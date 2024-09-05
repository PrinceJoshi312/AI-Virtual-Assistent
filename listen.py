import speech_recognition as sr 
import pyaudio

def Listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 1
        r.energy_threshold = 300
        audio = r.listen(source, 0, 3)
    
    try:
        print('Recognizing...')
        query = r.recognize_google(audio, language='en-in')
        print(f'User said: {query}\n')

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return 'None'
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return 'None'
    
    query = str(query)
    return query.lower()

Listen()
