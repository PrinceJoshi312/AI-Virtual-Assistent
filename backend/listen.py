import speech_recognition as sr

def Listen():
    r = sr.Recognizer()
    r.energy_threshold = 300  # Can be tuned based on your mic sensitivity
    r.pause_threshold = 1.5   # Allows longer pauses while speaking

    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = r.listen(source, phrase_time_limit=15)  # Captures up to 15 seconds
            query = r.recognize_google(audio)
            print("User said:", query)
            return query.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out.")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError:
            print("Could not request results from the speech recognition service.")

    return "none"

