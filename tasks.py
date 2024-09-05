import datetime
from say import say


def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    say(time)
    
def Date():
    date = datetime.date.today()
    say(date)

def Gsearch():
    print("hello")
    


def Noninputfun(query):
    query = str(query)
    
    if "time" in query:
        Time()
    elif "date" in query:
        Date()
    elif "google search" in query:
        Gsearch()
