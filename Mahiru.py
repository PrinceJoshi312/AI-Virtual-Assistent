import torch
import json
import random
from tasks import Noninputfun
from say import say
from listen import Listen
from brain import Neuralnet
from NLP import bag_of_words, tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json",'r') as json_data:
    intents = json.load(json_data)
    
FILE = "TrainData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_layer = data['output_layer']
all_words = data["all_words"]
tags = data["tags"]
model_state = data["modle_state"]

model = Neuralnet(input_size,hidden_size,output_layer).to(device)
model.load_state_dict(model_state)
model.eval()


Name = "Bugg"

def Main():
    sentense = Listen()
    
    if sentense == "bye":
        exit()
    sentense = tokenizer(sentense)
    X = bag_of_words(sentense,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _ , predicted = torch.max(output,dim=1)
    
    tag = tags[predicted.item()]
    probab = torch.softmax(output, dim=1)
    prob = probab[0][predicted.item()]
    
    if prob.item() > 0.70:
        for intent in intents['intents']:
            if tag ==intent["tag"]:
                reply = random.choice(intent["responses"])
                
                if "time" in reply:
                    Noninputfun(reply)
                    
                elif "date" in reply:
                    Noninputfun(reply)
                    
                else:
                    say(reply)
while True:                    
    Main()