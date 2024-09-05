import numpy as np
import torch
import torch.nn as nn
import json
from torch .utils.data import DataLoader,Dataset
from NLP import bag_of_words, tokenizer, stem
from brain import Neuralnet


with open ('intents.json','r') as f:
    intents= json.load(f)
        
all_words= []
tags= []
xy= []

for intent in intents['intents']:
    tag= intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        w = tokenizer(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
ignore_words = ['?','/','*','.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags= sorted(set(tags))

x_train = []
y_train = []

for(pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)
    
    lable = tags.index(tag)
    y_train.append(lable)
    
x_train = np.array(x_train)
y_train = np.array(y_train)

num_epoches = 1000
batch_size = 8
learning_rate = .001
input_size = len(x_train[0])
hidden_size = 8
output_layer = len(tags)

class ChatData(Dataset):
    
    def __init__(self):
        self.n_semples =len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_semples
    
dataset = ChatData()

train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Neuralnet(input_size,hidden_size,output_layer).to(device=device)
criteria = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(),lr=learning_rate)
   
for epoch in range(num_epoches):
    for(words,lables) in train_loader:
        words = words.to(device)
        lables = lables.to(dtype=torch.long).to(device)
        output = model(words)
        loss = criteria(output,lables)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if(epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epoches}], Loss: {loss.item(): .4f}')

data = {"modle_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size":hidden_size,
        "output_layer":output_layer,
        "tags":tags,
        "all_words":all_words}

FILE = "TrainData.pth"
torch.save(data,FILE)
