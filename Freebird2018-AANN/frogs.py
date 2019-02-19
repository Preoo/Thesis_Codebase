import os
import logging
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as D
from torchvision import datasets, transforms

from frogs_data import frogs_dataset
from frogs_utils import generate_datasets
from pathlib import Path

#hyperparameter
epochs = 12
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
report_interval = 100
use_stratified = True
#dataloading
frogs_csv = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
train_set, eval_set, species_names = generate_datasets(frogs_csv, split=0.9, stratified=use_stratified)

train_loader = D.DataLoader(train_set, batch_size)
eval_loader = D.DataLoader(eval_set, batch_size)

# Models, input is of shape 
class FrogsNet(nn.Module):
    def __init__(self, inputs=22, outputs=10, hidden=10):
        super(FrogsNet, self).__init__()
        #init variables
        self.i = inputs
        self.o = outputs
        self.h = hidden
        self.block = nn.Sequential(
            nn.Linear(self.i, self.h),
            nn.Tanh(),
            nn.Dropout(0,3),
            nn.Linear(self.h, self.o)
            )
    def forward(self, x):
        x = self.block(x)
        
        return x

model = FrogsNet(inputs=22,outputs=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

#training

for epoch in range(1, epochs + 1):
    model.train()
    try:
        #print("Training loop for epoch %d" % epoch)

        loss_print = 0

        for batch, (data, labels) in enumerate(train_loader):
            #print("Enumerate Train", batch, data, label)

            data, labels = data.to(device), labels.to(device)

            #print("Batch #", batch)
            optimizer.zero_grad()
            predicted = model(data)
            loss = loss_function(predicted, labels)
            loss.backward()
            optimizer.step()
            loss_print += loss.item()

            if batch % report_interval == 0:
                print("Epoch:%d, Batch:%d, Loss:%f" % (epoch, batch, loss_print))
                loss_print = 0
    except (KeyboardInterrupt, SystemExit):
        print("Exiting...")
        raise
#eval
with torch.no_grad():
    model.eval()
    print("Evaluation loop")
    total = 0
    correct = 0
    for batch, (data, labels) in enumerate(eval_loader):
        #print("Sample")
        data, labels = data.to(device), labels.to(device)
        predicted = model(data)
        #Create a tensor with class indexes
        _, predicted_label = torch.max(predicted.data, 1)
        #.size(0) is length of row dimension from matrix
        total += labels.size(0)
        #Sum over tensor where predicted label == target label and increment correct by that value
        correct += (predicted_label == labels).sum().item()
    accuracy = correct / total
    print("Accuracy: %f" % accuracy)
#exit and save checkpoint

#Frogs().test()