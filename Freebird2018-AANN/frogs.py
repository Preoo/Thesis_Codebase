import os
import logging
import time
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as D
from torchvision import datasets, transforms

from frogs_data import frogs_dataset
from frogs_utils import generate_datasets, generate_kfolds_datasets
from pathlib import Path

#hyperparameter
epochs = 20
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
report_interval = 500
use_stratified = True
n_folds = 10
run_kfolds = True
#dataloading
frogs_csv = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
train_set, eval_set, species_names = generate_datasets(frogs_csv, split=0.9, stratified=use_stratified)

train_loader = D.DataLoader(train_set, batch_size)
eval_loader = D.DataLoader(eval_set, batch_size)

# Models, input is of shape 
class FrogsNet(nn.Module):
    def __init__(self, inputs=22, outputs=10, hidden=10):
        super(FrogsNet, self).__init__()
        
        self.i = inputs
        self.o = outputs
        self.h = hidden
        #self.h2 = max(int(self.h * 3), self.o)
        #self.h3 = max(int(self.h * 2), self.o)
        self.f = nn.ReLU()
        #self.block = nn.Sequential(
        #    nn.Linear(self.i, self.h2),
        #    nn.BatchNorm1d(self.h2),
        #    nn.ELU(),
        #    nn.Dropout(p=0.2),
        #    nn.Linear(self.h2, self.h),
        #    nn.BatchNorm1d(self.h),
        #    nn.ELU(),
        #    nn.Dropout(p=0.2),
        #    nn.Linear(self.h, self.o),
        #    nn.BatchNorm1d(self.o)
        #    #CrossEntropyLoss combines LogSoftmax and NLLoss
        #    )

        self.block = nn.Sequential(
            nn.Linear(self.i, self.h),
            self.f,
            nn.Dropout(p=0.3),
            nn.Linear(self.h, self.o)
            )

    def forward(self, x):
        out =  self.block(x)
        return out

def loss_fn(pred, target):
    a = cel_loss(pred, target)
    b = reg_loss(torch.argmax(pred).float(), target.float())
    return a + (b * 0.000001)

model = FrogsNet(inputs=22,outputs=10,hidden=260).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
cel_loss = nn.CrossEntropyLoss()
reg_loss = nn.MSELoss()
loss_function = cel_loss

#timer start
timer_start = time.perf_counter()
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
    
    labels_collection = {key:0 for key in range(10)}
    

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

        #collect information on label distributation in eval set
        for l in labels.tolist():
            labels_collection[l] += 1

    accuracy = correct / total
    print("Accuracy: %f" % (accuracy * 100.0))
    print("Eval loop had following instances:")
    print(labels_collection)
    print("Number of eval samples: %d" % total)

#timer end and print
print("Training loop and eval processing length: %f secs" % (time.perf_counter() - timer_start))

#TODO: This is latenightbodge --> whole file should be refactored into methods train(),eval()
if run_kfolds:
    print("===========================")
    print("Running 10-fold eval")

    stats = []
    
    
    for ftrain, fevail, _ in generate_kfolds_datasets(frogs_csv):
        
        model = FrogsNet(inputs=22,outputs=10,hidden=22).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        #loss_function = nn.CrossEntropyLoss()
        train_loader = D.DataLoader(ftrain, batch_size)
        eval_loader = D.DataLoader(fevail, batch_size)
        
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

        with torch.no_grad():
            model.eval()
            print("Evaluation loop")
            total = 0
            correct = 0
    
            labels_collection = {key:0 for key in range(10)}
    

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

                #collect information on label distributation in eval set
                for l in labels.tolist():
                    labels_collection[l] += 1

            accuracy = correct / total
            print("Accuracy: %f" % (accuracy * 100.0))
            print("Eval loop had following instances:")
            print(labels_collection)
            print("Number of eval samples: %d" % total)

            stats.append(accuracy)
    
    #get averaged accuracy
    n_acc = sum(stats)/float(len(stats))
    print("n-folds")
    print(n_acc*100)

print("Training loop and eval processing length: %f secs" % (time.perf_counter() - timer_start))
#exit and save checkpoint
#Frogs().test()