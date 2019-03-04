import os
import logging
import time
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as D
from torchvision import datasets, transforms

#from frogs_data import Frogs_Dataset
from frogs_utils import generate_datasets, generate_kfolds_datasets, ConfusionMatrix
from pathlib import Path

#hyperparameter
epochs = 20
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 200
report_interval = 500
use_stratified = True
n_folds = 10
run_kfolds = False
model_layout = {
    "input":22,
    "output":10,
    "hidden":260,
    "dropout":0.3
    }
#dataloading
frogs_csv = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
train_set, eval_set, species_names = generate_datasets(frogs_csv, split=0.9, stratified=use_stratified)

train_loader = D.DataLoader(train_set, batch_size)
eval_loader = D.DataLoader(eval_set, batch_size)

# Models, input is of shape 
class FrogsNet(nn.Module):
    def __init__(self, model_layout:dict={}):
        super(FrogsNet, self).__init__()
        try:
            self.i = model_layout["input"]
            self.o = model_layout["output"]
            self.h = model_layout["hidden"]
            self.d = model_layout["dropout"]
        except KeyError as ecpt:
            raise ValueError("Passed model_layout with invalid params: ", ecpt)
        self.f = nn.ReLU
        # *** Generates real good results with train/test split but tanks against CV ***
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
            self.f(),
            nn.Dropout(p=self.d),
            nn.Linear(self.h, self.o)
            )

    def forward(self, x):
        out =  self.block(x)
        return out

model = FrogsNet(model_layout).to(device)
#weight_decay=1e-3 in few research papers such as in https://arxiv.org/pdf/1711.05101.pdf
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

#timer start
timer_start = time.perf_counter()

def train(model, optimizer, loss_function=loss_function, train_loader=train_loader, epochs=epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        try:
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

def eval(model, eval_loader=eval_loader, species_names=species_names):
    with torch.no_grad():
        model.eval()
        print("Evaluation loop")
        total = 0
        correct = 0
    
        labels_collection = {key:0 for key in range(10)}
        cm = ConfusionMatrix(labels=species_names)

        for batch, (data, labels) in enumerate(eval_loader):
            data, labels = data.to(device), labels.to(device)
            predicted = model(data)
            #Create a tensor with class indexes

            #predicted is [class0, class1, ... , classN-1]
            #where each element is models 'energy' representing probability of label for instance
            #torch.max(, 1) returns an tensor with index along axis=1 e.g along rows.
            #so max([[0, 0.25, 1, 0.25], [0.15, 0.6, 0.25, 0]],axis=1) => [2, 1] with shape (2, 1)

            _, predicted_label = torch.max(predicted.data, 1)
            #.size(0) is length of row dimension from matrix
            total += labels.size(0)
            #Sum over tensor where predicted label == target label and increment correct by that value
            correct += (predicted_label == labels).sum().item()

            #collect information on label distributation in eval set
            for l in labels.tolist():
                labels_collection[l] += 1

            #Tally up confusionmatrix, datafrom cuda needs to moved to system memory first.. there has to be a better way
            #Perhaps create a confusion matrix for each batch with tensors and move that to cpu and do a elementwise add?
            cm.add_batch(predicted_labels=predicted_label.cpu().numpy() , target_labels=labels.cpu().numpy())

        accuracy = correct / total
        print("Accuracy: %f" % (accuracy * 100.0))
        print("Eval loop had following instances:")
        print(labels_collection)
        print("Number of eval samples: %d" % total)
        #print(cm)
        return accuracy, cm

#timer end and print


train(model, optimizer)
_, conmat = eval(model)
print(conmat)
print("Training loop and eval processing length: %f secs" % (time.perf_counter() - timer_start))
#NOTE: This is latenightbodge --> whole file should be refactored into methods train(),eval()
if run_kfolds:
    print("===========================")
    print("Running 10-fold eval")

    stats = []
    
    
    for ftrain, fevail, _ in generate_kfolds_datasets(frogs_csv):
        
        model = FrogsNet(model_layout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loader = D.DataLoader(ftrain, batch_size)
        eval_loader = D.DataLoader(fevail, batch_size)
        
        for epoch in range(1, epochs + 1):

            model.train()
            try:
                loss_print = 0

                for batch, (data, labels) in enumerate(train_loader):

                    data, labels = data.to(device), labels.to(device)

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

            stats.append(accuracy)
    
    #get averaged accuracy
    n_acc = sum(stats)/float(len(stats))
    #n_acc *= 100
    print("n-fold mean accuracy: %f" % (n_acc * 100.0))
    #print(n_acc*100)

    print("Training loop and eval processing length: %f secs" % (time.perf_counter() - timer_start))


#Acc-metric should be used to compare and save checkpoint, implement that later