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
epochs = 1
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
#dataloading
frogs_csv = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
train_set, eval_set, species = generate_datasets(frogs_csv)

train_loader = D.DataLoader(train_set, batch_size)
eval_loader = D.DataLoader(eval_set, batch_size)

#model
class FrogsNet(nn.Module):
    def __init__(self):
        super(FrogsNet, self).__init__()
        #init variables

    def forward(self, x):
        return x


#optimizer = optim.Adam()
#criterion = nn.MSELoss()

#training
for epoch in range(1, epochs + 1):
        try:
            print("Training loop for epoch %d" % epoch)
            for batch, (data, label) in enumerate(train_loader):
                print("Enumerate Train", batch, data, label)

        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            raise
#eval
with torch.no_grad():
        print("Evaluation loop")
        
#exit and save checkpoint

#Frogs().test()