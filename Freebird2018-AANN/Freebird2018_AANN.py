import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as D
from Freebird2018DataSet import Freebird2018DataSet
from Features import FeaturesMFCC

parser = argparse.ArgumentParser(description='FreeBird2018_AANN_exp')
parser.add_argument('--batch-size', type=int, default=4, help='Input batch size, default 4')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Number of epoch to train')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Use cpu as device instead of CUDA')
parser.add_argument('--file-dir', help='Directory of metadata and ./wav folder')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
args.file_dir = args.file_dir or os.path.join('F:\\Downloads\\', 'ff1010bird_wav')

feature_method = FeaturesMFCC()

fb_dataset = Freebird2018DataSet(args.file_dir, feature_method)

dataset_len = len(fb_dataset)
train_len = int(0.8*dataset_len)
eval_len = dataset_len - train_len
train_set, eval_set = D.random_split(fb_dataset, lengths=[train_len, eval_len])

train_loader = D.DataLoader(train_set, batch_size = args.batch_size, shuffle=True, **kwargs)

eval_loader = D.DataLoader(eval_set, batch_size = args.batch_size, shuffle=True, **kwargs)
print('Trainset has: ' + str(len(train_set)) + '. Evalset has: ' + str(len(eval_set)))
# Model AANN as a class here
class AANN(nn.Module):
    def __init__(self):
        super(AANN, self).__init__()

        #define layers

    #define forward pass with ^ layers, should return output of model
    def forward(self, x):
        return x

model = AANN().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-4)


# Training loop

# Validation loop

# Printing statistics
