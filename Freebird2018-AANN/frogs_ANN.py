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

#feature from audio using librosa
#import librosa

from frogs_hyperparams import hyperparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Models, input is of shape 
class FrogsNet(nn.Module):
    def __init__(self, model_layout:dict={}, activation_fn=nn.ReLU):
        super(FrogsNet, self).__init__()
        try:
            self.i = model_layout["input"]
            self.o = model_layout["output"]
            self.h = model_layout["hidden"]
            self.l = model_layout["hidden_layers"]
            self.d = model_layout["dropout"]
        except KeyError as ecpt:
            raise ValueError("Passed model_layout with invalid params: ", ecpt)
        self.f = activation_fn

        # Return to this and override or correct initialized  weights for ReLU.
        # Default init is uniform with 0 mean and can result in dead units.
        # Update: nevermind, nn.ReLU sets correct weights in it class __init__
        # Update: nn.Linear set initial weights with kaiming_uniform_ for leaky_relu nonlinearity by default.
        # no need to extend Linear class with such functionality.
        self.i2h = nn.Sequential(
            nn.Linear(self.i, self.h),
            nn.GroupNorm(1, self.h),
            #nn.BatchNorm1d(self.h), #equal in performance with GroupNorm for 1 group.
            self.f(),
            nn.Dropout(p=self.d), #doesn't seems to affect acc if used with layernorm
            )
        self.h2h = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.GroupNorm(1, self.h),
            self.f(),
            nn.Dropout(p=self.d), #doesn't seems to affect acc if used with layernorm
            )
        self.h2o = nn.Sequential(
            nn.Linear(self.h, self.o),
            )
        # To not use group norm, we remove these layers from container
        if not model_layout['use_groupnorm']:
            del self.i2h[1]
            del self.h2h[1]
            

    def forward(self, x):
        out = self.i2h(x)

        if self.l > 1:
            for _ in range(self.l):
                out = self.h2h(out)

        out = self.h2o(out)
        return out

class FrogsCNN(nn.Module):
    def __init__(self):
        super(FrogsCNN, self).__init__()
        self.cnv = nn.Sequential(
            nn.Conv1d(1, 11, 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(21*11, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 10)
            )

    def forward(self, x):
        x = x.view(-1, 1, 22) #reshape from (22) => (2,11)
        x = self.cnv(x)
        x = x.view(-1, 11*21) #conv returns (featuremaps=11, i-k=21)
        x = self.fc(x)
        return x

def NLLRLoss(predicted_classes, correct_class, reduction='mean'):
    """
    Calculates NLLRLoss described in paper: https://arxiv.org/pdf/1804.10690.pdf

    L(x,y_i) = -[log(x[y_i] - log((sum(x) - x[y_i]))]

    Inputs should be raw outputs of net. Since this net used crossentropyloss, 
    this function will apply softmax as well to play well with current model. 
    inputs: (prediction:Tensor, correctClassIndex:number) ->
    output: loss:Tensor after reduction operation, as backward() expects a scalar value
    """
    reduction_ops = {
        'mean': torch.mean,
        'sum' : torch.sum
        }
    #Use of regular softmax was unstable, which quickly led to nan-losses, prob since denominator tends to 0.
    #This mess gives results comparable to common crossentropy-loss imported from pytorch module.
    probs_class = nn.functional.log_softmax(predicted_classes)

    """
    Could set probs to 0 after extracting correct_probs before summing. This could simplify division into substraction.
    Based on snippet from: https://discuss.pytorch.org/t/update-specific-columns-of-each-row-in-a-torch-tensor/5597

    x = torch.FloatTensor([[ 0,  1,  2],
                           [ 3,  4,  5],
                           [ 6,  7,  8],
                           [ 9, 10, 11]])

    j = torch.arange(x.size(0)).long()

    x[j, correct_class] = 0
    """
    correct_probs = torch.gather(probs_class, 1, correct_class.view(-1,1)).squeeze()

    #Implementation of above scheme..

    #Detach disconnects this tensor from accumulating gradients.
    #Reason: we manipulate values and don'twant them to affect gradients.
    sum_incorrect_probs = probs_class.clone().detach()
    j = torch.arange(sum_incorrect_probs.size(0)).long()
    sum_incorrect_probs[j, correct_class] = 0

    sum_incorrect_probs = torch.exp(sum_incorrect_probs)

    sum_incorrect_probs = torch.sum(sum_incorrect_probs, 1)

    sum_incorrect_probs = torch.log(sum_incorrect_probs)

    #Calculate negative log loss. 
    loss = torch.neg(correct_probs - sum_incorrect_probs)
    return reduction_ops[reduction](loss)

# Combines LogSoftmax and NLLLoss(negative log likelihood loss) in one layer
#loss_function = nn.CrossEntropyLoss()
#loss_function = NLLRLoss

def build_model_optimizer(model_layout=None, learning_rate=1e-4, w_decay=0, activation_function=nn.ReLU, **kwargs):
    """Build and return model and optimzer to simplify flow"""
    if model_layout is None: 
        model_layout = {
            "input" : kwargs['input_nodes'],
            "output" : kwargs['output_nodes'],
            "hidden" :kwargs['hidden_nodes'],
            "hidden_layers" : kwargs['hidden_layers'],
            "dropout" : kwargs['dropout_prop'],
            "use_groupnorm" : kwargs['use_groupnorm']
            }

    model = FrogsNet(model_layout, activation_fn=activation_function).to(device)
    #model = FrogsCNN().to(device)
    #weight_decay=1e-3 in few research papers such as in https://arxiv.org/pdf/1711.05101.pdf
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    #optimizer = optim.Adadelta(model.parameters())

    return model, optimizer

def train(model, optimizer, loss_function, train_loader=None, epochs=0, **kwargs):
    if train_loader is None:
        raise TypeError("Callee is required to pass in valid instance of dataloader. This is fatal.")
    
    #train_stats = []
    training_stats = {'epoch':[], 'loss':[], 'training_eval':[] }

    verbose = kwargs.get('verbose', False)
    report_interval = kwargs.get('report_interval', 1)
    for epoch in range(1, epochs + 1):
        model.train()
        
        try:
            loss_print = 0

            for batch, (data, labels) in enumerate(train_loader):
                #print("Enumerate Train", batch, data, label)
                with torch.autograd.detect_anomaly():
                    data, labels = data.to(device), labels.to(device)

                    #print("Batch #", batch)
                    optimizer.zero_grad()
                    predicted = model(data)
                    loss = loss_function(predicted, labels)
                    loss.backward() #calc w.grads for model
                    optimizer.step() #apply w.grads in backprop pass
                    loss_print += loss.item()

                    if verbose and (batch % report_interval == 0):
                        print("Epoch:%d, Batch:%d, Loss:%f" % (epoch, batch, loss_print))
                        loss_print = 0

            #Append statistics to return dict after each epoch
            training_stats['epoch'].append(epoch)
            training_stats['loss'].append(loss_print)

            #somestats = {'epoch':epoch, 'loss':loss_print}
            if 'eval_loader' in kwargs:
                train_acc, _ = eval(model, kwargs['eval_loader'])
                training_stats['training_eval'].append(train_acc)
                #somestats['train_acc'] = train_acc
                if verbose:
                    print(f"Evaluation during training loop# epoch {epoch} : accuracy {train_acc}")

            
            #train_stats.append(somestats)
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            raise
    #return train_stats
    return training_stats

def eval(model, eval_loader=None, species_names=None, **kwargs):
    if eval_loader is None:
        raise TypeError("Callee is required to pass in valid instance of dataloader. This is fatal.")
    #if not isinstance(species_names, list):
    #    raise TypeError("species_names must be a list of strings representing true labels.")
    with torch.no_grad():
        model.eval()
        
        total = 0
        correct = 0
        verbose = kwargs.get('verbose', False)

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
        if verbose:
            print("Accuracy: %f" % (accuracy * 100.0))
        
        return accuracy, cm

def run(n_folds=0, epochs=0, batch_size=1, **kwargs):
    """ Runs ANN-classifier """
    frogs_csv = Path.cwd() / 'Data' / 'Frogs_MFCCs.csv'
    #timer start
    timer_start = time.perf_counter()
    run_kfolds = kwargs.get('run_kfolds', False)

    loss_function = {
        'crossentropy' : nn.CrossEntropyLoss(),
        'nllr' : NLLRLoss
        }.get(kwargs.pop('loss_function'))

    if run_kfolds:
        print("===========================")
        print(f"Running ANN-classifier against {n_folds}-fold cross evaluation")
        if 'epochs' in kwargs:
            epochs = kwargs.pop('epochs')
        stats = []
        train_stats = []
        for ftrain, fevail, species_names in generate_kfolds_datasets(frogs_csv, kfolds=n_folds):
            print('.', end='', flush=True)
            model, optimizer = build_model_optimizer(**kwargs)
        
            train_loader = D.DataLoader(ftrain, batch_size, shuffle=True)
            eval_loader = D.DataLoader(fevail, batch_size, shuffle=True)
            
            #specify eval_loader = eval_loader if you want eval stats after each epoch
            train_loss_stats = train(model, optimizer, loss_function, train_loader=train_loader, epochs=epochs, eval_loader = eval_loader, **kwargs)
            acc_nfold, cm_nfold = eval(model, eval_loader=eval_loader, species_names=species_names)
            stats.append((acc_nfold, cm_nfold))
            train_stats.append(train_loss_stats)
        print("") #newline
        #get averaged accuracy
        n_acc = sum([a for a, _ in stats])/float(len(stats))
        std_deviation = ((sum([(a - n_acc) ** 2 for a, _ in stats]))/float(len(stats) - 1) ) ** 0.5

        cuml_cm = ConfusionMatrix(labels=species_names)
        for _, cm_nfold in stats:
            cuml_cm + cm_nfold

        run_time = time.perf_counter() - timer_start

        if kwargs.get('verbose', False):
            print("n-fold mean accuracy(std): %f (%f)" % (n_acc * 100.0, std_deviation * 100))
            print(f"======= {n_folds}-fold cumulutive confusion matrix =======")
            print(cuml_cm)
            print(f"Training loop and eval for k-fold processing length: {run_time} secs")
        #print(train_stats)
        return {'Accuracy':n_acc, 
                'ConfusionMatrix':cuml_cm, 
                'EvaluationStats':[eval_acc for eval_acc, _ in stats],
                'Runtime':run_time, 
                'TrainingStats':train_stats,
                'Labels':species_names
               }
    else:
        print("== Training loop ==")
        model, optimizer = build_model_optimizer()
        train(model, optimizer)
        print("== Evaluating loop ==")
        acc_split, cm_split = eval(model)
        #print(cm_split)
        run_time = time.perf_counter() - timer_start
        print(f"Training loop and eval split processing length: {run_time} secs")
        return {'Accuracy':acc_split, 'ConfusionMatrix':cm_split, 'Runtime':run_time}

def save_model(to_file):
    """ Save model to file with eval statistics. """

    raise NotImplementedError

def load_model(from_file):
    """ Load saved model from file. """

    raise NotImplementedError

if __name__ == "__main__":
    r = run(**hyperparams)