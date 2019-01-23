import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as D
from Freebird2018DataSet import Freebird2018DataSet
from Features import FeaturesMFCC

parser = argparse.ArgumentParser(description='FreeBird2018_AANN_exp')
parser.add_argument('--batch-size', type=int, default=1, help='Input batch size, default 4')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='Number of epoch to train')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Use cpu as device instead of CUDA')
parser.add_argument('--file-dir', default='F:\\Downloads\\ff1010bird_wav', help='Directory of metadata and ./wav folder')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
#device = torch.device("cuda" if args.cuda else "cpu")
device = "cpu"
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
        self.fc_in = nn.Linear(13, 38)
        self.c1 = nn.Linear(38, 4)
        self.fc_e = nn.Linear(4, 38)
        self.fc_out = nn.Linear(38, 13)


    def encode(self, x):
        h1 = F.relu(self.fc_in(x))
        return F.relu(self.c1(h1))

    def decode(self, x):
        h2 = F.relu(self.fc_e(x))
        return F.relu(self.fc_out(h2))

    #define forward pass with ^ layers, should return output of model
    def forward(self, x):
        x = x.view(-1, 13)
        x = self.encode(x)
        x = self.decode(x)
        
        return x

    def init_weights(self):
        pass

#AANN Classifier will be based on N-number of AANN's where N is number classes to classify. Thus we need a container class.
#Perhaps it should have a dict with [label]: aann<object> structure
class AANN_Classifer():
    def __init__(self, *args, **kwargs):
        self.classifiers = {}
        
    def add_new_model(self, model, label):
        
        self.classifiers[label] = (model, optim.Adam(model.parameters(), lr=1e-5))

    def classify(self, input):
        # should run input over all classifiers and classify it by highest score, 
        # return dict-key (used as label). ex: return '1' if input seems to have bird in it.
        loss_f = self.loss_function
        predicted_outputs = {label:loss_f(input.view(1,-1,13), model(input).view(1,-1,13)).item() for label, (model, _) in self.classifiers.items()}
        #predicted_label, _ = max(predicted_outputs, key=lambda x: x[1])
        predicted_label = '0'
        if predicted_outputs['0'] > predicted_outputs['1']:
            predicted_label = '1'
        return predicted_label
    
    def training_loop(self, trainingDataLoader:D.DataLoader):
        # based on label given as training data -> train corresponding aann from classifiers-dict.
        for label, (model, _) in self.classifiers.items():
            model.train()
        
        for batch, dataset in enumerate(trainingDataLoader):
            
            #for (input, label) in dataset:
            label = dataset[1][0]
            
            (model, optimizer) = self.classifiers[label]
            
            model.to(device)
            for elem in dataset[0]:
                elem = elem.float()
                predicted = model(elem)

                loss = self.loss_function(elem.view(1,-1,13), predicted.view(1,-1,13))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Sample %d :: Loss %f :: Model idx %d" % (batch, loss.item(), int(label)))
            
    def eval_loop(self, evalDataLoader:D.DataLoader):
        for label, (model, _) in self.classifiers.items():
            model.eval()

        accuracy = 0.0
        correct_labels = 0

        with torch.no_grad():
            for batch, dataset in enumerate(evalDataLoader):
                #for input, label in dataset:
                predicted_labels = []
                label = dataset[1][0]
                for e in dataset[0]:
                    e = e.float()
                    predicted_labels.append(self.classify(e))
                if max(predicted_labels, key=lambda o: predicted_labels.count(o)) == label:
                    correct_labels += 1

            accuracy = correct_labels / len(evalDataLoader)
            print("Eval accuracy: %f" % accuracy)

        
    def loss_function(self, target, predicted):
        #fn = nn.BCELoss()
        #fn = nn.BCEWithLogitsLoss()
        fn = nn.MSELoss()
        return fn(predicted, target)


aann_classifier = AANN_Classifer()

def main():
    print("main?")
    for label in fb_dataset.labels:
        aann_classifier.add_new_model(AANN(), label)
    for epoch in range(args.epochs):
        try:

            print("Training loop for epoch %d" % epoch)
            aann_classifier.training_loop(train_loader)
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            raise
    print("Evaluation loop")
    aann_classifier.eval_loop(eval_loader)

if __name__ == "__main__":
    main()