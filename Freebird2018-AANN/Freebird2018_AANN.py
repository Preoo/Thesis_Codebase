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
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='Number of epoch to train')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Use cpu as device instead of CUDA')
parser.add_argument('--file-dir', default='F:\\Downloads\\ff1010bird_wav', help='Directory of metadata and ./wav folder')
parser.add_argument('--num-features', type=int, default=8, metavar='N', help='Number of MFCC Features')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
device = "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
args.file_dir = args.file_dir or os.path.join('F:\\Downloads\\', 'ff1010bird_wav')

feature_method = FeaturesMFCC(numFeatures=args.num_features)
n_feat = args.num_features
fb_dataset = Freebird2018DataSet(args.file_dir, feature_method)

dataset_len = len(fb_dataset)
train_len = int(0.8*dataset_len)
eval_len = dataset_len - train_len

train_set, eval_set = D.random_split(fb_dataset, lengths=[train_len, eval_len])

train_loader = D.DataLoader(train_set, batch_size = args.batch_size, **kwargs)

eval_loader = D.DataLoader(eval_set, batch_size = args.batch_size, **kwargs)

print('Trainset has: ' + str(len(train_set)) + '. Evalset has: ' + str(len(eval_set)))

# Model AANN as a class here
class AANN(nn.Module):
    def __init__(self, num_feat):
        super(AANN, self).__init__()
        self.nfeat = num_feat
        #define layers - input of 8
        self.encode = nn.Sequential(
            nn.Linear(self.nfeat, 38),
            nn.ReLU(True),
            nn.Linear(38, 4)
            )

        self.decode = nn.Sequential(
            nn.Linear(4,38),
            nn.ReLU(True),
            nn.Linear(38, self.nfeat),
            nn.Tanh()
            )


    #def encode(self, x):
    #    h1 = F.relu(self.fc_in(x))
    #    return F.relu(self.c1(h1))

    #def decode(self, x):
    #    h2 = F.relu(self.fc_e(x))
    #    return F.relu(self.fc_out(h2))
        #return F.tanh(self.fc_out(h2))

    #define forward pass with ^ layers, should return output of model
    def forward(self, x):
        #print(x.shape)
        #x = x.view(-1, self.nfeat)
        
        x = self.encode(x)
        x = self.decode(x)
        #x = x.view(1,-1, self.nfeat)
        return x

    def init_weights(self):
        pass

#AANN Classifier will be based on N-number of AANN's where N is number classes to classify. Thus we need a container class.
#Perhaps it should have a dict with [label]: aann<object> structure
class AANN_Classifer():
    def __init__(self, *args, **kwargs):
        self.classifiers = {}
        
    def add_new_model(self, model, label):
        #lr = 1e-3 <- copypaste since autocomplite is ass
        self.classifiers[label] = (model, optim.Adam(model.parameters()))

    def classify(self, input):
        # should run input over all classifiers and classify it by highest score, 
        # return dict-key (used as label). ex: return '1' if input seems to have bird in it.
        loss_f = self.loss_function
        #predicted_outputs = {label:loss_f(input.view(1,-1,args.num_features), model(input).view(1,-1,args.num_features)).item() for label, (model, _) in self.classifiers.items()}
        predicted_outputs = {label:loss_f(input.view(1,-1,args.num_features), model(input).view(1,-1,args.num_features)).item() for label, (model, _) in self.classifiers.items()}
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
            #label = '0'
            (model, optimizer) = self.classifiers[label]
            
            model.to(device)
            for frames in dataset[0]:
                optimizer.zero_grad()
                #print("Frames:: %d" % len(frames))
                #asd = 0
                for elem in frames:
                    elem = elem.float()
                    predicted = model(elem)

                    loss = self.loss_function(elem.view(1,-1,args.num_features), predicted.view(1,-1,args.num_features))
                    loss.backward()
                    optimizer.step()
                    #asd += 1
            print("Sample %d :: Loss %f :: Model idx %d" % (batch, loss.item(), int(label)))
            #print("Sanity %d" % asd)
            
    def eval_loop(self, evalDataLoader:D.DataLoader):
        for label, (model, _) in self.classifiers.items():
            model.eval()
            model.to("cpu")
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
                    #print("Correct")
                #else:
                #    print("False")
            accuracy = correct_labels / len(evalDataLoader)
            print("Eval accuracy: %f" % accuracy)

        
    def loss_function(self, target, predicted):
        #fn = nn.BCELoss()
        #fn = nn.BCEWithLogitsLoss()
        fn = nn.MSELoss()
        #y, x = predicted // max(torch.argmax(predicted)), target // max(torch.argmax(target))
        return fn(predicted, target)


aann_classifier = AANN_Classifer()

def main():
    print("main?")
    for label in fb_dataset.labels:
        aann_classifier.add_new_model(AANN(args.num_features), label)
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