import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as D
from torchvision import datasets, transforms
from Freebird2018DataSet import Freebird2018DataSet
from Features import FeaturesMFCC

parser = argparse.ArgumentParser(description='FreeBird2018_AANN_exp')
parser.add_argument('--batch-size', type=int, default=1, help='Input batch size, default 4')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='Number of epoch to train')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Use cpu as device instead of CUDA')
parser.add_argument('--file-dir', default='F:\\Downloads\\ff1010bird_wav', help='Directory of metadata and ./wav folder')
parser.add_argument('--num-features', type=int, default=8, metavar='N', help='Number of MFCC Features')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
#device = "cpu"
#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
args.file_dir = os.path.join('F:\\Downloads\\', 'MNIST')

#feature_method = FeaturesMFCC(numFeatures=args.num_features)
#n_feat = args.num_features
#fb_dataset = Freebird2018DataSet(args.file_dir, feature_method)

#dataset_len = len(fb_dataset)
#train_len = int(0.8*dataset_len)
#eval_len = dataset_len - train_len

#train_set, eval_set = D.random_split(fb_dataset, lengths=[train_len, eval_len])

#train_loader = D.DataLoader(train_set, batch_size = args.batch_size, **kwargs)

#eval_loader = D.DataLoader(eval_set, batch_size = args.batch_size, **kwargs)

#print('Trainset has: ' + str(len(train_set)) + '. Evalset has: ' + str(len(eval_set)))
print("Dataset init")
train_loader = D.DataLoader(
    datasets.MNIST(args.file_dir, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
eval_loader = D.DataLoader(
    datasets.MNIST(args.file_dir, train=False, transform=transforms.ToTensor()),
batch_size=args.batch_size, shuffle=True)

print("Downloaded?")

# Model AANN as a class here
class AANN(nn.Module):
    def __init__(self):
        super(AANN, self).__init__()
        self.nfeat = 1
        
        self.encode = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU(True),
            nn.Linear(400, 20)
            )

        self.decode = nn.Sequential(
            nn.Linear(20,400),
            nn.ReLU(True),
            nn.Linear(400, 28*28),
            nn.Tanh()
            )

    #define forward pass with ^ layers, should return output of model
    def forward(self, x):
        #x.hape  = batch_nums x channels x width x height |1x1x28x28
        x = x.view(-1, 28*28)
        
        x = self.encode(x)
        x = self.decode(x)
        #x = x.view(1,-1, self.nfeat)
        return x


#AANN Classifier will be based on N-number of AANN's where N is number classes to classify. Thus we need a container class.
#Perhaps it should have a dict with [label]: aann<object> structure
class AANN_Classifer():
    def __init__(self, *args, **kwargs):
        self.classifiers = {}
        
    def add_new_model(self, model:AANN, label):
        #lr = 1e-3 <- copypaste since autocomplite is ass
        self.classifiers[label] = (model, optim.Adam(model.parameters()))

    def classify(self, input):
        # should run input over all classifiers and classify it by highest score, 
        # return dict-key (used as label). ex: return '1' if input seems to have bird in it.
        loss_f = self.loss_function
        #predicted_outputs = {label:loss_f(input.view(1,-1,args.num_features), model(input).view(1,-1,args.num_features)).item() for label, (model, _) in self.classifiers.items()}
        predicted_outputs = {label:loss_f(input, model(input)).item() for label, (model, _) in self.classifiers.items()}
        #predicted_label, _ = max(predicted_outputs, key=lambda x: x[1])
        predicted_label = -1
        if predicted_outputs.items():
            predicted_label = min(predicted_outputs, key=lambda k:predicted_outputs[k])
        return int(predicted_label)
    
    def training_loop(self, trainingDataLoader:D.DataLoader):
        # based on label given as training data -> train corresponding aann from classifiers-dict.
        for label, (model, _) in self.classifiers.items():
            model.train()
            #model.to(device)
        train_loss = 0
        for batch, (inputs, labels) in enumerate(trainingDataLoader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            (model, optimizer) = self.classifiers[labels.item()]
            model.to(device)
            optimizer.zero_grad()
            pred_y = model(inputs)
            #print(torch.max(inputs))
            loss = self.loss_function(inputs, pred_y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            #print(labels)

            
            if batch % 500 == 0:
                print("Sample %d :: Loss %f :: Model idx %d" % (batch, loss.item(), int(labels.item())))
                print("Progress: %d out of %d" % (batch, len(trainingDataLoader)))
        print("Train loss total: %d" % train_loss)
            
    def eval_loop(self, evalDataLoader:D.DataLoader):
        for label, (model, _) in self.classifiers.items():
            model.eval()
            model.to("cpu")
        accuracy = 0.0
        correct_labels = 0

        for batch, (inputs, labels) in enumerate(evalDataLoader):
            
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            if self.classify(inputs) == labels.item():
                correct_labels += 1

            if batch > 0 and batch % 500 == 0:
                print(correct_labels/batch)

        accuracy = correct_labels / len(evalDataLoader)
        print("Correct: %f pcent" % accuracy)

    def loss_function(self, target, predicted):
        #fn = nn.BCELoss()
        fn = nn.BCEWithLogitsLoss()
        #fn = nn.MSELoss()
        #y, x = predicted // max(torch.argmax(predicted)), target // max(torch.argmax(target))
        return fn(predicted, target.view(-1, 28*28))


aann_classifier = AANN_Classifer()

def main():
    print("main?")
    for label in range(0,10):
        aann_classifier.add_new_model(AANN(), label)
    for epoch in range(1, args.epochs + 1):
        try:

            print("Training loop for epoch %d" % epoch)
            aann_classifier.training_loop(train_loader)
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            raise
    with torch.no_grad():
        print("Evaluation loop")
        aann_classifier.eval_loop(eval_loader)

if __name__ == "__main__":
    main()