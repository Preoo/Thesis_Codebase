import torch #needed since dataset returns features as tensor
import numpy as np
from collections import Counter
from pathlib import Path
from frogs_utils import generate_kfolds_datasets, ConfusionMatrix

class Frogs_kNN:
    def __init__(self, k_nearest=1):
        self.distance_index = []
        self.k = k_nearest

    def get_euclidian_dist(self, x, y):
        #assume we have two tensors
        #p-norm of (x-y), p=2 yeilds euclidian norm
        #return torch.dist(x, y, p=2).item()

        #Using numpy broadcasting
        #https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-numpy.doc.broadcasting
        #x.shape(N,22) y.shape(22)
        w = torch.pow(x-y,2)
        w = torch.sum(w,1)
        w = torch.sqrt(w)
        return w
        
    def fit(self, X, y):
        #X is dataset with neightbours, Y is datapoint e.g. single measurement to classify
        #Returns predicted label from dataset.
        self.distance_index.clear()
        
        #for i, (feature, label) in enumerate(X):
        #    distance = self.get_euclidian_dist(feature, y)
        #    self.distance_index.append((distance, i))
        f, _ = X[:]
        distances = self.get_euclidian_dist(f,y) #should return with shape X.0, 

        #self.distance_index.sort()
        #sorted by distance => just take k from this list
        #k_nearest_distances_idx = [idx for _, idx in self.distance_index[:self.k]]
        _, k_nearest_distances_idx = torch.topk(distances, self.k, largest=False)
        k_nearest_distances_idx = k_nearest_distances_idx[:]
        #get most common, if unclear, resort to first
        most_common = Counter(k_nearest_distances_idx).most_common(1)
        most_common, _ = most_common[0] #Counter.most_common returns a list of tuples(what,count)
        #return label from dataset based on most common index
        _, predicted = X[most_common]
        return predicted

def run():
    n_folds = 2
    frogs_csv = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
    print("===========================")
    print("Running %d-fold eval" % n_folds)

    stats = []
    for ftrain, fevail, species_names in generate_kfolds_datasets(frogs_csv, kfolds=n_folds):
        kNN = Frogs_kNN(k_nearest=1)
        correct_fold = 0
        real = []
        pred = []
        print("Debug 1")

        cm_nfold = ConfusionMatrix(labels=species_names)
        for b, (y, t) in enumerate(fevail):
            if b % 300 == 0:
                print("Debug 2")
            
            y_pred = kNN.fit(ftrain, y)
            if y_pred  == t:
                correct_fold += 1
            real.append(t)
            pred.append(y_pred)
        cm_nfold.add_batch(predicted_labels=np.asarray(pred), target_labels=np.asarray(real))
        acc_nfold = correct_fold / len(fevail)
        stats.append((acc_nfold, cm_nfold))
    #get averaged accuracy
    n_acc = sum([a for a, _ in stats])/float(len(stats))
    print("n-fold mean accuracy: %f" % (n_acc * 100.0))
    cuml_cm = ConfusionMatrix(labels=species_names)
    for _, cm_nfold in stats:
        cuml_cm + cm_nfold
    print("======= %d-fold cumulutive confusion matrix =======" % n_folds)
    print(cuml_cm)

run()