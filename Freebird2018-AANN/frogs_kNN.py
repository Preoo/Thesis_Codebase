import torch #needed since dataset returns features as tensor
import numpy as np
from collections import Counter
from pathlib import Path
from frogs_utils import generate_kfolds_datasets, ConfusionMatrix

class Frogs_kNN:
    """
    k Nearest Neightbour classifier for Frogs dataset.
    Usage: Initialize this class first to set value for k. Call fit to classify.
    """
    def __init__(self, k_nearest=1, metric='euclidian'):
        self.k = k_nearest
        self.metric = metric

    def get_euclidian_dist(self, x, y):
        #Using numpy broadcasting
        #https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-numpy.doc.broadcasting

        w = torch.pow(x-y,2)
        w = torch.sum(w,1)
        w = torch.sqrt(w)
        return w
    
    def get_cosine_similarity(self, x, y):
        a = torch.sum(torch.mul(x,y) ,1)
        b = torch.sqrt( torch.sum(torch.pow(x,2),1 ) ) * torch.sqrt( torch.sum(torch.pow(y,2) ) )
        return a / b

    def get_criterion_metric(self, metric=''):
        available_metrics = {
            "euclidian" : (self.get_euclidian_dist, False),
            "cosine" : (self.get_cosine_similarity, True)
            }
        try:
            return available_metrics[metric]
        except KeyError:
            print(f'Metric for "{metric}" is not implemented for kNN.')
            raise

    def fit(self, X, y):
        """
        X is dataset with neightbours, Y is datapoint e.g. single measurement to classify
        Returns predicted label from dataset.
        """
        
        #select all features from training Frogs dataset
        x, _ = X[:]
        #distances = self.get_euclidian_dist(x,y) #Set largest=False
        #distances = self.get_cosine_similarity(x,y) #Set largest=True
        metric_func, sort_by = self.get_criterion_metric(self.metric)
        distances = metric_func(x,y)
        # topk returns a tensor with k-first values from other tensor. largest=False inverts order
        _, k_nearest_distances_idx = torch.topk(distances, self.k, largest=sort_by)
        #Retrieve values with python slicing, .item() only works for single values not lists or arrays
        k_nearest_distances_idx = k_nearest_distances_idx[:]
        
        _, labels = X[k_nearest_distances_idx]
        #for numpy arrays, we need most common
        if isinstance(labels, np.ndarray):
            labels = labels.astype(int) #from object to int, makes a copy
            labels, counts = np.unique(labels, return_counts=True)
            return labels[np.argmax(counts)]
        #is single label, just return as is
        return labels

def run():
    n_folds = 10
    frogs_csv = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
    print("===========================")
    print("Running %d-fold eval" % n_folds)

    stats = []
    for ftrain, fevail, species_names in generate_kfolds_datasets(frogs_csv, kfolds=n_folds):
        kNN = Frogs_kNN(k_nearest=1, metric='cosine')
        correct_fold = 0
        real = []
        pred = []
        print(".")

        cm_nfold = ConfusionMatrix(labels=species_names)
        for b, (y, t) in enumerate(fevail):
            #if b % 300 == 0:
            #    print("Debug 2")
            
            y_pred = kNN.fit(ftrain, y)
            if y_pred == t:
                correct_fold += 1
            real.append(t)
            pred.append(y_pred)
        cm_nfold.add_batch(predicted_labels=np.asarray(pred), target_labels=np.asarray(real))
        acc_nfold = correct_fold / len(fevail)
        stats.append((acc_nfold, cm_nfold))
    #get averaged accuracy
    n_acc = sum([a for a, _ in stats])/float(len(stats))
    print("%d-fold mean accuracy: %f" % (n_folds, n_acc * 100.0))
    cuml_cm = ConfusionMatrix(labels=species_names)
    for _, cm_nfold in stats:
        cuml_cm + cm_nfold
    print("======= %d-fold cumulutive confusion matrix =======" % n_folds)
    print(cuml_cm)

if __name__ == "__main__":
    run()