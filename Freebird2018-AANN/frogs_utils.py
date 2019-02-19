from frogs_data import frogs_dataset as FrogsSet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_datasets(from_file, split=0.9, class_label="Species", stratified=False, seed=None):
    """ Generates tranining and eval datasets from source file(s) """
    
    if type(from_file) is not pd.DataFrame:    
        frogs = pd.read_csv(from_file)
    else:
        frogs = from_file.copy()
    
    frogs.drop(columns=["Family","Genus","RecordID"], inplace=True, errors='ignore')
    
    if frogs["labels"].empty:
        frogs["labels"] = pd.get_dummies(self.frogs[class_label]).values.tolist()

    species_names = list(frogs.Species.unique())
    frogs_np = frogs.to_numpy()

    len_trainingset = int(len(frogs_np) * split)
    test_frac = 1 - split

    #should be a numpy array of 
    #Intrested in data from indices:
    #0:22, 24, 26 => since dropped columns, then 0:22, 23 for labels
    frogs_features = frogs_np[:, 0:22]
    frogs_labels = frogs_np[:, 23]

    if stratified:
        #Need to make sure train_set and eval_set have same fraction of classes
        #needed if dataset has unbalanced number of instances of classes
        frogs_train, frogs_eval, labels_train, labels_eval = \
            train_test_split(frogs_features, frogs_labels, test_size=test_frac, stratify=frogs_labels)

    else:
        frogs_train, frogs_eval, labels_train, labels_eval = \
            train_test_split(frogs_features, frogs_labels, test_size=test_frac)
        
    

    return FrogsSet(frogs_train, labels_train), FrogsSet(frogs_eval, labels_eval), species_names