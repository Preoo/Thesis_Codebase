from frogs_data import frogs_dataset as FrogsSet
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

def generate_datasets(from_file, frac_folds=0.5, class_label="Species", stratified=False, seed=None):
    """ Generates tranining and eval datasets from source file(s) """
    
    if type(from_file) is not pd.DataFrame:    
        frogs = pd.read_csv(from_file)
    else:
        frogs = from_file
    
    frogs.drop(columns=["Family","Genus","RecordID"], inplace=True, errors='ignore')
    self.frogs_data["labels"] = pd.get_dummies(self.frogs_data.Species).values.tolist()
    frogs_all = frogs.to_numpy()
    #should be a numpy array of 


    if stratified:
        #Need to make sure train_set and eval_set have same fraction of classes
        #needed if dataset has unbalanced number of instances of classes
        pass

    else:
        pass
    

    return FrogsSet(frogs_train, labels_train), FrogsSet(frogs_eval, frogs_train), species_names