from frogs_data import Frogs_Dataset as FrogsSet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

def generate_frogs_from_dataframe(from_file, class_label="Species"):
    """ 
    Generate pandas.Dataframe from existing dataframe or read a .csv into a dataframe.
    Drop useless columns and add new such as labels to index.
    Returns numpy array and label strings list. Inserted label index can be used to retrieve
    class name corresponding to that label index.
    """

    if isinstance(from_file, pd.DataFrame):
        frogs = from_file.copy()
    else:
        frogs = pd.read_csv(from_file)
    
    frogs.drop(columns=["Family","Genus","RecordID"], inplace=True, errors='ignore')

    species_names = list(frogs[class_label].unique())
    temp_labels = [species_names.index(frog) for frog in list(frogs[class_label])]
    frogs["labels"] = temp_labels
    frogs_np = frogs.to_numpy()
    return frogs_np, species_names

def get_features_labels_as_numpy(frogs_np):
    """
    Exctract only features and labels you are about.
    Return objecs should be used as inputs to train/test splitter
    """

    frogs_features = frogs_np[:, 0:22].astype(np.float32)
    frogs_labels = frogs_np[:, 23]

    return frogs_features, frogs_labels

def generate_datasets(from_file, split=0.9, class_label="Species", shuffle=True, stratified=False, seed=None):
    """ Generates tranining and eval datasets from source file(s) """
    
    frogs_np, species_names = generate_frogs_from_dataframe(from_file, class_label=class_label)
    
    test_frac = 1 - split

    #should be a numpy array of 
    #Intrested in data from indices:
    #0:22, 24, 26 => since dropped columns, then 0:22, 23 for labels
    frogs_features, frogs_labels = get_features_labels_as_numpy(frogs_np)
    
    if stratified:
        #Need to make sure train_set and eval_set have same fraction of classes
        #needed if dataset has unbalanced number of instances of classes
        frogs_train, frogs_eval, labels_train, labels_eval = \
            train_test_split(frogs_features, frogs_labels, test_size=test_frac, stratify=frogs_labels)

    else:
        frogs_train, frogs_eval, labels_train, labels_eval = \
            train_test_split(frogs_features, frogs_labels, test_size=test_frac, stratify=None)
        
    

    return FrogsSet(frogs_train, labels_train), FrogsSet(frogs_eval, labels_eval), species_names

def generate_kfolds_datasets(from_file, kfolds=10, class_label="Species", shuffle=True, stratified=None, seed=None):
    """ Returns an iterable which yields folds """
    
    frogs_np, species_names = generate_frogs_from_dataframe(from_file, class_label=class_label)

    frogs_features, frogs_labels = get_features_labels_as_numpy(frogs_np)  
    
    foldG = StratifiedKFold(n_splits=kfolds, shuffle=shuffle)

    for train_index, eval_index in foldG.split(frogs_features, frogs_np[:, 22]):
        yield FrogsSet(frogs_features[train_index], frogs_labels[train_index]), FrogsSet(frogs_features[eval_index], frogs_labels[eval_index]), species_names
    
class ConfusionMatrix:
    def __init__(self, labels=None):
        self.labels = labels #This should be a [str, str, ..] containing dataset labels as strings.
        self.columns_len = len(labels)
        self.rows_len = len(labels)
        self._shape = (self.rows_len, self.columns_len)
        self._matrix = np.zeros(self._shape, dtype=np.int)

        self.results = pd.DataFrame(self._matrix, index=self.labels, columns=self.labels, dtype=np.int)
    @property
    def shape(self):
        ''' Returns shape of confusion matrix as tuple of (rows,columns) '''
        return self._shape

    @property
    def matrix(self):
        return self.results.to_numpy()

    def add_batch(self, predicted_labels=None, target_labels=None):
        ''' 
        Inputs are numpy arrays of shape (batch_size, 1) where dimension 1 contains index of label for frogs dataset.
        That index is used to retrive species name from species list return by generate_* functions in frogs_util.
        Since pytorch method tensor => numpy returns ndarray with references to same memlocations, do not modify those arrays
        '''
        try:
            if predicted_labels.shape != target_labels.shape:
                raise ValueError("Predicted and target shapes should be same")
        except:
            raise TypeError
        
        for r, c in zip(predicted_labels.flat, target_labels.flat):
            self.results.at[self.labels[r], self.labels[c]] += 1

    def __add__(self, other):
        ''' Overload + operator for easier summing of confusion matrixes! Beware this operation mutates 1st class! '''
        if not isinstance(other, ConfusionMatrix):
            raise TypeError("Add only with same type")
        self.results = self.results.add(other.results)
        return self

    def __repr__(self):
        ''' 
        Yes, __repr__ should reprisent actual object and aid in reconstruction. 
        However this is not required, and if __str__ is not defined => __repr__ is called
        '''
        return self.results.to_string()

# Create timer decorators which prints out statistics?