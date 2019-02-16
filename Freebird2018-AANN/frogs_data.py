from torch.utils.data import Dataset

class frogs_dataset(Dataset):

    def __init__(self, frogs=None, labels=None, unittest=False):
        
        # Only species is of intrest to us
        
        #self.frogs_data["labels"] = pd.get_dummies(self.frogs_data.Species).values.tolist()
        #self.generate_dataset(select_set, from_samples, n_folds=n_folds, stratified=stratified_folds)
        self.frogs = frogs
        self.labels = labels
        self.testing =unittest

    def __len__(self):
        return len(self.frogs)

    def __getitem__(self, index):

        #Dummy method to aid with unittests, there must be a better way :P
        if self.testing:
            print(":D")
        else:
            return self.frogs[index], self.labels[index]


    #def test(self):
    #    print("???")
    #    #print(self.file_path) #print correct path as string
    #    #print(self.file_path.exists()) #prints False WTF??
    #    print(self.frogs_data.head())
    #    print(len(self.frogs_data))
    #    print(self.frogs_data.describe())

    #    self.frogs_data["labels"] = pd.get_dummies(self.frogs_data.Species).values.tolist()
    #    #print(pd.get_dummies(self.frogs_df.Species))
    #    print(self.frogs_data.head())
    #    print(self.frogs_data.describe())
        
    #    test = self.frogs_data.iloc[0].take([i for i in range(22)])
    #    print(test)
"""
Frogs dataframe describe
                MFCCs_1      MFCCs_2      MFCCs_3  ...     MFCCs_20     MFCCs_21     MFCCs_22
    count  7195.000000  7195.000000  7195.000000  ...  7195.000000  7195.000000  7195.000000
    mean      0.989885     0.323584     0.311224  ...    -0.053244     0.037313     0.087567
    std       0.069016     0.218653     0.263527  ...     0.094181     0.079470     0.123442
    min      -0.251179    -0.673025    -0.436028  ...    -0.361649    -0.430812    -0.379304
    25%       1.000000     0.165945     0.138445  ...    -0.120971    -0.017620     0.000533
    50%       1.000000     0.302184     0.274626  ...    -0.055180     0.031274     0.105373
    75%       1.000000     0.466566     0.430695  ...     0.001342     0.089619     0.194819
    max       1.000000     1.000000     1.000000  ...     0.467831     0.389797     0.432207
"""