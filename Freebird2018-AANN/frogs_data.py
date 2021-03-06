from torch.utils.data import Dataset
import torch
class Frogs_Dataset(Dataset):
    """
    Dataset holding frogs features and labels. 
    Features are numpy array with columns corresponding to MFCC features.
    Labels are single int representing the index position in species list.
    Transform function is applied to frogs data before returning next item in iter() block,
    if None is provided => default function to_tensor is used to create tensors.
    User is required to handle tensor creation operation when custom transform_fn is provided.

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
    def __init__(self, frogs=None, labels=None, transform_fn=None):
        
        self.frogs = frogs
        self.labels = labels
        self.transform = transform_fn
        #Pytorch doesn't expect numpy arrays but tensors, therefore we convert
        if not self.transform:
            self.transform = self.to_tensor

    def __len__(self):
        return len(self.frogs)

    def __getitem__(self, index):
        return self.transform(self.frogs[index]), self.labels[index]


    def to_tensor(self, input):
        if isinstance(input,list):
            return torch.as_tensor(input, dtype=torch.long)
        return torch.from_numpy(input)