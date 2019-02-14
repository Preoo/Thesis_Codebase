from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

class frogs_dataloader(Dataset):
    """description of class"""

    def __init__(self, data_file="Frogs_MFCC.csv"):
        self.project_path = "f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data"
        self.file_path = Path(self.project_path) / data_file
        
        #self.frogs_df = pd.read_csv(self.file_path)
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def test(self):
        print("???")
        print(self.file_path)
        print(Path.exists(self.file_path))
        #print(self.frogs_df.head())