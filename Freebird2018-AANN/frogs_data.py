from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

class frogs_dataloader(Dataset):
    """description of class"""

    def __init__(self, data_file="Frogs_MFCC.csv"):
        #self.project_path = Path(__file__).parent
        #self.file_path = Path(self.project_path) / "Data" / data_file
        self.wtf = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Data\Frogs_MFCCs.csv")
        self.frogs_df = pd.read_csv(self.wtf)

    def __len__(self):
        return len(self.frogs_df)

    def __getitem__(self, index):
        return self.frogs_df.iloc[index]

    def test(self):
        print("???")
        #print(self.file_path) #print correct path as string
        #print(self.file_path.exists()) #prints False WTF??
        print(self.frogs_df.head())
        print(len(self.frogs_df))