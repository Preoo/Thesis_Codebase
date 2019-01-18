from torch.utils.data import Dataset
import csv
import os
class Freebird2018DataSet(Dataset):
    """
    Creates dataset for Freebird2018

    """
    metadataFile = 'ff1010bird_metadata_2018.csv'

    def __init__(self, data_dir, featureMethod):
        """
        Args:
            data_dir (string): Path to root directory with metadata.csv and raw/<audio files> directory
        """
        self.recordings = None
        self.labels = None
        self.dir = data_dir
        self.featureExtractor = featureMethod
        self.recordings, self.labels = self.parseCSV(self.dir)

    def __len__(self):
        
        return len(self.labels)

    def __getitem__(self, index):
        
        return self.featureExtractor.getFeatures(self.getRecording(index, self.dir)), self.labels[index]

    def parseCSV(self, data_dir):
        """
        Assumes that csv has shape id,dataset,label without header. Assuming metadata.csv has same entries as there are wav files.
        """
        metacsvfilepath = os.path.join(data_dir, self.metadataFile)
        #print(metacsvfilepath)
        with open(metacsvfilepath, 'r', newline='') as f:
            reader = csv.reader(f)
            #parsed_recordings = list(reader, delimiter=',')[1:]
            ids = []
            labels = []
            for line in reader:
                # line is a list of ['id', 'dataset', 'label']
                rec_id, label = line[0], line[-1]
                ids.append(rec_id)
                labels.append(label)

            return ids, labels

    def getRecording(self, index, from_dir):
        """ Retrieves pat hto wav file based on recording id """
        return os.path.join(from_dir,'wav', self.recordings[index] + '.wav')
