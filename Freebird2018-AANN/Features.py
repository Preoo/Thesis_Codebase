from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

class FeaturesMFCC(object):
    """
    Class containing MFCC variables and methods to extract features from wav file
    """
    def __init__(self, winlengthSec = 0.02, winStepSec = 0.01, numFeatures = 20):
        """
        Args:
            winLengthSec (float): Window length for FTT
            winStepSec (float): Window step for FFT
            numFeatures (int): Number of MFCC features to calculate
        """
        self.winLengthSec = winlengthSec
        self.winStepSec = winStepSec
        self.numFeatures = numFeatures

    def getFeatures(self, filePath):
        #assert os.path.exists(filePath)
        (samplerate, signal) = wav.read(filePath)
        #normalize along first column in array of Nx1
        signal /= np.max(signal,axis=0)
        features = mfcc(signal, samplerate, self.winLengthSec, self.winStepSec, self.numFeatures)
        return features

    def segmentSignal(self):
        pass