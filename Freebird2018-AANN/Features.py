import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc


class FeaturesMFCC:
    """
    Class containing MFCC variables and methods to extract features from wav file
    """
    def __init__(self, winlengthSec=0.02, winStepSec=0.01, numFeatures=13):
        """
        Args:
            winLengthSec (float): Window length for FTT
            winStepSec (float): Window step for FFT
            numFeatures (int): Number of MFCC features to calculate
        """
        self.winLengthSec = winlengthSec
        self.winStepSec = winStepSec
        self.numFeatures = numFeatures
        self.sample_rate_default = 44100
    def get_features(self, filePath):
        try:
            #assert os.path.exists(filePath)
            (samplerate, signal) = wav.read(filePath)
            #normalize along first column in array of Nx1
            norm_signal = signal / np.max(signal, axis=0)
            #print("Signal shape: %s" % str(norm_signal.shape)) #(441000,)
            #features shape is NumFrames x Numpcep, each row per featurevector (numframes)
            
        except ValueError:
            print("Incomplete wav chunk in file: %s. Substituting random.." % filePath)
            norm_signal = np.random.uniform(low=0.0, high=1.0, size=(441000,))
            samplerate = self.sample_rate_default
        features = mfcc(norm_signal, samplerate, self.winLengthSec, self.winStepSec, self.numFeatures)
        #print("Features shape for 10s sample: ")
        #print(features.shape)
        return features
    def segment_signal(self):
        pass