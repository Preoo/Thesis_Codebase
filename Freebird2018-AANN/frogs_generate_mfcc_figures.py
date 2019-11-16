import numpy
import scipy.io.wavfile
import scipy.signal
from scipy.fftpack import dct
from python_speech_features import mfcc, fbank, get_filterbanks
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

# Use Data/bird_sample.wav to mock how data is processed during MFCC-feature extraction process.
# Good resources: 
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html and http://wellesleynlp.github.io/spring16/speechAnalysis/index.html

def waveplot(sample_rate, wave):
    """Visualize (a segment of) a wave file. Adapted from resource[2]"""
    # maxf = maximum number of frames
    frames = scipy.arange(wave.size)   # x-axis

    plt.plot(frames, wave)
    plt.xlabel('Näyte')
    plt.ylabel('Amplitudi')
    plt.show()

def windowplot(window, frame_size):

    plt.plot(window(frame_size))
    plt.xlabel('Näyte')
    plt.ylabel('Skaalausarvo')
    plt.show()

def rfft_bins_2_freq(sample_rate, NFFT):
    return numpy.fft.rfftfreq(NFFT, d=1/sample_rate)

def fftplot(sample_rate, wave, fft_bins_2_freq):

    mag_frames = numpy.absolute(numpy.fft.rfft(wave, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    plt.plot(fft_bins_2_freq, pow_frames) #pow_frames has size: 257 == NFFT//2 + 1
    plt.ylabel('Teho')
    plt.xlabel('Taajuus (Hz)')
    plt.show()
    

def spectrogram(sample_rate, wave, *args):

    spec, freq, *others = plt.specgram(wave, NFFT=NFFT, Fs=sample_rate, noverlap=sample_rate*frame_stride)
    plt.xlabel('Aika (s)')
    plt.ylabel('Taajuus (Hz)')
    plt.show()

def filterbanks(sample_rate, nfilt, nfft, fft_bins_2_freq):
    
    fb = get_filterbanks(nfilt, nfft, sample_rate)
    
    for filter in fb:
        plt.plot(fft_bins_2_freq, filter)

    plt.ylabel('Skaalausarvo')
    plt.xlabel('Taajuus (Hz)')
    plt.show()

def mel_fbank(sample_rate, wave):

    fbanks, total_energies = fbank(wave, samplerate=sample_rate, winlen=frame_size, winstep=frame_stride, nfilt=mel_filters, nfft=NFFT, winfunc=window_func, preemph=0)

    plt.stem(fbanks[0])
    plt.ylabel('Teho')
    plt.xlabel('Suodatin')
    plt.show()

def mel_spectrofram(sample_rate, wave, *args):

    mfccs = mfcc(wave, samplerate=sample_rate, winlen=frame_size, winstep=frame_stride, numcep=int(mel_filters/2), nfilt=mel_filters, nfft=NFFT, winfunc=window_func, appendEnergy=True, preemph=0, ceplifter=0) #Default: ceplifter=22 
    
    plt.stem(mfccs[0])
    plt.xlabel('MFC-kertoimet')
    plt.show()

    plt.imshow(mfccs)
    plt.xlabel('MFC-kertoimet')
    plt.xticks([x for x in range(0, int(mel_filters/2), 1)])
    plt.yticks([])
    plt.ylabel('Kehykset')
    plt.show()

#constants
start = 0.25
stop = 0.3
pre_emphasis=0.97
frame_size = 0.025
frame_stride = 0.01
NFFT = 512
mel_filters = 44
window_func = numpy.hanning



#setup
sample_rate, wave = scipy.io.wavfile.read('Data/bird_sample.wav')  # File assumed to be in the same directory
wave = wave[int(start * sample_rate) : int(stop * sample_rate)]  # Take subset of waveform
fft_bins_2_freq = rfft_bins_2_freq(sample_rate, NFFT) # Generate mapping for FFT Bins <=> Frequency

#preemphasis 
if pre_emphasis > 0:
    emphasized_signal = numpy.append(wave[0], wave[1:] - pre_emphasis * wave[:-1])
    
    plt.plot(wave, label='Signaali')
    plt.plot(emphasized_signal, label='Esikorostettu signaali')
    plt.legend()
    plt.xlabel('Näyte')
    plt.ylabel('Amplitudi')
    plt.show()

    wave = emphasized_signal

# Generate plots, requires user to save them.
waveplot(sample_rate, wave)
windowplot(window_func, frame_size*sample_rate)
fftplot(sample_rate, wave, fft_bins_2_freq)
spectrogram(sample_rate, wave, fft_bins_2_freq)
filterbanks(sample_rate, mel_filters, NFFT, fft_bins_2_freq)
mel_fbank(sample_rate, wave)
mel_spectrofram(sample_rate, wave)