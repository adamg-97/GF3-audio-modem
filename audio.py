import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]


import numpy as np
import sys
import scipy.io.wavfile
from scipy.signal import chirp, spectrogram, convolve
from IPython.display import Audio
import sounddevice as sd
sd.default.channels = 1


# Record audio using:
# import sounddevice as sp

# myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)

# Play and record simultaneously:
# myrecording = sd.playrec(myarray, fs, channels=1)

# Playback:
# sd.play(myarray, fs)


# Read/Write audio to file using
# fs, signal = scipy.io.wavfile.read('filename')
# scipy.io.wavfile.write(r"filename", fs, myarray)

def play_record(data, fs, padding_before=0.5, padding_after=0):

    data_padded = np.pad(data, (int(padding_before*fs), int(padding_after*fs)), 'constant', constant_values=0)
    print("Recording...")
    signal = sd.playrec(data_padded, fs)
    sd.wait()
    print("Finished recording")
    return signal[:,0]

def plot_audio(filename):

    # Extract Raw Audio from Wav File
    fs, signal = scipy.io.wavfile.read(filename)

    time = np.linspace(0, len(signal) / fs, num=len(signal))

    y_max_index= np.where(signal == np.amax(signal))
    max_x= time[y_max_index]

    x= time[::100]
    y= signal[::100]

    plt.figure(1)
    plt.title("Audio Signal- Stairs")
    plt.xlabel('Time, s')
    plt.plot(x, y)
    plt.xlim([max_x, max_x+0.5])
    plt.grid()
    plt.savefig('plots/claps/stairs.jpeg')
    plt.show()

    return


def get_peaks(signal, thresh):
    peaks = []
    
    for i in range(len(signal)):
        if(abs(signal[i]) > thresh): peaks.append([signal[i], i])
        
    return(peaks)

#record_audio('recorded.wav')
#play_audio('recorded.wav')
#plot_audio('recorded.wav')



