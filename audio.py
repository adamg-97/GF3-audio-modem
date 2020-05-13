import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.io.wavfile
from scipy.signal import chirp, spectrogram, convolve
from IPython.display import Audio


def play_audio(filename):
    chunk = 1024   # Set chunk size of 1024 samples per data frame

    wf = wave.open(filename, 'rb')   # Open the sound file 

    p = pyaudio.PyAudio()   # Create an interface to PortAudio

    # Open a .Stream object to write the WAV file to
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    data = wf.readframes(chunk)    # Read data in chunks

    # Play the sound by writing the audio data to the stream
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    # Close and terminate the stream
    stream.close()
    p.terminate()



def record_audio(filename, time, fs):
    chunk = 1024 
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = fs  # Record at fs samples per second
    seconds = time

    p = pyaudio.PyAudio()

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def plot_audio(filename):

    spf = wave.open(filename, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    fs = spf.getframerate()

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

def get_audio(filename):
    spf = wave.open(filename, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    fs = spf.getframerate()
    
    return signal, fs

def get_peaks(signal, thresh):
    peaks = []
    
    for i in range(len(signal)):
        if(abs(signal[i]) > thresh): peaks.append([signal[i], i])
        
    return(peaks)

#record_audio('recorded.wav')
#play_audio('recorded.wav')
#plot_audio('recorded.wav')



