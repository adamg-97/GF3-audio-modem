import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
import sys


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
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

    # Close and terminate the stream
    stream.close()
    p.terminate()



def record_audio(filename):
    chunk = 1024 
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 3

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

    plt.figure(1)
    plt.title("Audio Signal")
    plt.xlabel('Time, s')
    plt.plot(time, signal)
    plt.show()



record_audio('recorded.wav')
play_audio('recorded.wav')
plot_audio('recorded.wav')
