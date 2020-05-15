import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.wavfile
import math as m

from scipy import signal
#from IPython.display import Audio
#from audio import *



class Transmitter:
    def __init__(self, K = 64, cp = 16, modulation = "QAM", P = 8):
        # Input parameters
        self.K = K                   # Number of OFDM subcarriers i.e. DFT size
        self.cp = cp                 # Length of cyclic prefix, prepended to the payload before the circular convolution
        self.modulation = modulation # Modulation method used (QPSK, QAM or 16QAM)
        self.P = P                   # Number of pilot carriers per block
        self.pre = K // 4            # Length of the preamble used in the Schmidl & Cox Synchronization method for OFDM
        self.signal = None
        self.preamble = None
    
        # Other parameters
        self.pilot_value = -5 - 5j   # Standard value that pilot transmits (used for channel estimation at the Rx)
        self.none_value = 5 + 5j     # Standard value used for non-coding symbols 
        if self.cp is None:          # By default, the cyclig prefix is 25 % of the number of subcarriers
            self.cp = K // 4
        
        
        # Carriers mapping
        self.all_carriers = np.arange(self.K)                                   # Indicies of all subcarriers [0... K-1]
        try: 
            self.pilot_carriers = self.all_carriers[::self.K//self.P]           # Pilot carriers every K / Pth carrier
        except: 
            self.pilot_carriers = np.array([])                                  # Exception for no pilot carriers
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers)  # The remaining carriers are set to data carriers (default)
        
        
        # Setting mapping tables and modulation indices for different modulations
        if(self.modulation == "QPSK"):
            self.mapping_table = { 
                (0,0) : (1+1j)/np.sqrt(2),
                (1,0) : (1-1j)/np.sqrt(2),
                (1,1) : (-1-1j)/np.sqrt(2),
                (0,1) : (-1+1j)/np.sqrt(2),
            }
            self.mu = 2 # Number of bits per symbol
        elif(self.modulation == "QAM"):
            self.mapping_table = { 
                (0,0) : (1+0j),
                (0,1) : (0+1j),
                (1,1) : (-1-0j),
                (1,0) : (0-1j),
            }
            self.mu = 2 # Number of bits per symbol
        elif(self.modulation == "16QAM"):
            self.mapping_table = {
                (0,0,0,0) : -3-3j,
                (0,0,0,1) : -3-1j,
                (0,0,1,0) : -3+3j,
                (0,0,1,1) : -3+1j,
                (0,1,0,0) : -1-3j,
                (0,1,0,1) : -1-1j,
                (0,1,1,0) : -1+3j,
                (0,1,1,1) : -1+1j,
                (1,0,0,0) :  3-3j,
                (1,0,0,1) :  3-1j,
                (1,0,1,0) :  3+3j,
                (1,0,1,1) :  3+1j,
                (1,1,0,0) :  1-3j,
                (1,1,0,1) :  1-1j,
                (1,1,1,0) :  1+3j,
                (1,1,1,1) :  1+1j
            }
            self.mu = 2 # Number of bits per symbol     
        else:
            raise ValueError("Invalid Modulation Type")
        
        self.bits_per_symbol = len(self.data_carriers) * self.mu # Bits per OFDM symbol = number of carriers x modulation index


    # Shapes serial bits into parallel stream for OFDM
    def SP(self, bits):
        return bits.reshape((len(self.data_carriers), self.mu))
    
    
    # Maps the bits to symbols
    def map(self, bits):
        symbols = np.zeros((len(bits)), dtype = np.complex)
        i = 0
        if(type(bits) != np.ndarray): 
            raise ValueError("Bits must be numpy array")
        for b in bits:
            indnone = 0
            while indnone < len(b):
                if b[indnone] == -1:
                    break
                indnone += 1
        
            
            if indnone == 0:
                symbols[i] = self.none_value
            else:
                symbols[i] = self.mapping_table[tuple(b[:indnone])]

            i += 1
        return symbols
    
    
    # Allocates symbols and pilots to OFDM symbol -- needs adjusting for multiple OFDM symbols
    def OFDM_symbol(self, payload):
        symbol = np.zeros(self.K, dtype=complex)               # Symbols in the overall K subcarriers
        try: 
            symbol[self.pilot_carriers] = self.pilot_value     # Allocate pilot subcarriers
        except:
            symbol[self.data_carriers] = payload               # Allocate data carriers
        
        return symbol
    

    # Add the cyclic prefix -- doesnt work yet
    def add_cp(self, time_data):
        cyclic_prefix = time_data[-self.cp:]          # take the last cp samples
        return np.hstack([cyclic_prefix, time_data])    # add them to the beginning
    
    
    # Displays the constellation of the modulation method
    def plot_constellation(self):
        for k, v in self.mapping_table:
            x, y = v.real, v.imag
            plt.plot(x, y, 'bo')
            plt.text(x, y + 0.2, "".join(str(c) for c in k), ha='center')
        plt.show()
          

    # Prints out key OFDM attributes
    def __repr__(self):
        return 'Number of Sub Carriers: {:.0f} \nCyclic prefix length: {:.0f} \nModulation method: {}'.format(self.K, self.cp, self.modulation)


    # Generates a chirp signal
    def chirp(self, flow, fhigh, T):
        t = np.linspace(0, T, T*fs)
        w = chirp(t, f0=f0, f1=f1, t1=T, method='linear')
        
        
    # Allocates symbols and pilots to OFDM symbol -- needs adjusting for multiple OFDM symbols
    def OFDM_symbol(self, payload):
        symbol = np.zeros(self.K, dtype=complex) # overall K subcarriers
        symbol[self.pilot_carriers] = self.pilot_value     # Allocate pilot subcarriers
        symbol[self.data_carriers] = payload          # Allocate data carriers
        
        return symbol # Return the carriers allocation
    
    
    # Inverse Fourier Transform
    def IFFT(self, data):
        return np.fft.ifft(data)
        
        
    # The entire workflow for the transmitting part
    def transmit(self, signal):
        self.signal = signal
        
        print("\n" * 2 + "-" * 42 + "\n2. TRANSMISSION\n" + "-" * 42 + "\n")
        print("Number of bits to transmit : " + str(np.sum([np.sum([1 if e != -1 else 0 for e in signal[i]]) for i in range(len(signal))])))
        print("First 10 bits of the input signal : " + "".join(str(signal[0, i]) for i in range(10)))
        bits_parallel = self.SP(signal[0]) # We parallelize the bits in order to send one constellation symbol per data carrier
                
        print("\nNumber of bits per constellation symbol : " + str(bits_parallel.shape[1]))
        print("Number of constellation symbols to transmit : " + str(bits_parallel.shape[0]))

        constellation = self.map(bits_parallel) # We transform the mu-length bits sequences into their corresponding constellation symbol
        print("\nFirst 10 parallelized bit sequences and their corresponding symbol : ")
        for i in range(10):
            print(" ".join(str(x) for x in list(bits_parallel[i, :])) + " ---(modulation)--> " + str(constellation[i]))
            
        OFDM_data = self.OFDM_symbol(constellation) # Assigning the symbols to carriers
        print(constellation)
        print("\nNumber of OFDM carriers : " + str(len(OFDM_data)))
        print("Number of OFDM pilot carriers : " + str(len(self.pilot_carriers)))
        print("Number of OFDM data carriers : " + str(len(self.data_carriers)))
        print("\nFirst 10 OFDM carriers and their assigned symbol :")
        for i in range(10):
            print("Carrier {0} --> {1}".format(str(i), OFDM_data[i]))
        
        for i in range(int(self.K / self.P)):
            if i in self.pilot_carriers:
                plt.plot(i, 1, 'bo')
                plt.text(i, 1.2, str(np.around(OFDM_data[i], 2)), ha="center")
            else:
                plt.plot(i, 0, 'ro')
                plt.text(i, 0.2, str(np.around(OFDM_data[i], 2)), ha="center")
        plt.show()
        
        OFDM_time = self.IFFT(OFDM_data) # Inverse Fourier transform of signal data (final step of modulation)
        OFDM_with_cp = self.add_cp(OFDM_time) # Adding the cyclic prefix to the sequence
        
        OFDM_transmit = OFDM_with_cp
       
        for i in range(1, signal.shape[0]):
            bits_parallel = self.SP(signal[i]) # We parallelize the bits in order to send one constellation symbol per data carrier

            constellation = self.map(bits_parallel) # We transform the mu-length bits sequences into their corresponding constellation symbol
            OFDM_data = self.OFDM_symbol(constellation) # Assigning the symbols to carriers    
            
            OFDM_time = self.IFFT(OFDM_data) # Inverse Fourier transform of signal data (final step of modulation)
            OFDM_with_cp = self.add_cp(OFDM_time) # Adding the cyclic prefix to the sequence
            
            OFDM_transmit = np.hstack([OFDM_transmit, OFDM_with_cp])
        return OFDM_transmit

    
        
        
    
    
    
class Receiver(Transmitter): # Class inheritance : all the attributes of Transmitter will be attributes of Receiver
    # De-maps symbols to bits using min distance
    def demap(self, symbols):
        if(type(symbols) != np.ndarray): 
            raise ValueError("Symbols must be numpy array")
             
        demapping_table = {v : k for k, v in self.mapping_table.items()}
        constellation = np.array(list(demapping_table.keys()), dtype=np.complex)
        dists = []
        indices = []
        
        # Array of possible constellation points
        for s in symbols:
            d = abs(np.ones_like(constellation) * s - constellation)
            dists.append(d)
            indices.append(d.argmin())
        
        hardDecision = constellation[indices]
        
        # Transform constellation points into bit groups
        return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision
        
    
    # Remove the cyclic prefix
    def remove_cp(self, signal):
        return signal[self.cp : self.cp + self.K]   # Only taking indicies of the data we want
 
    
    # Equalise the received OFDM signal using the estimated channel coefficients
    def equalise(self, data):
        h_pilots = data[self.pilot_carriers] / self.pilot_value
        h_real_pilots, h_imag_pilots = np.real(h_pilots), np.imag(h_pilots)
        
        h_real_interp = scipy.interpolate.barycentric_interpolate(self.pilot_carriers, h_real_pilots, self.data_carriers)
        h_imag_interp = scipy.interpolate.barycentric_interpolate(self.pilot_carriers, h_imag_pilots, self.data_carriers)
        
        h_interp = h_real_interp + 1j * h_imag_interp

        return data[Tx.data_carriers] / h_interp
    
    
    # Fast Fourier Transform
    def FFT(self, data):
        return np.fft.fft(data)
    
    
    # Shapes parallel bits back into serial stream
    def PS(self, bits):
        return bits.reshape((-1,))
    
    
    # The entire workflow for the receiving part
    def receive(self, signal):
        ofdm_symbols = int(len(signal) / (Tx.cp + len(Tx.data_carriers)))
        bits_serial = np.array([])
        
        print("\n" * 2 + "-" * 42 + "\n4. RECEPTION\n" + "-" * 42 + "\n")
        
        OFDM_rx = signal[:Tx.K + Tx.cp] # We select the channel output range corresponding to the ith OFDM symbol
        OFDM_no_cp = self.remove_cp(OFDM_rx) # Removing the cyclic prefix from the received signal
        OFDM_demod = self.FFT(OFDM_no_cp) # Unmodulating the received signal (we obtain a sequence of symbols spanning the constellation)
                
        OFDM_data = self.equalise(OFDM_demod) # Performing channel estimation on the received signal
        print("\nFirst 10 estimated OFDM carriers and their assigned symbol :")
        for i in range(10):
            print("Carrier {0} --> {1}".format(str(i), np.around(OFDM_data[i], 2)))
           
        bits_parallel, descision_symbols = self.demap(OFDM_data) # Demapping the constellation symbols to bits using minimum distance rule
        print("\nFirst 10 data-coding symbols and their corresponding parallelized bit sequences : ")
        for i in range(10):
            print(str(np.around(OFDM_data[i], 2)) + " ---(demodulation)--> " + "".join([str(x) for x in bits_parallel[i]]))

        bits_serial = self.PS(bits_parallel)
        print("\nFirst 20 bits of the output signal : " + "".join(str(Tx.signal[0, i]) for i in range(20)))
        print("First 20 bits of the output signal : " + "".join(str(bits_serial[i]) for i in range(20)))
        
        for i in range(1, ofdm_symbols):
            OFDM_rx = signal[i * (Tx.K + Tx.cp):(i + 1) * (Tx.K + Tx.cp)] # We select the channel output range corresponding to the ith OFDM symbol
            OFDM_no_cp = self.remove_cp(OFDM_rx) # Removing the cyclic prefix from the received signal
            OFDM_demod = self.FFT(OFDM_no_cp) # Unmodulating the received signal (we obtain a sequence of symbols spanning the constellation)
                    
            OFDM_data = self.equalise(OFDM_demod) # Performing channel estimation on the received signal
               
            bits_parallel, descision_symbols = self.demap(OFDM_data) # Demapping the constellation symbols to bits using minimum distance rule
           
            bits_serial = np.append(bits_serial, self.PS(bits_parallel)) # We add the binary sequence corresponding to the current OFDM symbol to the full binary signal
           
        bitDiff = abs(bits_serial - np.hstack(Tx.signal)) # Simple distance between the original signal and the recovered signal
        bitErrorRate = np.sum(bitDiff) / len(bitDiff)
        print("\nTotal bit error rate : " + str(np.around(bitErrorRate, 4)))

        return bits_serial
    
    
    
class Channel(Transmitter):
    def __init__(self, snr = 20):
        self.snr = snr
        self.responseType = None
        self.response = None
        
    
    # Defines 
    def setResponse(self, responseType = "Convolution", response = np.array([1, 0.3, 0.2, 0.1j + 0.1, 0.02])):
        if responseType == "Convolution":
            self.response = response
        elif responseType == "H":
            self.response = pd.read_csv(response, header=None, delimiter=r"\s+").values[0]

            H = np.fft.fft(channel_data, Tx.K)
            plt.plot(wc.all_carriers, abs(H))
            plt.title("Channel Frequency Response")
            plt.xlabel("Subcarrier Index")
            plt.ylabel("|H(f)|")
            plt.show()
            
        self.responseType = responseType
        
        
    # Applies the channel response to the input signal and yields the received signal
    def process(self, signal):
        print("\n" * 2 + "-" * 42 + "\n2. CHANNEL\n" + "-" * 42 + "\n")
        
        if self.responseType == "Convolution":
            output_clear = np.convolve(signal, self.response)
        elif self.responseType == "H":
            pass
        
        print("Input spectral density (before channel transformation) :")
        
        powerTx = np.mean(abs(output_clear ** 2)) # Simple definition of the power of a signal
        powerNoise = powerTx * 10 ** (- self.snr / 10) # We compute the noise power such that the resulting SNR matches the value specified
        print("\nSignal to noise ratio : " + str(self.snr))
        print("Signal power : " + str(powerTx))
        print("Noise power : " + str(powerNoise))
        
        # The real and imaginary parts of the noise process can be defined independently as two AWGN whose power is 1/sqrt(2) a fraction of the original complex noise power
        noiseReal = np.sqrt(powerNoise / 2) * np.random.randn(output_clear.shape[0])
        noiseImag = 1j * np.sqrt(powerNoise / 2) * np.random.randn(output_clear.shape[0])
        noise = noiseReal + noiseImag
        output_noise = output_clear + noiseReal + noiseImag # AWGN model : the noise is simply added to the clear source
        
        return output_noise
    
    
    
class Signal:
    def __init__(self, fs = 44100, duration = 5):
        self.fs = fs # Sampling frequency
    
    
    # Generates a binary signal from file
    def fromFile(self, file): 
        print("\n" * 2 + "-" * 42 + "\n1. SIGNAL GENERATION\n" + "-" * 42 + "\n")
        
        signal = []
        if ".csv" in file:
            fileContent = pd.read_csv(response, header=None, delimiter=r"\s+").values[0]
        else:
            f = open(file, "r")
            fileContent = np.loadtxt()
        
        n = len(fileContent) # Number of bits to transmit
        ofdm_symbols = m.ceil(n / Tx.bits_per_symbol) # Number of OFDM symbols necessary to encode the data
        print("OFDM symbols generated : " + str(ofdm_symbols))
        
        signal = np.zeros((ofdm_symbols, Tx.bits_per_symbol))
        
        # We add trailing "-1 bits" to the last OFDM_symbol, which signify the absence of signal for these carriers
        for i in range(ofdm_symbols - 1): # We fill in every complete line of bits corresponding to one OFDM symbol
            signal[i] = fileContent[i * Tx.bits_per_symbol : (i + 1) * Tx.bits_per_symbol]
        
        # We fill the last line
        coding_bits = fileContent[i * Tx.bits_per_symbol : n]
        trailing_bits = np.array([-1 for i in range(Tx.bits_per_symbol - (n % Tx.bits_per_symbol))])
        signal[-1] = np.append(coding_bits, trailing_bits)
                    
        return np.array(signal, dtype = int)
    
    
    # Generates a random binary signal
    def random(self, p = 0.5, n = 300, periods = None): 
        # p         : probability of success in a Bernoulli experiment
        # n         : number of bits to transmit
        # periods   : (alternatively) number of fully filled OFDM symbols to transmit
        
        print("\n" * 2 + "-" * 42 + "\n1. SIGNAL GENERATION\n" + "-" * 42 + "\n")
        
        if periods is not None: # If the number of bits to transmit is an exact multiple of the number of bits transmitted per OFDM symbol
            print("Random Bernoulli signal generation : p = " + str(p) + ", n = " + str(Tx.bits_per_symbol * periods) + " bits")
            
            signal = np.zeros((periods, Tx.bits_per_symbol))
            print("OFDM symbols generated : " + str(periods))
            for i in range(periods):
                signal[i] = np.random.binomial(n = 1, p = p, size = Tx.bits_per_symbol)
                
        else: # Otherwise, we add trailing "-1 bits" to the last OFDM_symbol, which signify the absence of signal for these carriers
            print("Random Bernoulli signal generation : p = " + str(p) + ", n = " + str(n) + " bits")
            ofdm_symbols = m.ceil(n / Tx.bits_per_symbol)
            print("OFDM symbols generated : " + str(ofdm_symbols))
            
            signal = np.zeros((ofdm_symbols, Tx.bits_per_symbol))
            
            for i in range(ofdm_symbols - 1): # We fill in every complete line of bits corresponding to one OFDM symbol
                signal[i] = np.random.binomial(n = 1, p = p, size = Tx.bits_per_symbol)
            
            # We fill the last line
            coding_bits = np.random.binomial(n = 1, p = p, size = n % Tx.bits_per_symbol)
            trailing_bits = np.array([-1 for i in range(Tx.bits_per_symbol - (n % Tx.bits_per_symbol))])
            signal[-1] = np.append(coding_bits, trailing_bits)

        return np.array(signal, dtype = int)
        

###############################################################################

    
# Insitialising the three modules of our experiment
Tx = Transmitter()
Ch = Channel()
Rx = Receiver()

# Starting with a random bit source
S = Signal()
signal = S.random()

# Defining a convolutional response for the channel
Ch.setResponse()

signalTx = Tx.transmit(signal) # Transforming the bit sequence into a waveform as an input to the channel
signalTx = Ch.process(signalTx) # Processing the evolution of the waveform into the channel
output = Rx.receive(signalTx) # Interpreting the received waveform and recovering the original bit sequence