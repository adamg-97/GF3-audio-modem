# Header file for OFDM, add libraries and functions here
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve
from scipy.io import wavfile
import sounddevice as sd
import pandas as pd
sd.default.channels = 1
from IPython.display import Audio


#########################################
#                 CamG                  #
#########################################

class CamG:
    def __init__(self, ofdm_symbol_size, cp_length, modulation, fs=44100, pilot_sequence=np.array([]), sync_method= "chirp", end_chirp=True):
    
        # Audio params
        self.fs = fs                                # Audio sampling rate
    
        # OFDM params
        self.ofdm_symbol_size = ofdm_symbol_size    # OFDM symbol size
        self.cp_length = cp_length                  # length of cyclic prefix
        self.K = self.ofdm_symbol_size // 2 - 1     # Number of carriers in OFDM symbol i.e. DFT size / 2 - 1
        self.data_carriers = np.arange(1,self.K+1)  # indicies of subcarriers [1... K+1]
        
        self.pilot_sequence = pilot_sequence        # Known values of the pilot symbols to transmit before data
        self.Hest = np.ones(self.K)                 # Default flat channel response
 
        # Sync params
        self.sync_method = sync_method              # Synchronisation method
        
        # Schmidl & Cox params
        #self.symbols_per_frame = -1                # Number of sybmols per frame with SchmidlCox pramble
        self.L =  2048                              # Preamble length for Schmidl Cox Symbol
        
        # Chirp params
        self.f0 = 100                               # Chirp start frequency
        self.f1 = 8000                              # Chirp finish frequency
        self.chirp_length = 1                       # Chirp time
        self.chirp_method = 'linear'                # Chirp type
        self.end_chirp = end_chirp                  # Turns terminating chirp on/off
        self.gap_length = 0                         # Gap between chirp and transmission in samples


        # Modulation params
        self.modulation = modulation            # Modulation method
        
        if(self.modulation == "QPSK"):
            self.mapping_table = {              # Mapping table
                (0,0) : (1+1j)/np.sqrt(2),
                (1,0) : (1-1j)/np.sqrt(2),
                (1,1) : (-1-1j)/np.sqrt(2),
                (0,1) : (-1+1j)/np.sqrt(2),
            }
            self.mu = 2                         # Bits per symbol
        
        elif(self.modulation == "QAM"):
            self.mapping_table = {              # Mapping table
                (0,0) : (1+0j),
                (0,1) : (0+1j),
                (1,1) : (-1-0j),
                (1,0) : (0-1j),
            }
            self.mu = 2                         # Bits per symbol
            

        else:
            raise ValueError("Invalid Modulation Type")
            
        
        self.bits_per_symbol = len(self.data_carriers) * self.mu    # Bits per OFDM symbol
        
        
    # Create chirp for synchronisation
    def sync_chirp(self):
            
        t = np.linspace(0, self.chirp_length, self.chirp_length*self.fs)
        return chirp(t, f0=self.f0, f1=self.f1, t1=self.chirp_length, method=self.chirp_method)
        
            
            
    # Prints out key OFDM attributes
    def __repr__(self):
        return  "Number of actual Sub Carriers:      {:.0f} \nCyclic prefix length:               {:.0f} \nModulation method:                  {} \nSync Method:                        {}".format(self.K, self.cp_length, self.modulation, self.sync_method)
        
    

        
        
        
        
        
        
        
    #########################################
    #              Transmitter              #
    #########################################
    
class transmitter(CamG):                                # Inherits class attributes from CamG

    # Pad bits with zeros to right length
    def pad(self,bits):
        padding_length = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
        padding = np.zeros(padding_length)
        return np.hstack([bits,padding])
    
    def add_pilots(self,bits):
        if(self.pilot_sequence.size == 0):
            return bits
            
        elif(self.pilot_sequence.size == self.bits_per_symbol):
            return np.hstack([self.pilot_sequence,bits])

        else:
            raise ValueError("Known Sequence length must equal bits per OFDM symbol")
        
    
    # Shapes serial bits into parallel stream for OFDM
    def SP(self, bits):
        return bits.reshape(-1,len(self.data_carriers), self.mu)
        

    # Maps the bits to symbols
    def map(self, bits):
        if(type(bits) != np.ndarray): raise ValueError("Bits must be numpy array")
        return np.array([[self.mapping_table[tuple(b)] for b in bits[i,:]] for i in range(bits.shape[0])])
    
    
    # Allocates symbols and pilots, and adds zeros and conjugate to make signal real
    def build_OFDM_symbol(self, payload):
    
        symbols = np.zeros([payload.shape[0],self.ofdm_symbol_size], dtype=complex)     # overall subcarriers in symbols
        
        # Allocate data carriers
        symbols[:,self.data_carriers] = payload
        symbols[:,-self.data_carriers] = np.conj(payload)

        return symbols
        
    
    # Add the cyclic prefix
    def add_cp(self, time_data):
        cyclic_prefix = time_data[:,-self.cp_length:]            # take the last cp samples
        return np.hstack([cyclic_prefix, time_data])            # add them to the beginning
    
    
    # Prepare data for stream
    def send_to_stream(self, time_data):
        
        # Convert to serial stream and take real part
        tx_data = time_data.reshape(-1)
        tx = tx_data.real

        
        # Pad with zeros to separate from chirp
        padding = np.zeros(int(self.gap_length))
        
        # Chirp
        fsweep = self.sync_chirp()
        fsweep_reverse = fsweep[::-1]
        
        # Add chirp and return
        if(self.end_chirp == True):
            return np.hstack([fsweep,padding,tx,padding,fsweep_reverse])
        else:
            return np.hstack([fsweep,padding,tx])
    
    
    def graphs(self):
    
        # Print constellation
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b1, b0)
                Q = self.mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.1, "".join(str(x) for x in B), ha='center')
        plt.grid(alpha=0.5)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.title("QPSK Constellation with Gray Mapping")
        plt.show()
                
        
        # Print output data
    
    # Overall transmit function
    def transmit(self,bits, graph_output=False):
        
        
        print("-" * 42 + "\nTRANSMIT\n" + "-" * 42)
        print("OFDM Paramters:")
        print(self)
        
        bits_padded = self.pad(bits)
        
        bits_wpilots = self.add_pilots(bits_padded)
        
        bits_parallel = self.SP(bits_wpilots)
        constellation = self.map(bits_parallel)
        
        OFDM_symbols = self.build_OFDM_symbol(constellation)
        print("Number of bits to transmit:         " + str(len(bits)))
        print("Number of OFDM symbols to transmit: " + str(OFDM_symbols.shape[0]))
        
        time_data = np.fft.ifft(OFDM_symbols)

        time_data_cp = self.add_cp(time_data)
        
        signal = self.send_to_stream(time_data_cp)
    
        if(graph_output == True):
            time = np.linspace(0,len(signal)/self.fs,len(signal))
            plt.plot(time, signal)
            plt.title("Signal")
            plt.xlabel("time")
            plt.ylabel("Signal")
        
        return signal
        
        
        
        
        #########################################
        #               Receiver                #
        #########################################
        
class receiver(transmitter):
        
    
    # Get data from audio signal
    def get_symbols(self,r):
        
        # Chirp method
        if(self.sync_method == "chirp"):
        
            fsweep = self.sync_chirp()
            fsweep_reverse = fsweep[::-1]
            
            P = convolve(r, fsweep_reverse, mode="full")
            P_end = convolve(r, fsweep, mode="full")
            
            # Find max index and index of signal start
            start_index_max = np.where(P == np.amax(P))[0][0]
            end_index_max = np.where(P_end == np.amax(P_end))[0][0]
            
            zero_index = int(1 + start_index_max + self.gap_length)
            end_index = int(end_index_max + 1 - (self.chirp_length * self.fs + self.gap_length) )
            
            if(self.end_chirp == True):
                rx = r[zero_index:end_index]
            else:
                rx = r[zero_index:]
                end_index = len(rx) % self.ofdm_symbol_size
                rx = rx[:-end_index]
        
        # Schmidl & Cox method
        elif(self.sync_method == "schmidlcox"):
            
            # Search first 5s
            search_length = 5 * self.fs
            d_set = np.arange(0,search_length)
            
            # Calculate P using method in the paper
            P = np.zeros(len(d_set), dtype=complex)
            for d in d_set[:-1]:
                P[d+1] = P[d] + r[d+self.L].conj() * r[d+ 2*self.L] - r[d].conj() * r[d+self.L]
            
            zero_index = np.where(abs(P) == np.amax(abs(P)))[0][0] + self.ofdm_symbol_size - 1
            
            rx = r[zero_index:]
            end_index = len(rx) % self.ofdm_symbol_size
            rx = rx[:-end_index]
        
        else:
            raise ValueError("Invalid Synchronisation Method")
        
        return rx.reshape(-1,self.cp_length + self.ofdm_symbol_size), P
        
    
    # Remove the cyclic prefix
    def remove_cp(self, rx):
        return rx[:,self.cp_length:]
        
    
    # Separate data and pilot carriers
    def get_data(self, OFDM_symbols):
        
        # Extract pilot symbols
        if(self.pilot_sequence.size == 0):
            pilots = np.array([])
            data = OFDM_symbols[:,self.data_carriers]
            
        else:
            pilots = OFDM_symbols[0,self.data_carriers]
            data = OFDM_symbols[1:,self.data_carriers]
        return data, pilots
    
    
    # Calculate channel estimate from pilot carriers
    def channel_est(self, pilots):

        if(pilots.size == 0):
            pass
        
        else:
            # Map known pilot sequence onto values
            pilot_values = self.map(self.SP(self.pilot_sequence))
            # Find Hest
            self.Hest = pilots / pilot_values

        return self.Hest
        
        
    # Equalise data carriers from channel measurements
    def equalise(self, data_symbols, Hest):
        return data_symbols / Hest
    
    
    # De-map the constellation symbol to bits using min distance
    def demap(self, symbols):
        if(type(symbols) != np.ndarray): raise ValueError("Symbols must be numpy array")
        
        demapping_table = {v : k for k, v in self.mapping_table.items()}
        
        # Array of possible constellation points
        constellation = np.array([x for x in demapping_table.keys()])
        
        # Calulate distance of each received symbol to each point in the constellation
        dists = np.array([abs(y.reshape((-1,1)) - constellation.reshape((1,-1))) for y in symbols])

        # For each received symbol choose the nearest constellation point
        const_index = dists.argmin(axis=2)
        hardDecision = constellation[const_index]
        
        # Transform constellation points into bit groups
        return np.array([np.vstack([demapping_table[C] for C in hardDecision[i,:]]) for i in range(hardDecision.shape[0]) ]), hardDecision
        
    
    # Shapes parallel bits back into serial stream
    def PS(self, bits):
        return bits.reshape((-1,))

    # Print channel estimation
    # Print symbol mapping
    
    # Print graphs
    def graphs(self):
        
        plt.plot(self.all_carriers / self.ofdm_symbol_size * self.fs, self.Hest.real, label='Estimated channel')
        plt.ylabel("|H(f)|")
        plt.xlabel("Frequency")
        plt.title("Channel Frequency Response Estimate")
        plt.show()
        
        h = np.fft.ifft(self.Hest)
        time = np.linspace(0,(len(h)), len(h))
        plt.plot(time[:100],h.real[:100])
        plt.title("Channel Impulse Response")
        plt.ylabel("h")
        plt.xlabel("time (samples)")
        plt.show()
        
        


    # Overall receive function
    def receive(self, signal, graph_output=False):
        
        print("-" * 42 + "\nReceive \n" + "-" * 42)
        print("OFDM Paramters:")
        print(self)
        
        rx_signal_cp, start_sync_signal = self.get_symbols(signal)
        
        rx_signal = self.remove_cp(rx_signal_cp)
        
        print("Number of received OFDM symbols:    " + str(len(rx_signal)))
        
        OFDM_symbols = np.fft.fft(rx_signal)

        data, pilots = self.get_data(OFDM_symbols)
        
        Hest = self.channel_est(pilots)
        
        data_symbols = self.equalise(data, Hest)
        
        bits_parallel, hardDecision = self.demap(data_symbols)
        
        bits = self.PS(bits_parallel)
        
        print("Number of received bits:            " + str(len(bits)))
        
        if(graph_output == True):
            for qam, hard in zip(data_symbols[0,:10], hardDecision[0,:10]):
                plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
                plt.plot(hardDecision.real, hardDecision.imag, 'ro')
            plt.show()
        
        return bits
        
            
            


# Play Audio:
    # sd.play(myarray, fs)
    # sd.wait()

# Save audio:
    # wavfile.write(r"filename", fs, myarray)

# Record Audio
    # signal =sd.rec(int(duration * fs), samplerate=fs)
    # sd.wait()
    
# Play and record audio
def play_record(signal,fs, padding_before=1, padding_after=1):

    data_padded = np.pad(signal, (int(padding_before*fs), int(padding_after*fs)), 'constant', constant_values=0)
    print("Recording...")
    signal = sd.playrec(data_padded, fs)
    sd.wait()
    print("Finished recording")
    return signal[:,0]
    
    
    
# To do


#class channel(CamG):
    # Channel response for simulating channel

#def output file # Play if audio, show if picture

#def bitwise error correcting code


class channel(receiver):

    def measure_channel(self, bits):
    
        print("-" * 42 + "\nMeasure Channel\n" + "-" * 42)
        print("OFDM Paramters:")
        print(self)
    
        # Set pilots to 0
        self.P = 0
        
        # Transmit signal
        bits_padded = self.pad(bits)
        bits_parallel = self.SP(bits_padded)
        constellation = self.map(bits_parallel)
        OFDM_symbols = self.build_OFDM_symbol(constellation)
        time_data = np.fft.ifft(OFDM_symbols)
        time_data_cp = self.add_cp(time_data)
        signal = self.send_to_stream(time_data_cp)
        
        # Play and record signal
        signal = play_record(signal, self.fs)
        
        # Receive symbol data
        rx_signal_cp, sync = self.get_symbols(signal)
        rx_signal = self.remove_cp(rx_signal_cp)
        OFDM_symbols = np.fft.fft(rx_signal)
        data, pilots = self.get_data(OFDM_symbols)
        
        # Measure channel response
        H_all = data / constellation
        
        H = np.zeros(len(self.data_carriers),dtype=complex)
        
        for i in range(0,len(self.data_carriers)):
            H.real[i] = np.mean(H_all[:,i].real)
            H.imag[i] = np.mean(H_all[:,i].imag)
        
        plt.plot(self.data_carriers / self.ofdm_symbol_size * self.fs, H.real, label='Measured channel')
        plt.ylabel("|H(f)|")
        plt.xlabel("Frequency")
        plt.title("Channel Frequency Response")
        plt.savefig("plots/frequency_response")
        plt.show()
        
        h = np.fft.ifft(H)
        time = np.linspace(0,(len(h)), len(h))
        plt.plot(time[:250],h.real[:250])
        plt.title("Channel Impulse Response")
        plt.ylabel("h")
        plt.xlabel("time (samples)")
        plt.savefig("plots/impulse_response")
        plt.show()

        
        #np.savetxt('channel.csv', h[:100], delimiter=',')
        
        return H, h

# Save the output file using specified name and file size detecting null terminated string
def save_file(rx_bits, audio_fs=8000):

    file_name = []
    file_size = []
    data = np.packbits(rx_bits)

    for item in data:
        if(item == 0):
            data = data[1:]
            break
        else:
            file_name.append(item)
            data = data[1:]
        
    for item in data:
        if(item == 0):
            data = data[1:]
            break
        else:
            file_size.append(item)
            data = data[1:]

    file_name = "".join([chr(item) for item in file_name])
    file_size = "".join([chr(item) for item in file_size])

    print("File Name: " + file_name + "\nFile Size: " + file_size + " bytes")

    data = data[:int(file_size)]
    
    if(file_name[-3:] == "wav"):
        wavfile.write(r"output_files/" + file_name, audio_fs, data)
        
        
    else:
        f = open("output_files/" + file_name, 'wb')
        f.write(data)
        f.close()
        
    return file_name
