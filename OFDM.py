# Header file for OFDM, add libraries and functions here
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve, lfilter
from scipy.io import wavfile
import sounddevice as sd
import pandas as pd
sd.default.channels = 1
from IPython.display import Audio


#########################################
#                 CamG                  #
#########################################

class CamG:
    def __init__(self, mode, known_sequence=np.array([]), no_pilots=10, packet_length=180):
    
        # Audio params
        self.fs = 48000                                # Audio sampling rate
    
        # OFDM params
        self.ofdm_symbol_size = 4096                # OFDM symbol size
        self.K = self.ofdm_symbol_size // 2 - 1     # Number of carriers in OFDM symbol i.e. DFT size / 2 - 1
        self.data_carriers = np.arange(1,self.K+1)  # indicies of subcarriers [1... K+1]
        self.packet_length = packet_length          # Number of OFDM data symbols in a packet
        self.no_pilots = no_pilots                  # Number of pilot symbols preceeding and following data
        self.known_sequence = known_sequence        # Known values of the pilot symbols to transmit before data
        
        cp_modes = [224, 704, 1184]
        self.cp_length = cp_modes[mode-1]           # length of cyclic prefix based on mode
 
        # Sync params
        self.sync_method = "chirp"                  # Synchronisation method
        
        # Schmidl & Cox Params
        self.L = self.K + 1                         # Preamble size - note this is the number of actual OFDM carriers K + zero frequency
        
        # Chirp params
        self.f0 = 2000                                                  # Chirp start frequency
        self.f1 = 4000                                                  # Chirp finish frequency
        self.chirp_length = 5*(self.ofdm_symbol_size + self.cp_length)  # Chirp time



        # Modulation params
        self.modulation = "QPSK"                # Modulation method
        
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
            
        t = np.linspace(0, self.chirp_length/self.fs, self.chirp_length)
        return chirp(t, f0=self.f0, f1=self.f1, t1=self.chirp_length/self.fs, method='linear')
        
            
            
    # Prints out key OFDM attributes
    def __repr__(self):
        return  "Number of actual Sub Carriers:      {:.0f} \nCyclic prefix length:               {:.0f} \nModulation method:                  {} \nSync Method:                        {} \nPacket Length:                      {}".format(self.K, self.cp_length, self.modulation, self.sync_method, self.packet_length)
        
    

        
        
    #########################################
    #              Transmitter              #
    #########################################
    
class transmitter(CamG):                                # Inherits class attributes from CamG

    # Pad bits with zeros to right length for symbols and packets
    def pad(self,bits):
    
        # Pad to integer number of symbols
        if(self.packet_length == -1):
            padding_length = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
            
        # Pad to integer number of packets
        else:
            bits_per_packet = self.bits_per_symbol * self.packet_length
            padding_length = (bits_per_packet - len(bits) % bits_per_packet) % bits_per_packet
            
        # Pad with random data - packing with zeros is bad
        padding = np.random.binomial(n=1, p=0.5, size=(padding_length,))
        return np.hstack([bits,padding])
        
    
    # Shapes serial bits into parallel stream for OFDM
    def SP(self, bits):
        return bits.reshape(-1,self.K, self.mu)
        

    # Maps the bits to symbols
    def map(self, bits):
        return np.array([[self.mapping_table[tuple(b)] for b in bits[i,:]] for i in range(bits.shape[0])])

    
        
    # Allocates symbols and adds zeros and conjugate to make signal real
    def build_OFDM_symbol(self, payload):
        
        symbols = np.zeros([payload.shape[0],self.ofdm_symbol_size], dtype=complex)     # overall subcarriers in symbols
        
        # Allocate data carriers
        symbols[:,self.data_carriers] = payload
        symbols[:,-self.data_carriers] = np.conj(payload)
        
        return symbols
        
    
    # Add the cyclic prefix
    def add_cp(self, time_data):
        cyclic_prefix = time_data[:,-self.cp_length:]           # take the last cp samples
        if(self.cp_length == 0):
            return time_data
        else:
            return np.hstack([cyclic_prefix, time_data])        # add them to the beginning
    
    
    # Build Schmidl & Cox Symbols for synchronisation
    def build_schmidlcox(self):
        
        # Map known sequence to symbols
        symbols = self.map(self.SP(self.known_sequence))
        
        # Create all 0 symbol, adds known symbols at all odd bins
        p = np.zeros(self.K, dtype=complex)
        p[::2] = symbols[0,:self.K // 2]
        return p.reshape(-1, self.K)


    
    
    # Prepare data for stream
    def send_to_stream(self, time_data, sync):
    
        known_symbols = self.map(self.SP(self.known_sequence[:self.bits_per_symbol]))   # Map first mu*K bits from known bit sequence to symbols
        known_time = self.add_cp(np.fft.ifft(self.build_OFDM_symbol(known_symbols)))    # Convert to time domain and add CP
        
        
        
        # No packets
        if(self.packet_length == -1):
            known_time = np.tile(known_time, (self.no_pilots,1))                        # Tile known data across number of pilot symbols
            tx_data = np.hstack([known_time, time_data])                                # Add known data to start
            tx = np.hstack([sync, tx_data.reshape(-1)]).real                            # Make serial and add sync
            
            # Create frames for visuals
            sync_valid = np.zeros(len(tx)); known_valid = np.zeros(len(tx)); payload_valid = np.zeros(len(tx))
            sync_valid[0:len(sync)] = 1
            for f in range(self.no_pilots):
                known_valid[len(sync) + (self.cp_length + self.ofdm_symbol_size) * f + self.cp_length + np.arange(self.ofdm_symbol_size)] = 1
            for f in range(time_data.shape[0]):
                payload_valid[len(sync) + (self.cp_length + self.ofdm_symbol_size) * (self.no_pilots + f) + self.cp_length + np.arange(self.ofdm_symbol_size)] = 1
            self.no_packets = 1
            

        # Packets
        else:
            packets = time_data.reshape(-1, self.packet_length, self.ofdm_symbol_size+self.cp_length) # Reshape data into packets
            self.no_packets = packets.shape[0]                                          # Get number of packets in stream
            known_time = np.tile(known_time, (self.no_packets,self.no_pilots,1))        # Tile known data across number of pilot symbols and packets
            sync = np.tile(sync, (self.no_packets,1))                                   # Tile sync across packets
            tx_data = np.hstack([known_time, packets, known_time])                      # Add known data to start and end of each packet
            tx_data = np.hstack([sync, tx_data.reshape(self.no_packets,-1)])                 # Add sync to each packet
            tx = tx_data.reshape(-1).real                                               # Shape to serial
            
            # Create frames for visuals
            frame_length = tx_data.shape[1]
            sync_valid = np.zeros(frame_length); known_valid = np.zeros(frame_length); payload_valid = np.zeros(frame_length)
            sync_valid[0:sync.shape[1]] = 1
            for f in np.hstack([np.arange(self.no_pilots),np.arange(self.no_pilots+self.packet_length,2*self.no_pilots+self.packet_length)]):
                known_valid[sync.shape[1] + (self.cp_length + self.ofdm_symbol_size) * f * 1 + self.cp_length + np.arange(self.ofdm_symbol_size)] = 1
            for f in range(self.packet_length):
                payload_valid[sync.shape[1] + (self.cp_length + self.ofdm_symbol_size) * (self.no_pilots + f) + self.cp_length + np.arange(self.ofdm_symbol_size)] = 1
            sync_valid = np.tile(sync_valid,self.no_packets); known_valid = np.tile(known_valid, self.no_packets); payload_valid = np.tile(payload_valid, self.no_packets)
    
        return tx, sync_valid, known_valid, payload_valid
    
        
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
                
        
    
    # Overall transmit function
    def transmit(self,bits, graph_output=False):
        
        
        print("-" * 42 + "\nTRANSMIT\n" + "-" * 42)
        print("OFDM Paramters:")
        print(self)
        
        bits_padded = self.pad(bits)
    
        
        bits_parallel = self.SP(bits_padded)
        constellation = self.map(bits_parallel)
        
        OFDM_symbols = self.build_OFDM_symbol(constellation)
        print("Number of bits to transmit:         " + str(len(bits)))
        print("Number of OFDM symbols to transmit: " + str(OFDM_symbols.shape[0]))
        
        # Add Synchronisation
        if(self.sync_method =="schmidlcox"):
            sync = self.build_schmidlcox()

        elif(self.sync_method == "chirp"):
            sync = self.sync_chirp()
        
        time_data = np.fft.ifft(OFDM_symbols)
        time_data_cp = self.add_cp(time_data)
        
        signal, sync_valid, known_valid, payload_valid = self.send_to_stream(time_data_cp, sync)

        print("Number of packets to transmit:      " + str(self.no_packets))
        
        if(graph_output == True):
            time = np.linspace(0,len(signal)/self.fs,len(signal))
            plt.plot(time, signal, label="Signal")
            plt.plot(time, sync_valid, label="Sync")
            plt.plot(time, known_valid, label="Known Data")
            plt.plot(time, payload_valid, label="Payload")
            plt.title("OFDM Frame")
            plt.xlabel("time")
            plt.legend()
            plt.savefig("OFDM Frame")
            plt.show()
            
        return signal, sync_valid, known_valid, payload_valid
        
        
        
        
        #########################################
        #               Receiver                #
        #########################################
        
class receiver(transmitter):
    
    
    def chirp_method(self,r):
        fsweep = self.sync_chirp()[::-1]
        P = convolve(r, fsweep, mode="full")                # Convolve to find peaks
        P = P / np.amax(P)                                  # Normalise using max
        D = np.diff(P)                                      # Differentiate
        zeros = ((D[:-1] * D[1:]) <= 0) * (P[1:-1] > 0.9)   # Find zero crossings
        
        # Ignore zeros for chirp_length after first zero
        for i in range(len(zeros)):
            if(zeros[i] == 1):
                for j in range(self.chirp_length):
                    zeros[i+1+j] = 0
        return zeros
        
        
    def schmidlcox_method(self,r):
        # Schmidl & Cox method
            
        # Search first 5s
        search_length = 5 * self.fs
        d_set = np.arange(0,search_length)
                
        # Calculate P using method in the paper
        P = np.zeros(len(d_set), dtype=complex)
        for d in d_set[:-1]:
            P[d+1] = P[d] + r[d+self.L].conj() * r[d+ 2*self.L] - r[d].conj() * r[d+self.L]
                
        return np.where(abs(P) == np.amax(abs(P)))[0][0] + self.ofdm_symbol_size - 1
    
    
    # Get data from audio signal
    def get_symbols(self,r, zeros):
        
        # Get indicies of sync pulses
        zero_indicies = np.where(zeros == True)[0]
        
        if(self.packet_length == -1):
            self.no_packets = 1
            rx = r[zero_indicies[0]:]
            end_index = rx % (self.cp_length + self.ofdm_symbol_size)
            rx = rx[:-end_index]
            return rx.reshape(1, -1, self.cp_length + self.ofdm_symbol_size)
        
        else:
            self.no_packets = len(zero_indicies)
            
            # Extract the data and pilots from each packet
            rx = np.vstack((r[i:i + (2*self.no_pilots + self.packet_length) * (self.cp_length + self.ofdm_symbol_size)] for i in zero_indicies))
            
            # Return packets shaped into symbols + CP's
            return rx.reshape(-1, 2*self.no_pilots + self.packet_length, self.cp_length + self.ofdm_symbol_size)
        
    
    # Remove the cyclic prefix
    def remove_cp(self, rx):
        return rx[:,:,self.cp_length:]
        
    
    # Separate data and pilot carriers
    def get_data(self, OFDM_symbols):
        
        # Extract pilot symbols
        if(self.no_pilots == 0):
            pilots = np.array([])
            data_symbols = OFDM_symbols[:,:,self.data_carriers]
            
        else:
            start_pilots = OFDM_symbols[:,:self.no_pilots, self.data_carriers]
            end_pilots = OFDM_symbols[:,-self.no_pilots:, self.data_carriers]
            data_symbols = OFDM_symbols[:,self.no_pilots:-self.no_pilots,self.data_carriers]
        return data_symbols, start_pilots, end_pilots
    
    
    # Calculate channel estimate from pilot carriers and equalise
    def equalise(self, data_symbols, start_pilots, end_pilots):

        if(self.no_pilots == 0):
            return data_symbols.reshape(-1,self.K)
        
        else:
            # Map known pilot sequence onto values
            known_symbols = self.map(self.SP(self.known_sequence[:self.bits_per_symbol]))
            
            # Setup arrays for moving Hest
            Hest_mag = np.zeros((self.no_packets,self.K), dtype=complex)
            th0 = np.zeros((self.no_packets,self.K), dtype=complex)
            th1 = np.zeros((self.no_packets,self.K), dtype=complex)
            
            # Get magnitude and phase seperately
            for i in range(self.no_packets):
                for k in range(self.K):
                    Hest_mag[i,k] = np.mean(abs(start_pilots[i,:,k])) / abs(known_symbols[0,k])
                    th0[i,k] = np.mean(np.angle(start_pilots[i,:,k])) / np.angle(known_symbols[0,k])   # Phase at start
                    th1[i,k] = np.mean(np.angle(end_pilots[i,:,k])) / np.angle(known_symbols[0,k])     # Phase at end
                    
            
            # Equalise data carriers from measurements
            for i in range(self.no_packets):                                          # Iterate over packets
                for j in range(self.packet_length):                                   # Iterate over symbols in packet
                    # Equalise magnitude using initial hest and phase using linear phase interpolation
                    data_symbols[i,j] = data_symbols[i,j] / Hest_mag[i] * np.exp(-1j * (th0[i] + j*(th1[i]-th0[i])/self.packet_length))
            
        return data_symbols.reshape(-1,self.K)
    
    
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
        
        plt.plot(self.data_carriers / self.ofdm_symbol_size * self.fs, abs(self.Hest), label='Estimated channel')
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
        
        zeros = self.chirp_method(signal)
        
        rx_signal_cp = self.get_symbols(signal,zeros)
        
        rx_signal = self.remove_cp(rx_signal_cp)
        
        OFDM_symbols = np.fft.fft(rx_signal)
        print(OFDM_symbols)
        data_symbols, start_pilots, end_pilots = self.get_data(OFDM_symbols)
        
        print("Number of received OFDM symbols:    " + str(self.no_packets * self.packet_length))
        
        #data_symbols = self.equalise(data_symbols, start_pilots, end_pilots)
        data_symbols = data_symbols.reshape(-1,self.K)

        bits_parallel, hardDecision = self.demap(data_symbols)
        
        bits = self.PS(bits_parallel)
        
        print("Number of received bits:            " + str(len(bits)))
        
        if(graph_output == True):
            for j in range(100):
                plt.plot(data_symbols[0,j].real, data_symbols[0,j].imag, 'bo')
            for b1 in [0, 1]:
                for b0 in [0, 1]:
                    B = (b1, b0)
                    Q = self.mapping_table[B]
                    plt.plot(Q.real, Q.imag, 'ro')
            
            plt.show()
            
        if(False == True):
            for qam, hard in zip(data_symbols[0,::64], hardDecision[0,::64]):
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
        self.no_pilots = 0
        self.packet_length = -1
        
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
def save_file(rx_bits):

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
        
    return file_name, data
