# Header file for OFDM, add libraries and functions here
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve, lfilter
from scipy.io import wavfile
import sounddevice as sd
sd.default.channels = 1
from IPython.display import Audio
import pyldpc


#########################################
#                 CamG                  #
#########################################

class CamG:
    def __init__(self, mode, encoding="None", no_pilots=20, packet_length=180):
        
        # LDPC params
        self.encoding = encoding  # Encoding type LDPC (broken), XOR, or None
        
        # Audio params
        self.fs = 48000                                                             # Audio sampling rate
    
        # OFDM params
        self.ofdm_symbol_size = 4096                                                # OFDM symbol size
        self.K = self.ofdm_symbol_size // 2 - 1                                     # Number of carriers in OFDM symbol i.e. DFT size / 2 - 1
        
        modes = {
            "A1" : (224,(1,self.K)),
            "A2" : (224,(100,1500)),
            "A3" : (224,(100,1000)),
            "B1" : (704,(1,self.K)),
            "B2" : (704,(100,1500)),
            "B3" : (704,(100,1000)),
            "C1" : (1184,(1,self.K)),
            "C2" : (1184,(100,1500)),
            "C3" : (1184,(100,1000))
            }

        self.cp_length = modes[mode][0]                                             # length of cyclic prefix based on mode
        self.lowest_bin = modes[mode][1][0]                                         # Set lowest active freq bin
        self.highest_bin = modes[mode][1][1]                                        # Set highest active freq bin

        self.carriers = np.arange(1,self.K+1)                                       # indicies of subcarriers [1... K+1]
        self.data_carriers = np.arange(self.lowest_bin,self.highest_bin)            # Pick out data carriers
        self.data_carriers_per_symbol = len(self.data_carriers)                     # Number of turned on data carriers per OFDM symbol
        self.unused_carriers = np.delete(self.carriers, (self.data_carriers-1))     # Unused carriers
        self.packet_length = packet_length                                          # Number of OFDM data symbols in a packet
        self.no_pilots = no_pilots                                                  # Number of pilot symbols preceeding and following data
        self.known_sequence = np.array([],dtype=np.uint8)                           # Known values of the pilot symbols to transmit before data
        
 
        # Sync params
        self.sync_method = "chirp"                                                  # Synchronisation method
        
        # Schmidl & Cox Params
        self.L = self.K + 1                                                         # Preamble size - note this is the number of actual OFDM carriers K + zero frequency
        
        # Chirp params
        self.f0 = 0                                                                 # Chirp start frequency
        self.f1 = 8000                                                              # Chirp finish frequency
        self.chirp_length = 5*(self.ofdm_symbol_size + self.cp_length)              # Chirp time

        

        # Modulation params
        self.modulation = "QPSK"                                                    # Modulation method
        
        if(self.modulation == "QPSK"):
            self.mapping_table = {                                                  # Mapping table
                (0,0) : (1+1j)/np.sqrt(2),
                (1,0) : (1-1j)/np.sqrt(2),
                (1,1) : (-1-1j)/np.sqrt(2),
                (0,1) : (-1+1j)/np.sqrt(2),
            }
            self.mu = 2                                                             # Bits per symbol
        
        elif(self.modulation == "QAM"):
            self.mapping_table = {                                                  # Mapping table
                (0,0) : (1+0j),
                (0,1) : (0+1j),
                (1,1) : (-1-0j),
                (1,0) : (0-1j),
            }
            self.mu = 2                                                             # Bits per symbol
            

        else:
            raise ValueError("Invalid Modulation Type")
            
        
        self.data_bits_per_symbol = self.data_carriers_per_symbol * self.mu         # Data bits per OFDM symbol
        self.bits_per_symbol = self.K * self.mu                                     # Total bits per OFDM symbol
        
        
        # Get known sequence for channel estimation
        with open("handouts/random_bits.txt", mode='rb') as f:
            for i in range(self.ofdm_symbol_size):
                self.known_sequence = np.append(self.known_sequence, int(f.read(1)))
        
        
        
    # Create chirp for synchronisation
    def sync_chirp(self):
            
        t = np.linspace(0, self.chirp_length/self.fs, self.chirp_length)
        return chirp(t, f0=self.f0, f1=self.f1, t1=self.chirp_length/self.fs, method='linear') / 5         # Reduce amplitude of chirp to save ears!
        
            
    # Prints out key OFDM attributes
    def __repr__(self):
        return  "Number of actual Sub Carriers:      {:.0f} \nCyclic prefix length:               {:.0f} \nModulation method:                  {} \nSync Method:                        {} \nPacket Length:                      {}".format(self.K, self.cp_length, self.modulation, self.sync_method, self.packet_length)
        
    

        
        
    #########################################
    #              Transmitter              #
    #########################################
    
    
class transmitter(CamG):                                # Inherits class attributes from CamG

    # Bitwsie encoding
    def encode(self, bits):
        
        
        n = self.data_bits_per_symbol // 50     # Set n at symbol bit length / 50 = 28
        d_c = 7                                 # Set number of 1's in coded bits to be half
        d_v = 5

        ldpcSeed = 42
            
        if self.encoding == "LDPC":
            H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic = True, sparse = True)
            
            # Get k and the number of unencoded bits per packet so we can shape and pad the input bits
            k = G.shape[1]
            unencoded_bits_per_packet = 50 * k * self.packet_length
            
            # Pad bits to correct length
            padding_length = int((unencoded_bits_per_packet - (len(bits) % unencoded_bits_per_packet)) % unencoded_bits_per_packet)
            
            padding = np.random.binomial(n=1, p=0.5, size=(padding_length,))
            
            # Reshape to packets length k for coding
            padded_bits = np.reshape(np.hstack([bits,padding]), (k, -1))
            
            signal_coded = pyldpc.encode(G, padded_bits, snr = 10, seed = ldpcSeed)
            
            # Reshape back into serial array
            signal_coded = signal_coded.reshape(-1)
            
            # LDPC gives -1's and 1's, switch -1's to binary
            bits_encoded = np.where(signal_coded <= 0, 0, signal_coded)
            bits_encoded = np.where(bits_encoded > 0, 1, bits_encoded)
        
        
        # XOR with known data and pad
        elif(self.encoding == "XOR"):
            known_bits = np.tile(self.known_sequence[:self.data_bits_per_symbol], int(np.ceil(len(bits)/self.data_bits_per_symbol)))
            known_bits = known_bits[:len(bits)]
            bits = np.bitwise_xor(bits, known_bits)
            
            bits_per_packet = self.data_bits_per_symbol * self.packet_length
            padding_length = (bits_per_packet - len(bits) % bits_per_packet) % bits_per_packet
                
            # Pad with random data - packing with zeros is bad
            padding = np.random.binomial(n=1, p=0.5, size=(padding_length,))
            bits_encoded = np.hstack([bits,padding])
        
        
        # Else just pad the bits to correct length
        else:
            bits_per_packet = self.data_bits_per_symbol * self.packet_length
            
            padding_length = (bits_per_packet - len(bits) % bits_per_packet) % bits_per_packet
                
            # Pad with random data - packing with zeros is bad
            padding = np.random.binomial(n=1, p=0.5, size=(padding_length,))
            
            bits_encoded = np.hstack([bits,padding])

        return bits_encoded
        
    
    # Shapes serial bits into parallel stream for OFDM
    def SP(self, bits):
        return bits.reshape(-1,self.data_carriers_per_symbol, self.mu)
        

    # Maps the bits to symbols
    def map(self, bits):
        return np.array([[self.mapping_table[tuple(b)] for b in bits[i,:]] for i in range(bits.shape[0])])

    
    # Generates random QPSK symbols for filling unused frequencies
    def random_qpsk(self):
        qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        return np.random.choice(qpsk, size=(self.K - self.data_carriers_per_symbol), replace=True)
        
        
    # Allocates symbols and adds zeros and conjugate to make signal real
    def build_OFDM_symbol(self, payload):
        
        symbols = np.zeros([payload.shape[0],self.ofdm_symbol_size], dtype=complex)     # overall subcarriers in symbols
        rand_qpsk = self.random_qpsk()
        # Allocate data carriers
        symbols[:,self.data_carriers] = payload
        symbols[:,self.unused_carriers] = rand_qpsk
        symbols[:,-self.data_carriers] = np.conj(payload)
        symbols[:,-self.unused_carriers] = np.conj(rand_qpsk)
        
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
        
        known_symbols = self.map(self.known_sequence[:self.bits_per_symbol].reshape(-1,self.K, self.mu))        # Map first mu*K bits from known bit sequence to symbols
        known_ofdm = np.zeros([1,self.ofdm_symbol_size], dtype=complex)
        known_ofdm[0,self.carriers] = known_symbols
        known_ofdm[0,-self.carriers] = np.conj(known_symbols)
        known_time = self.add_cp(np.fft.ifft(known_ofdm))                                                       # Convert to time domain and add CP
        
        
        packets = time_data.reshape(-1, self.packet_length, self.ofdm_symbol_size+self.cp_length)               # Reshape data into packets
        self.no_packets = packets.shape[0]                                                                      # Get number of packets in stream
        known_time = np.tile(known_time, (self.no_packets,self.no_pilots,1))                                    # Tile known data across number of pilot symbols and packets
            
        sync = np.tile(sync, (self.no_packets,1))                                                               # Tile sync across packets
        tx_data = 2 * np.hstack([known_time, packets, known_time])                                              # Add known data to start and end of each packet x2 for volume (signal was too quiet on laptop speakers at full volume)
        tx_data = np.hstack([sync, tx_data.reshape(self.no_packets,-1)])                                        # Add sync to each packet
        tx = tx_data.reshape(-1).real                                                                           # Shape to serial
        tx = np.hstack([tx,sync[0]])                                                                            # Add a chirp at the very end of the signal
        
        
        # Create frames for visuals
        frame_length = tx.shape[0]
        sync_valid = np.zeros(frame_length); known_valid = np.zeros(frame_length); payload_valid = np.zeros(frame_length)
        sync_valid[0:sync.shape[1]] = 1; sync_valid[-sync.shape[1]:] = 1
        
        for f in np.hstack([np.arange(self.no_pilots),np.arange(self.no_pilots+self.packet_length,2*self.no_pilots+self.packet_length)]):
            known_valid[sync.shape[1] + (self.cp_length + self.ofdm_symbol_size) * f * 1 + self.cp_length + np.arange(self.ofdm_symbol_size)] = 1
            
        for f in range(self.packet_length):
            payload_valid[sync.shape[1] + (self.cp_length + self.ofdm_symbol_size) * (self.no_pilots + f) + self.cp_length + np.arange(self.ofdm_symbol_size)] = 1
            
        sync_valid = np.tile(sync_valid,self.no_packets); known_valid = np.tile(known_valid, self.no_packets); payload_valid = np.tile(payload_valid, self.no_packets)
    
        return tx, sync_valid, known_valid, payload_valid
    
    
    # Print out the QPSK mapping
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
        
        bits_encoded = self.encode(bits)
    
        
        bits_parallel = self.SP(bits_encoded)
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
        
        
        # Plot the output signal with frames superimposed on top
        if(graph_output == True):
            time = np.linspace(0,len(signal)/self.fs,len(signal))
            plt.plot(time, 5*signal, label="Signal")
            plt.plot(time, sync_valid, label="Sync")
            plt.plot(time, known_valid, label="Known Data")
            plt.plot(time, payload_valid, label="Payload")
            plt.title("OFDM Frame")
            plt.xlabel("time")
            plt.legend()
            plt.savefig("OFDM Frame")
            plt.show()
            
        return signal
        
        
        
        
        #########################################
        #               Receiver                #
        #########################################
        
        
class receiver(transmitter):

    
    def chirp_method(self,r):
        fsweep = self.sync_chirp()[::-1]
        P = convolve(r, fsweep, mode="full")                # Convolve to find peaks
        P = P / np.amax(P)                                  # Normalise using max
        D = np.diff(P)                                      # Differentiate
        zeros = ((D[:-1] * D[1:]) <= 0) * (P[1:-1] > 0.4)   # Find zero crossings


        for i in range(len(zeros)):
            if(zeros[i] == 1):
                try:
                    for j in range(self.chirp_length):
                        zeros[i+1+j] = 0
                except:
                    zeros[:i+1] = 0

        return zeros
        
        

    def schmidlcox_method(self,r):                          # No longer works with packets - unused after chirp sync written into standard
            
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
        
        zero_indicies = np.where(zeros == True)[0] + 2      # Get indices of sync pulses (shift by two to get exact start of data)
        
        zero_indicies = zero_indicies[:-1]                  # Get rid of the terminating chirp since we're not using it
        
        self.no_packets = len(zero_indicies)
            
        # Extract the data and pilots from each packet
        rx = np.vstack([[r[i:i + (2*self.no_pilots + self.packet_length) * (self.cp_length + self.ofdm_symbol_size)]] for i in zero_indicies])
            
        # Return packets shaped into symbols + CP's
        return rx.reshape(-1, 2*self.no_pilots + self.packet_length, self.cp_length + self.ofdm_symbol_size)
        
    
    # Remove the cyclic prefix
    def remove_cp(self, rx):
        return rx[:,:,self.cp_length:]
        
    
    # Separate data and pilot symbols
    def get_data(self, OFDM_symbols):
        
        start_pilots = OFDM_symbols[:,:self.no_pilots, self.carriers]
        end_pilots = OFDM_symbols[:,-self.no_pilots:, self.carriers]
        data_symbols = OFDM_symbols[:,self.no_pilots:-self.no_pilots,self.carriers]
            
        return data_symbols, start_pilots, end_pilots
    
    
    # Calculate channel estimate from pilot carriers and equalise
    def equalise(self, data_symbols, start_pilots, end_pilots):

        if(self.no_pilots == 0):
            return data_symbols.reshape(-1,self.K)
        
        else:
            # Map known pilot sequence onto values
            known_symbols = self.map(self.known_sequence[:self.bits_per_symbol].reshape(-1,self.K, self.mu))
            
            # Setup arrays for moving Hest
            Hest_mag = np.zeros((self.no_packets,self.K), dtype=complex)
            Hest = np.zeros_like(data_symbols)
            phase = np.zeros_like(data_symbols)
            
            # Setup array for equalised data
            data_eq = np.zeros_like(data_symbols)
                    
            # Take average over pilot values
            Hest_start = np.zeros((self.no_packets,self.K), dtype=complex)
            Hest_end = np.zeros((self.no_packets,self.K), dtype=complex)
            
            for i in range(self.no_packets):
                for k in range(self.K):
                    Hest_start.real[i,k] = np.mean(start_pilots[i,:,k].real)
                    Hest_start.imag[i,k] = np.mean(start_pilots[i,:,k].imag)
                    Hest_end.real[i,k] = np.mean(end_pilots[i,:,k].real)
                    Hest_end.imag[i,k] = np.mean(end_pilots[i,:,k].imag)
                # Divide by known symbols to get channel estimate
                Hest_start[i] = Hest_start[i] / known_symbols
                Hest_end[i] = Hest_end[i] / known_symbols
            
            Hest_mag = abs(Hest_end)
            phase0 = np.angle(Hest_start)                  # Phase at start
            phase1 = np.angle(Hest_end)                    # Phase at end
            
            phase_diff = np.unwrap(phase1) - np.unwrap(phase0)                   # Phase difference
            p = np.zeros(self.no_packets)                                        # Initialise array for gradient of phase difference at each packet
            
            # Get the gradient of the unwrapped phase difference using reliable points in the middle of the range
            for i in range(self.no_packets):
                p[i] = np.polyfit(np.arange(len(phase_diff[i,500:1000])),phase_diff[i,500:1000],1)[0]
                
            
            # Equalise data carriers from measurements
            for i in range(self.no_packets):                                          # Iterate over packets
                for l in range(self.packet_length):                                   # Iterate over symbols in packet
                    for n in range(self.K):
                    
                        # Correct magnitude
                        Hest[i,l,n] = abs(Hest_start[i,n]) + (abs(Hest_end[i,n]) - abs(Hest_start[i,n])) * (l + self.no_pilots/2) / (self.packet_length + self.no_pilots)
                        
                        # Correct phase
                        phase[i,l,n] = (p[i]*n*(l + self.no_pilots/2) / (self.packet_length + self.no_pilots))
                        Hest[i,l,n] = Hest[i,l,n] * np.exp(1j * (np.angle(Hest_start[i,n]) + phase[i,l,n].real))
                        
                        # Equalise data wrt to channel estimate
                        data_eq[i,l,n] = data_symbols[i,l,n] / Hest[i,l,n]
            
        return data_eq.reshape(-1,self.K), Hest_start, Hest_end, Hest
    
    
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


    # Decoding
    def decode(self, bits_encoded):
        if(self.encoding == "LDPC"):
            n = self.data_bits_per_symbol // 50           # Set n at symbol bit length / 50 = 28
            
            
            # Set d_c and d_v
            d_c = 7
            d_v = 5
            
            # Set LDPC seed
            ldpcSeed = 42
                
            H, G = pyldpc.make_ldpc(n, d_v, d_c, seed = ldpcSeed, systematic = True, sparse = True)
            k = G.shape[1]
            
            # pyldpc uses -1 and 1, not binary
            signal_encoded = np.where(bits_encoded == 0, -1, bits_encoded)
            
            # Shape into blocks for decoding
            signal_encoded = signal_encoded.reshape(n,-1)
            
            # Decode
            D = np.zeros(signal_encoded.shape)
            decoded_bits = np.zeros((k,signal_encoded.shape[1]))
            
            for i in range(signal_encoded.shape[1]):
                D[:, i] = pyldpc.decode(H, signal_encoded[:, i], snr = 10, maxiter = 10)
                decoded_bits[:, i] = np.around(pyldpc.get_message(G, D[:, i]))
            
            decoded_bits = decoded_bits.reshape(-1)
            print(decoded_bits)
            
        elif(self.encoding == "XOR"):
            known_bits = np.tile(self.known_sequence[:self.data_bits_per_symbol], int(np.ceil(len(bits_encoded)/self.data_bits_per_symbol)))
            known_bits = known_bits[:len(bits_encoded)]
            decoded_bits = np.bitwise_xor(bits_encoded, known_bits)
            
        else:
            decoded_bits = bits_encoded
        
        return decoded_bits
    
    
    # Print graphs
    def channel_response(self, Hest):
        
        plt.plot(self.carriers / self.ofdm_symbol_size * self.fs, abs(Hest), label='Estimated channel')
        plt.ylabel("|H(f)|")
        plt.xlabel("Frequency")
        plt.title("Channel Frequency Response Estimate")
        plt.savefig("plots/Channel_mag")
        plt.show()
        
        plt.plot(self.carriers / self.ofdm_symbol_size * self.fs, np.angle(Hest), label='Estimated channel')
        plt.ylabel("arg(H(f))")
        plt.xlabel("Frequency")
        plt.title("Channel Frequency Response Estimate")
        plt.savefig("plots/Channel_freq")
        plt.show()
        
        
        h = np.fft.ifft(Hest)
        time = np.linspace(0,(len(h)), len(h))
        plt.plot(time[:500],h.real[:500])
        plt.title("Channel Impulse Response")
        plt.ylabel("h")
        plt.xlabel("time (samples)")
        plt.savefig("plots/Channel_inpulse")
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

        data_symbols, start_pilots, end_pilots = self.get_data(OFDM_symbols)
        
        print("Number of received OFDM symbols:    " + str(self.no_packets * self.packet_length))
        
        data_symbols, Hest_start, Hest_end, Hest = self.equalise(data_symbols, start_pilots, end_pilots)
        
        
        # Extract data symbols and throw away unused carriers: data_carriers -1 as 0 frequency already removed
        data_symbols = data_symbols[:,self.data_carriers -1]

        bits_parallel, hardDecision = self.demap(data_symbols)

        bits = self.PS(bits_parallel)
        
        bits = self.decode(bits)
        
        print("Number of received bits:            " + str(len(bits)))
        
        
        # Plot phase correction due to frequency mismatch and equalised QPSK symbols
        if(graph_output == True):
            
            x = np.linspace(0,self.fs*self.K/self.ofdm_symbol_size,self.K)
            for i in np.arange(self.packet_length)[::10]:
                plt.plot(x,np.unwrap(np.angle(Hest[0,i,:])))
            plt.plot(x,np.unwrap(np.angle(Hest_end[0,:])), color="blue", label="Phase at End of Packet")
            plt.plot(x,np.unwrap(np.angle(Hest_start[0,:])), color="red",label="Phase at Start of Packet")
            plt.legend()
            plt.xlabel("Frequency")
            plt.ylabel("Phase Shift")
            plt.savefig("plots/Frequency_drift")
            plt.show()
        
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            #spine placement data centered
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            
            for i in range(17):
                for j in range(22):
                    plt.plot(data_symbols[10*i,8*j].real, data_symbols[10*i,8*j].imag, 'o')
            
            for b1 in [0, 1]:
                for b0 in [0, 1]:
                    B = (b1, b0)
                    Q = self.mapping_table[B]
                    plt.plot(Q.real, Q.imag, 'ro')
                    
            Q = self.mapping_table[(0,0)]
            plt.plot(Q.real, Q.imag, 'ro', label="Constellation Points")
            
            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.legend()
            plt.savefig("plots/constellation")
            plt.show()
        

        return bits, Hest_start[0], Hest_end[0]
        
            
            
            
            
            #########################################
            #          Additional Functions         #
            #########################################


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
    
    # Pad to account for delay in recording starting
    data_padded = np.pad(signal, (int(padding_before*fs), int(padding_after*fs)), 'constant', constant_values=0)
    print("Recording...")
    signal = sd.playrec(data_padded, fs)
    sd.wait()
    print("Finished recording")
    return signal[:,0]
    
    

# Not used since moving from pilot symbols to known OFDM symbols since known OFDM symbols provides the same function
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
        
        H = np.zeros(len(self.carriers),dtype=complex)
        
        for i in range(0,len(self.carriers)):
            H.real[i] = np.mean(H_all[:,i].real)
            H.imag[i] = np.mean(H_all[:,i].imag)
        
        plt.plot(self.carriers / self.ofdm_symbol_size * self.fs, H.real, label='Measured channel')
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


# Load input file
def load_file(file_name):
    data_bytes = np.fromfile("input_files/" + file_name,dtype=np.uint8)
    file_info = file_name + "\x00" + str(len(data_bytes)) + "\x00"
    b = bytearray()
    b.extend(map(ord, file_info))
    return np.unpackbits(np.hstack([b,data_bytes]))
    


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
    data.tofile("output_files/" + file_name[:-4] + "_received" + file_name[-4:])
    return file_name, data
