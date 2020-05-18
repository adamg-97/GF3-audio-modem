# Header file for OFDM, add libraries and functions here
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve
from scipy.io import wavfile
import sounddevice as sd
sd.default.channels = 1


#########################################
#                 CamG                  #
#########################################

class CamG:
    def __init__(self, ofdm_symbol_size, cp_length, modulation, P=0, pilot_value=None):
    
        # OFDM params
        self.ofdm_symbol_size = ofdm_symbol_size    # OFDM symbol size
        self.K = self.ofdm_symbol_size // 2 - 1     # Number of carriers in OFDM symbol i.e. DFT size / 2 - 1
        self.cp_length = cp_length                  # length of cyclic prefix
        self.P = P                                  # number of pilot carriers per block
        self.pilot_value = pilot_value              # Known value that pilot transmits
        
        # Audio params
        self.fs = 44100                             # Audio sampling rate
        self.gap_length = 1 * self.fs                    # Gap between chirp and transmitted data
        
        # Chirp params
        self.f0 = 500                               # Chirp start frequency
        self.f1 = 6000                              # Chirp finish frequency
        self.chirp_length = 2                       # Chirp time
        self.chirp_method = 'linear'                # Chirp type

        # Carrier params
        self.all_carriers = np.arange(1,self.K+1)                                   # indicies of all subcarriers [1... K]
        try:    self.pilot_carriers = self.all_carriers[::(self.K+1) // self.P]     # Pilot carriers every K/Pth carrier
        except: self.pilot_carriers = np.array([])                                  # Exception for no pilot carriers
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers-1)    # Remaining carriers are data carriers
        
        
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
            
        
        self.bits_per_symbol = len(self.data_carriers) * self.mu            # Bits per OFDM symbol = number of carriers x modulation index
        
    # Create chirp for synchronisation
    def sync_chirp(self):
            
        t = np.linspace(0, self.chirp_length, self.chirp_length*self.fs)
        return chirp(t, f0=self.f0, f1=self.f1, t1=self.chirp_length, method=self.chirp_method)
            
            
        # Prints out key OFDM attributes
    def __repr__(self):
        return  "Number of actual Sub Carriers:      {:.0f} \nCyclic prefix length:               {:.0f} \nModulation method:                  {}".format(self.K, self.cp_length, self.modulation)
        
        

        
        
        
        
        
        
        
    #########################################
    #              Transmitter              #
    #########################################
    
class transmitter(CamG):                                # Inherits class attributes from CamG

    # Pad bits to right length

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
        
        # Allocate pilot subcarriers
        try:
            symbols[:,self.pilot_carriers] = self.pilot_value
            symbols[:,-self.pilot_carriers] = np.conj(self.pilot_value)
        except: None
        
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
        tx = np.pad(tx, (self.gap_length,0), mode='constant', constant_values=0)
        
        # Chirp
        fsweep = self.sync_chirp()
        
        # Add chirp and return
        return np.hstack([fsweep,tx])
        
    
    def graphs(self):
    
        # Print constellation
        plt.figure(figsize=(3,2), dpi= 120)
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
                
        # Print carriers
        plt.figure(figsize=(6,1), dpi=80)
        plt.plot(self.pilot_carriers, np.zeros_like(self.pilot_carriers), 'bo', label='pilot')
        plt.plot(self.data_carriers, np.zeros_like(self.data_carriers), 'ro', label='data')
        plt.yticks([])
        plt.xlim(0,32)
        plt.title("Carriers")
        plt.legend(bbox_to_anchor=(1.1, 0.75))
        plt.show()
        
        # Print output data
    
    # Overall transmit function
    def transmit(self,bits):
        
        
        print("\n" + "-" * 42 + "\nTRANSMIT\n" + "-" * 42 + "\n")
        print("\nOFDM Paramters:")
        print(self)
        
        bits_parallel = self.SP(bits)
    
        constellation = self.map(bits_parallel)
        
        OFDM_symbols = self.build_OFDM_symbol(constellation)
        print("Number of bits to transmit:         " + str(len(bits)))
        print("Number of OFDM symbols to transmit: " + str(OFDM_symbols.shape[0]))
        
        time_data = np.fft.ifft(OFDM_symbols)

        time_data_cp = self.add_cp(time_data)
        
        return self.send_to_stream(time_data_cp)
        
        
        
        
        
        #########################################
        #               Receiver                #
        #########################################
        
class receiver(CamG):
    
    # Get data from audio signal
    def get_symbols(self,signal):
        
        fsweep_reverse = self.sync_chirp()[::-1]
        
        sync_signal = np.convolve(signal,fsweep_reverse,mode="same")
        
        # Find max index and index of signal start
        index_max = np.where(sync_signal == np.amax(sync_signal))[0][0]
        zero_index = int(1 + index_max + self.chirp_length/2 * self.fs + self.gap_length)
        
        rx = signal[zero_index -1:zero_index + 100 * (self.ofdm_symbol_size + self.cp_length)]               # This should be adjusted when second end chirp is added in
        
        return rx.reshape(-1,self.cp_length + self.ofdm_symbol_size)
        
    
    # Remove the cyclic prefix
    def remove_cp(self, rx):
        return rx[:,self.cp_length:]
        
    
    # Separate data and pilot carriers
    def get_data(self, OFDM_symbols):
        
        # Extract pilot symbols
        try:
            pilots = OFDM_symbols[:,self.pilot_carriers]
        except:
            pilots = None
        
        # Extract data symbols
        data = OFDM_symbols[:,self.data_carriers]
        
        return data, pilots
    
    
    # Calculate channel estimate from pilot carriers
    def channel_est(self, pilots):
        try:
            # Divide by transmitted pilot values and average over OFDM symbols
            H_pilots = pilots / self.pilot_value
        
            H_est_pilots = np.zeros(H_pilots.shape[1],dtype=complex)
        
            for i in range(0,H_pilots.shape[1]):
                H_est_pilots.real[i] = np.mean(H[:,i].real)
                H_est_pilots.imag[i] = np.mean(H[:,i].imag)
        
            # Interpolate between the pilot carriers to get an estimate of the channel. Interpolate absolute value and phase eparately
        
            h_real_interp = scipy.interpolate.barycentric_interpolate(self.pilot_carriers, h_real_pilots, self.data_carriers)
            h_imag_interp = scipy.interpolate.barycentric_interpolate(self.pilot_carriers, h_imag_pilots, self.data_carriers)
        
            Hest = h_real_interp + 1j * h_imag_interp
        
        # Handle case of no pilot carriers
        except:
            Hest = np.ones(len(self.all_carriers))
            
        return Hest
        
        
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
    

    # Overall receive function
    def receive(self, signal):
        
        print("\n" + "-" * 42 + "\n Receive \n" + "-" * 42 + "\n")
        
        rx_signal_cp = self.get_symbols(signal)
        
        rx_signal = self.remove_cp(rx_signal_cp)
        
        OFDM_symbols = np.fft.fft(rx_signal)

        data, pilots = self.get_data(OFDM_symbols)
        
        Hest = self.channel_est(pilots)
        
        data_symbols = self.equalise(data, Hest)
        
        bits_parallel, hardDecision = self.demap(data_symbols)
        
        bits = self.PS(bits_parallel)

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
    print(type(data_padded))
    signal = sd.playrec(data_padded, fs)
    sd.wait()
    print("Finished recording")
    return signal[:,0]
    
    
    
# To do


#class channel(CamG):
    # Channel response for simulating channel

#def output file # Play if audio, show if picture

#def bitwise error correcting code
