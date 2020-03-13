import numpy as np
import Function.Modulation as Modulation

# Generate some random bits, encode it to valid codewords and simulate transmission
def AWGN_transmission(SNR, batch_size, top_config, code, channel):
    K =  top_config.K_code
    #baseband signal
    x_bits = np.random.randint(0, 2, size=(batch_size, K))
    # coding
    u_coded_bits = code.encode_LDPC(x_bits)

    # BPSK modulation
    s_mod = Modulation.BPSK(u_coded_bits)
    # plus the noise
    y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, SNR)
    
    return x_bits, u_coded_bits, s_mod, ch_noise, y_receive

def ACGN_transmission(SNR, batch_size, top_config, code, channel):
    K =  top_config.K_code
    #baseband signal
    x_bits = np.random.randint(0, 2, size=(batch_size, K))
    # coding
    u_coded_bits = code.encode_LDPC(x_bits)

    # BPSK modulation
    s_mod = Modulation.BPSK(u_coded_bits)
    # plus the noise
    y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, SNR)
    
    return x_bits, u_coded_bits, s_mod, ch_noise, y_receive

def RLN_transmission(SNR, batch_size, top_config, code, channel):
    K =  top_config.K_code
    N =  top_config.N_code
    #baseband signal
    x_bits = np.random.randint(0, 2, size=(batch_size, K))
    
    #rayleigh factor
    r_factor = np.sqrt(np.square(np.sqrt(1/2)*np.random.randn(batch_size, N))+np.square(np.sqrt(1/2)*np.random.randn(batch_size, N)))
    
    # coding
    u_coded_bits = code.encode_LDPC(x_bits)

    # BPSK modulation
    s_mod = Modulation.BPSK(u_coded_bits)
    
    # plus the noise
    y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, r_factor, SNR)
    
    return x_bits, u_coded_bits, s_mod, ch_noise, y_receive, r_factor