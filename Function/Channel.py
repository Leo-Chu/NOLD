import numpy as np
import tensorflow as tf

class AWGN:
    def __init__(self, top_config, rng_seed=None):
        self.blk_len = top_config.blk_len
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        
    def generate_noise(self, batch_size):
        noise_awgn = self.rng.randn(batch_size, self.blk_len)
        return noise_awgn.astype(np.float32)
    
    def reset_noise_generator(self):
        self.rng = np.random.RandomState(self.rng_seed)

    def channel_transmit(self, batch_size, s_mod, SNR):
        ch_noise_normalize = self.generate_noise(batch_size)
        ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
        ch_noise = ch_noise_normalize * ch_noise_sigma
        y_receive = s_mod + ch_noise
        return y_receive, ch_noise_sigma, ch_noise

class ACGN:
    def __init__(self, top_config, rng_seed=None):
        self.blk_len = top_config.blk_len
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        self.cov_1_2_file = top_config.cov_1_2_file
        fin_cov_file = open(self.cov_1_2_file , 'rb')
        cov_1_2_mat = np.fromfile(fin_cov_file, np.float32, self.blk_len*self.blk_len)
        cov_1_2_mat = np.reshape(cov_1_2_mat, [self.blk_len, self.blk_len])
        fin_cov_file.close()
        self.awgn_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.blk_len])
        self.noise_tf = tf.matmul(self.awgn_noise, cov_1_2_mat)
        self.sess = tf.Session()
            
    def __del__(self): 
        self.sess.close()
        
    def generate_noise(self, batch_size):
        noise_awgn = self.rng.randn(batch_size, self.blk_len)
        noise_awgn = noise_awgn.astype(np.float32)
        noise_co_awgn = self.sess.run(self.noise_tf, feed_dict={self.awgn_noise: noise_awgn})
        return noise_co_awgn.astype(np.float32)
    
    def reset_noise_generator(self): # this function resets the file pointer or the rng generator to generate the same noise data
        self.rng = np.random.RandomState(self.rng_seed)

    def channel_transmit(self, batch_size, s_mod, SNR):
        ch_noise_normalize = self.generate_noise(batch_size)
        ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
        ch_noise = ch_noise_normalize * ch_noise_sigma
        y_receive = s_mod + ch_noise
        return y_receive, ch_noise_sigma, ch_noise
    
class RLN:
    def __init__(self, top_config, rng_seed=None):
        self.blk_len = top_config.blk_len
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        
    def generate_noise(self, batch_size):
        noise_awgn = self.rng.randn(batch_size, self.blk_len)
        return noise_awgn.astype(np.float32)
    
    def reset_noise_generator(self): # this function resets the file pointer or the rng generator to generate the same noise data
        self.rng = np.random.RandomState(self.rng_seed)

    def channel_transmit(self, batch_size, s_mod, rayleigh_factor, SNR):
        ch_noise_normalize = self.generate_noise(batch_size)
        ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
        ch_noise = ch_noise_normalize * ch_noise_sigma
        y_receive = np.multiply(s_mod, rayleigh_factor) + ch_noise
        return y_receive, ch_noise_sigma, ch_noise   