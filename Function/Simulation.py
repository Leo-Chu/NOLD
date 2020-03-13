import numpy as np
import datetime
import tensorflow as tf
import Function.Modulation as Modulation
import Function.Transmission as Transmission
import Function.DataIO as DataIO
import Function.BP_MS_Decoder as BP_MS_Decoder


def LDPC_BP_MS_AWGN_test(code, dec_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from dec_config
    N = dec_config.N_code
    K = dec_config.K_code
    H_matrix = code.H_matrix
    SNR_set = dec_config.SNR_set
    BP_iter_num = dec_config.BP_iter_nums
    alpha = dec_config.alpha
    beta = dec_config.beta
    function = 'LDPC_BP_MS_AWGN_test'
    # build BP decoding network
    bp_decoder = BP_MS_Decoder.BP_NetDecoder(H_matrix, batch_size, alpha, beta)
    
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (dec_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (dec_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (dec_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, dec_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod         
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])
            
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

def LDPC_BP_MS_ACGN_test(code, dec_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from dec_config
    N = dec_config.N_code
    K = dec_config.K_code
    H_matrix = code.H_matrix
    SNR_set = dec_config.SNR_set
    BP_iter_num = dec_config.BP_iter_nums
    alpha = dec_config.alpha
    beta = dec_config.beta
    function = 'LDPC_BP_MS_ACGN_test'
    # build BP decoding network
    bp_decoder = BP_MS_Decoder.BP_NetDecoder(H_matrix, batch_size, alpha, beta)
    
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (dec_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (dec_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (dec_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, dec_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])
            
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')
    
def LDPC_BP_MS_RLN_test(code, dec_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from dec_config
    N = dec_config.N_code
    K = dec_config.K_code
    H_matrix = code.H_matrix
    SNR_set = dec_config.SNR_set
    BP_iter_num = dec_config.BP_iter_nums
    alpha = dec_config.alpha
    beta = dec_config.beta
    function = 'LDPC_BP_MS_RLN_test'
    # build BP decoding network
    bp_decoder = BP_MS_Decoder.BP_NetDecoder(H_matrix, batch_size, alpha, beta)
    
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (dec_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (dec_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (dec_config.decoding_x_file, SNR))
        r_factor_file = format('%s_%.1f.dat' % (dec_config.decoding_r_file, SNR))
        factorio_decode = DataIO.FactorDataIO(r_factor_file, dec_config)
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, dec_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            r_factor = factorio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)      
            ch_noise = y_receive - np.multiply(s_mod, r_factor)    
            LLR = y_receive
            ##practical noise 
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])
            
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')
    
def LDPC_BP_QMS_AWGN_test(code, dec_config, simutimes_range, target_err_bits_num, batch_size, bp_decoder):
    # load configurations from dec_config
    N = dec_config.N_code
    K = dec_config.K_code
    SNR_set = dec_config.SNR_set
    BP_iter_num = dec_config.BP_iter_nums
    para_file = dec_config.para_file
    function = 'LDPC_BP_QMS_AWGN_test'

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    # initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (dec_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        para_data_file = format('%sPARA(%d_%d)_SNR%.1f_Iter%d.txt' % (para_file, N, K, SNR, BP_iter_num))
        para = np.loadtxt(para_data_file, np.float32)
        alpha = para[0,:]
        beta = para[1,:]
        
        y_recieve_file = format('%s_%.1f.dat' % (dec_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (dec_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, dec_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.quantized_decode(LLR.astype(np.float32), BP_iter_num[0], alpha, beta)
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

def LDPC_BP_QMS_ACGN_test(code, dec_config, simutimes_range, target_err_bits_num, batch_size, bp_decoder):
    # load configurations from dec_config
    N = dec_config.N_code
    K = dec_config.K_code
    SNR_set = dec_config.SNR_set
    BP_iter_num = dec_config.BP_iter_nums
    para_file = dec_config.para_file
    function = 'LDPC_BP_QMS_ACGN_test'

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    # initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (dec_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        para_data_file = format('%s/PARA(%d_%d)_SNR%.1f_Iter%d.txt' % (para_file, N, K, SNR, BP_iter_num))
        para = np.loadtxt(para_data_file, np.float32)
        alpha = para[0,:]
        beta = para[1,:]
        
        y_recieve_file = format('%s_%.1f.dat' % (dec_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (dec_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, dec_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.quantized_decode(LLR.astype(np.float32), BP_iter_num[0], alpha, beta)
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

def LDPC_BP_QMS_RLN_test(code, dec_config, simutimes_range, target_err_bits_num, batch_size, bp_decoder):
    # load configurations from dec_config
    N = dec_config.N_code
    K = dec_config.K_code
    SNR_set = dec_config.SNR_set
    BP_iter_num = dec_config.BP_iter_nums
    para_file = dec_config.para_file
    function = 'LDPC_BP_QMS_RLN_test'

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    # initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (dec_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        para_data_file = format('%s/PARA(%d_%d)_SNR%.1f_Iter%d.txt' % (para_file, N, K, SNR, BP_iter_num))
        para = np.loadtxt(para_data_file, np.float32)
        alpha = para[0,:]
        beta = para[1,:]
        
        y_recieve_file = format('%s_%.1f.dat' % (dec_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (dec_config.decoding_x_file, SNR))
        r_factor_file = format('%s_%.1f.dat' % (dec_config.decoding_r_file, SNR))
        factorio_decode = DataIO.FactorDataIO(r_factor_file, dec_config)
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, dec_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            r_factor = factorio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)      
            ch_noise = y_receive - np.multiply(s_mod, r_factor)
            LLR = y_receive
            ##practical noise 
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.quantized_decode(LLR.astype(np.float32), BP_iter_num[0], alpha, beta)
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

# Generate LDPC information data
def softsign(x_in):
    x_temp = x_in/(np.abs(x_in) + 0.0001)
    y_out = np.divide(1-x_temp, 2)
    return y_out

def sigmoid(x_in):
    y_out = 1/(1+np.exp(-x_in))
    return y_out
    
def Generate_AWGN_Training_Data(code, channel, train_config, generate_data_for):
    #initialized    
    SNR_set = train_config.SNR_set
    
    if generate_data_for == 'Training':
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
        batch_size = train_config.training_minibatch_size
        
    elif generate_data_for == 'Test':
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
        batch_size = train_config.test_minibatch_size
        
    else:
        print('Invalid objective of data generation!')
        exit(0)
    
    ## Data generating starts
    start = datetime.datetime.now()     
    for SNR in SNR_set:
        if generate_data_for == 'Training':
            fout_feature = open(format('%s_%.1f.dat' % (train_config.training_feature_file, SNR)), 'wb')
            fout_label = open(format('%s_%.1f.dat' % (train_config.training_label_file, SNR)), 'wb')
        elif generate_data_for == 'Test':
            fout_feature = open(format('%s_%.1f.dat' % (train_config.test_feature_file, SNR)), 'wb')
            fout_label = open(format('%s_%.1f.dat' % (train_config.test_label_file, SNR)), 'wb')
        for ik in range(0, total_batches):
            x_bits, u_coded_bits, s_mod, ch_noise, y_receive = Transmission.AWGN_transmission(SNR, batch_size, train_config, code, channel)
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_feature)  # write features to file
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_label)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    
def Generate_ACGN_Training_Data(code, channel, train_config, generate_data_for):
    #initialized    
    SNR_set = train_config.SNR_set
    
    if generate_data_for == 'Training':
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
        batch_size = train_config.training_minibatch_size
        
    elif generate_data_for == 'Test':
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
        batch_size = train_config.test_minibatch_size
        
    else:
        print('Invalid objective of data generation!')
        exit(0)
    
    ## Data generating starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        if generate_data_for == 'Training':
            fout_feature = open(format('%s_%.1f.dat' % (train_config.training_feature_file, SNR)), 'wb')
            fout_label = open(format('%s_%.1f.dat' % (train_config.training_label_file, SNR)), 'wb')
        elif generate_data_for == 'Test':
            fout_feature = open(format('%s_%.1f.dat' % (train_config.test_feature_file, SNR)), 'wb')
            fout_label = open(format('%s_%.1f.dat' % (train_config.test_label_file, SNR)), 'wb')
        for ik in range(0, total_batches):
            x_bits, u_coded_bits, s_mod, ch_noise, y_receive = Transmission.ACGN_transmission(SNR, batch_size, train_config, code, channel)
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_feature)  # write features to file
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_label)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    
def Generate_RLN_Training_Data(code, channel, train_config, generate_data_for):
    #initialized    
    SNR_set = train_config.SNR_set
    
    if generate_data_for == 'Training':
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
        batch_size = train_config.training_minibatch_size
        
    elif generate_data_for == 'Test':
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
        batch_size = train_config.test_minibatch_size
        
    else:
        print('Invalid objective of data generation!')
        exit(0)
    
    ## Data generating starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        if generate_data_for == 'Training':
            fout_feature = open(format('%s_%.1f.dat' % (train_config.training_feature_file, SNR)), 'wb')
            fout_label = open(format('%s_%.1f.dat' % (train_config.training_label_file, SNR)), 'wb')
            fout_factor = open(format('%s_%.1f.dat' % (train_config.training_factor_file, SNR)), 'wb')
        elif generate_data_for == 'Test':
            fout_feature = open(format('%s_%.1f.dat' % (train_config.test_feature_file, SNR)), 'wb')
            fout_label = open(format('%s_%.1f.dat' % (train_config.test_label_file, SNR)), 'wb')
            fout_factor = open(format('%s_%.1f.dat' % (train_config.test_factor_file, SNR)), 'wb')
        for ik in range(0, total_batches):
            x_bits, u_coded_bits, s_mod, ch_noise, y_receive, r_factor = Transmission.RLN_transmission(SNR, batch_size, train_config, code, channel)
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_feature)  # write features to file
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_label)
            r_factor = r_factor.astype(np.float32)
            r_factor.tofile(fout_factor)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")

def Generate_AWGN_Decoding_Data(gen_config, code):
    #initialized    
    SNR_set = gen_config.SNR_set
    total_samples = gen_config.total_samples
    batch_size = 5000
    K = gen_config.K_code
    N = gen_config.N_code
    rng = np.random.RandomState(None)
    total_batches = int(total_samples // (batch_size*K))
    ## Data generating starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (gen_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (gen_config.decoding_x_file, SNR))
        fout_yrecieve = open(y_recieve_file, 'wb')
        fout_xtransmit = open(x_transmit_file, 'wb')
        for ik in range(0, total_batches):
            x_bits = np.zeros((batch_size, K))
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            noise_awgn = rng.randn(batch_size, N)
            ch_noise_normalize = noise_awgn.astype(np.float32)
            ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
            ch_noise = ch_noise_normalize * ch_noise_sigma
            y_receive = s_mod + ch_noise
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_yrecieve)
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_xtransmit)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")

def Generate_ACGN_Decoding_Data(gen_config, code):
    #initialized    
    SNR_set = gen_config.SNR_set
    total_samples = gen_config.total_samples
    batch_size = 5000
    K = gen_config.K_code
    N = gen_config.N_code
    rng = np.random.RandomState(None)
    total_batches = int(total_samples // (batch_size*K))
    ## Data generating starts
    start = datetime.datetime.now()
    fin_cov_file = open(gen_config.cov_1_2_file , 'rb')
    cov_1_2_mat = np.fromfile(fin_cov_file, np.float32, N*N)
    cov_1_2_mat = np.reshape(cov_1_2_mat, [N, N])
    fin_cov_file.close()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (gen_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (gen_config.decoding_x_file, SNR))
        fout_yrecieve = open(y_recieve_file, 'wb')
        fout_xtransmit = open(x_transmit_file, 'wb')
        for ik in range(0, total_batches):
            x_bits = np.zeros((batch_size, K))
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            noise_awgn = rng.randn(batch_size, N)
            ch_noise_normalize = noise_awgn.astype(np.float32)
            ch_noise_normalize = np.matmul(ch_noise_normalize, cov_1_2_mat)
            ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
            ch_noise = ch_noise_normalize * ch_noise_sigma
            y_receive = s_mod + ch_noise
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_yrecieve)
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_xtransmit)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    
def Generate_RLN_Decoding_Data(gen_config, code):
    #initialized    
    SNR_set = gen_config.SNR_set
    total_samples = gen_config.total_samples
    batch_size = 5000
    K = gen_config.K_code
    N = gen_config.N_code
    rng = np.random.RandomState(None)
    total_batches = int(total_samples // (batch_size*K))
    ## Data generating starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (gen_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (gen_config.decoding_x_file, SNR))
        r_factor_file = format('%s_%.1f.dat' % (gen_config.decoding_r_file, SNR))
        fout_yrecieve = open(y_recieve_file, 'wb')
        fout_xtransmit = open(x_transmit_file, 'wb')
        fout_rfactor = open(r_factor_file, 'wb')
        for ik in range(0, total_batches):
            x_bits = np.zeros([batch_size, K])
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            noise_awgn = rng.randn(batch_size, N)
            ch_noise_normalize = noise_awgn.astype(np.float32)
            ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
            rayleigh_factor = np.sqrt(np.square(np.sqrt(1/2)*np.random.randn(batch_size, N))+np.square(np.sqrt(1/2)*np.random.randn(batch_size, N)))
            ch_noise = ch_noise_normalize * ch_noise_sigma
            y_receive = np.multiply(rayleigh_factor, s_mod) + ch_noise
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_yrecieve)
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_xtransmit)
            r_factor = rayleigh_factor.astype(np.float32)
            r_factor.tofile(fout_rfactor)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")