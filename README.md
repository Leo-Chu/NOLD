# NOLD
Here we provide codes fo papers：

H. He, Lei Chu, Robert Qiu; NOLD: A Neural-Network Optimized Low-Resolution Decoder for LDPC Codes， Submitted for an IEEE journal;


Code Specifications:

Requirements
Numpy>=1.13.3, Python >= 3.6, tensorflow>=1.10.0

The codes include 5 modules to carry out simulations in our papers. They are "GenDecodingData", "GenTrainingData", "Floating MS", "Quantized MS" and "NeuralOptimization"

The configuration entrances of these modules are in "Configuration.py". Each module can be operated under different channels by configurating the channel parameters. 
Other details are as follows:
    
GenDecodingData:
    This module is executed by setting "GenConfiure" in "Configuration.py". 
    The outputs of this module are two files containing transmitting signals data and receiving signals data in "./Decoding".
    These datas are used to execute the MS decoing algorithm.

GenTrainingData
    This module is executed by setting "TrainingConfiure" in "Configuration.py".  
    The outputs of this module are two files containing training data set and validation data set in "./TrainingData" and "./TestData".
    The training data set is used in training process and the validation data set is to test the network performance during training.
    
Floating MS
    This module is executed by setting "DecConfiure" in "Configuration.py".
    The outputs of theis module is a BER result in "./results"
    
NeuralOptimization
    This module is executed by setting "TrainingConfiure" in "Configuration.py".  
    The theis module are a series of files contain the training scaling and quantization parameters for each iteration  in "./parameters"

Quantized MS
    This module is executed by setting "DecConfiure" in "Configuration.py".
    The outputs of this module is a BER result in "./results"
    This module uses the scaling and quantization parameters in  "./parameters"
    
Noise setting
    In each module, the channel is selected by setting "channel" as "AWGN", "ACGN", "RLN" in configuration. 
    Among three noise, the "ACGN" needs the correlated parameters to be set.

The codes of this paper are partially based on the codes of  <An Iterative BP-CNN Architecture for
Channel Decoding>(https://github.com/liangfei-info/Iterative-BP-CNN). 
In our codes，the MS decoding module is derived form the SP decoding module of Liang's code. 
And the Neural optimization Module is constructed based on the NN module in Liang's work.
We appreciate <An Iterative BP-CNN Architecture for Channel Decoding>'s codes and thank the authors
 of <An Iterative BP-CNN Architecture for Channel Decoding>.

