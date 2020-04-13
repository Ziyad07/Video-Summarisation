# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:25:37 2018

@author: ziyad
"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from Prepare_Data import Prepare_Data
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Lambda
from sklearn.metrics import classification_report
from keras.layers import LSTM
import numpy as np
from sklearn.metrics import average_precision_score
from keras.losses import mse, binary_crossentropy
from keras import backend as K
get_data = Prepare_Data()

#train_x, train_y, test_x, _ = get_data.read_data()
#listOfTestingVideos, listOftargets = get_data.get_training_data_mAP()

train_x, train_y, test_x, _ = get_data.read_c3d_hog_data()
listOfTestingVideos, listOftargets = get_data.get_testing_data_c3d_hog()

from keras.layers import RepeatVector


input_dimensions = 7232 #c3d with hog
#input_dimensions = 4096 #c3d only


# network parameters
intermediate_dim = 512
batch_size = 32
latent_dim = 2
epochs = 5

def seq2seqAutoEncoder():
    timesteps = 1
    latent_dim = 32
    inputs = Input(shape=(1, input_dimensions))
    encoded = LSTM(latent_dim)(inputs)
    
    decoded =  RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dimensions, return_sequences=True)(decoded)
    
    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    
    return sequence_autoencoder


def simpleAutoEncoder():
    encoding_dim = 32
    # this is our input placeholder
    input_frame = Input(shape=(input_dimensions,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_frame)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dimensions, activation='sigmoid')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_frame, decoded)
    
    # this model maps an input to its encoded representation
    encoder = Model(input_frame, encoded)
    
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    
    return autoencoder
    
def deepAutoEncoder():
        
    input_img = Input(shape=(input_dimensions,))
    encoded = Dense(1024, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(input_dimensions, activation='sigmoid')(decoded)
    autoencoder = Model(input_img, decoded)
    
    return autoencoder

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def VAE():
        
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(input_dimensions,), name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#    encoder.summary()
    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(input_dimensions, activation='sigmoid')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
#    decoder.summary()
    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    vae.summary()
    
    return vae

def vae_loss(x, x_decoded_mean):
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss
    
    
def compileVAE(vae, train_x):
    vae.compile(optimizer='adam', loss=vae_loss)
    vae.fit(train_x, train_x,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)
        

##autoencoder = simpleAutoEncoder()
#autoencoder = deepAutoEncoder()
#
#
#
#time_steps = 1
##Since this is a RNN structure we will have to reshape the inputs to include a timestep as well     
##Dimensions are of size (samples, features) and need to be in size (samples, timesteps, features)
##train_x = np.reshape(train_x, (train_x.shape[0], time_steps, train_x.shape[1]))    
##autoencoder = seq2seqAutoEncoder()
#
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#
#autoencoder.fit(train_x, train_x,
#                epochs=5,
#                batch_size=32,
#                shuffle=True)

autoencoder = VAE()
compileVAE(autoencoder, train_x)


def expandTargetsVector(partitioned_targets):
    expanded_targets = []
    for snippet in range(len(partitioned_targets)):
        if partitioned_targets[snippet] > 0:
            for j in range(16):
                expanded_targets.append(partitioned_targets[snippet])
        else:
            for j in range(16):
                expanded_targets.append(partitioned_targets[snippet])

    return expanded_targets
    
    
def poisson_error(predicted, actual):
    error = predicted - actual * np.log(predicted)
    
    return np.sum(error,axis=1) / predicted.shape[1]
    
def L2_error(predicted, actual):
    recon_error = (predicted- actual)**2
    summed_error = np.sum(recon_error, axis=1)
    
    return summed_error
    
    
import evaluate_metrics as em
evas = em.evaluate_metrics()            
for i in range(len(listOfTestingVideos)):    
    testVideo = listOfTestingVideos[i]
    videoName = testVideo [0].split('.')[0]
    features = testVideo[1]
    preds = autoencoder.predict(features)
    
#    recon_error = (preds - features)**2
#    summed_error = np.sum(recon_error, axis=1)
  
#    error = L2_error(preds, features)
    error = poisson_error(preds, features) * 100.0
  
    targets = expandTargetsVector(error)
    
    print videoName
    evas.getSumMeEvaluationMetrics(targets, videoName)
    
    

    
#keeptop = map(lambda q: (round(q) if (q >= np.percentile(summed_error,85)) else 0),summed_error)

