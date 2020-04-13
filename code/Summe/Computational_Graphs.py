import numpy as np
from keras.layers import BatchNormalization, Dense, Conv3D, Conv2D, Conv1D, Dropout, MaxPooling3D, MaxPooling1D, Flatten, Activation, Input, Reshape, Conv3DTranspose, LSTM, GRU, UpSampling3D, TimeDistributed, Bidirectional
from keras.models import Sequential, Model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the random seeds
tf.set_random_seed(101)
np.random.seed(101)
        

class Computational_Graphs(object):
    
    def getModelWeights(self, model):
        for layer in model.layers:
            print("{}, {}".format(layer.name, layer.get_weights()))
            
    # Test  this to see if it works             
    def plot_conv_weights(self, model, layer):
        W = model.get_layer(name=layer).get_weights()[0][0]
        print W.shape
        if len(W.shape) == 4:
            print W.shape
            W = np.squeeze(W)
            print W.shape
            W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
            print W.shape
            fig, axs = plt.subplots(5,5, figsize=(8,8))
            fig.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()
            for i in range(25):
                axs[i].imshow(W[:,:,i])
                axs[i].set_title(str(i))
            
#    def simpleNeuralNetwork(self):
    
    # This graph takes input of size - (5, 112, 112, 3) i.e 5 images per cube that has 3 channels 
    def simple3DCNN_fiveFrameTargets(self, training_X):
        
        model = Sequential()
        model.add(Conv3D(filters=16, 
                         kernel_size=(3, 3, 3),
                         activation='relu', 
                         kernel_initializer='he_normal',
                         padding='same',
                         input_shape=(training_X.shape[1], training_X.shape[2], training_X.shape[3], training_X.shape[4])))                     
        model.add(MaxPooling3D())
        model.add(Conv3D(filters=8,
                         kernel_size=(3, 3, 3),
                         kernel_initializer='he_normal',
                         activation='relu',
                         padding='same'))
        model.add(MaxPooling3D())
        model.add(Flatten())
        model.add(Dense(128)) 
#                        kernel_initializer='he_normal',
#                        bias_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dropout(rate=0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        return model
        
        
    def eval_simpleCNN(self, training_X, training_Y, num_epochs=10, num_batch_size=8):
        model = self.simple3DCNN_fiveFrameTargets(training_X)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(training_X, training_Y, epochs=num_epochs, batch_size=num_batch_size)
    
        return model
        
        
    #def comp_graph():
    #    m = Sequential()
    #    m.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=(5, 112, 112, 3)))
    #    m.add(MaxPooling3D())
    #    m.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
    #    m.add(MaxPooling3D())
    #    m.add(Flatten())
    #    m.add(Dense(128, activation='relu'))
    #    m.add(Dense(1, activation='relu'))
    #    
    #    return m        
    #
    #def eval_model(training_X, training_Y):
    #    m = comp_graph()
    #    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #    m.fit(training_X, training_Y, epochs=3, batch_size=32)
    #    
    #    return m

    # This model take in input of shape (5,112,112,3) i.e 5 images per cube that has 3 channels 


    def Conv_3D_AE(self, training_X):

        input_video = Input(shape=(5, 112, 112, 3))
        encoded = Conv3D(32, (3, 3, 3), padding='same')(input_video)
        encoded = MaxPooling3D((1, 2, 2))(encoded)
        encoded = Conv3D(16, (3, 3, 3), padding='same')(encoded)
        
        latent = Conv3D(8, (3, 3, 3), padding='same')(encoded)
        
        decode_l1 = Conv3D(16, (3, 3, 3), padding='same')
        decode_l2 = UpSampling3D((1, 2, 2))
        decode_l3 = Conv3D(32, (3, 3, 3), padding='same')
        output_l = Conv3D(3, (3, 3, 3), padding='same')
        
        decoded = decode_l1(latent)
        decoded = decode_l2(decoded)
        decoded = decode_l3(decoded)
        decoded = output_l(decoded)
        
        ae = Model(input_video, decoded)
        ae.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        encoder = Model(input_video, latent)
        
        ae.fit(training_X, training_X, batch_size=2, epochs=3)

        return ae, encoder   


    def simple_Convolutional_AutoEncoder(self, training_X):
        encoding_dim = 512
        filters = 8
        kernel_size = 3
        strides = 1
        input_shape = (5, 112, 112, 3)
        inputs = Input(shape=input_shape)
        encoded = Conv3D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         kernel_initializer='he_normal',
                         activation='relu',
                         padding='same')(inputs)
                
        # shape info needed to build decoder model
        shape = K.int_shape(encoded)
        print shape
        
        encoded = Flatten()(encoded)
#        encoded = Dense(2048, activation='relu')(encoded)

        latent_nodes = Dense(encoding_dim)(encoded)
        
        encoder = Model(inputs, latent_nodes, name='encoder')
        
        decoded = Dense(shape[1]*shape[2]*shape[3]*shape[4], activation='relu')(latent_nodes)
        print decoded.shape
        decoded = Reshape((shape[1], shape[2], shape[3], shape[4]))(decoded)
        
        decoded = Conv3DTranspose(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  kernel_initializer='he_normal',
                                  activation='relu',
                                  padding='same')(decoded)
    
        outputs = Conv3DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(decoded)
        
        
#        decoder = Model(latent_nodes, outputs, name = 'decoder')        
        autoencoder = Model(inputs, outputs)
        print autoencoder.summary()
        model = self.eval_Conv_Autoencoder(training_X, autoencoder)
        encoder.predict
        
        return model, encoder#, decoder
        
    def eval_Conv_Autoencoder(self, training_X, model, num_epochs=3, num_batch_size=2):
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(training_X, training_X, epochs=num_epochs, batch_size=num_batch_size)
    
        return model
        

    def GRU_RNN(self, training_X, training_Y, num_batch_size=1, num_epochs=3):
#        input_dims = Input(shape=(5,112,112,3))
        num_samples = training_X.shape[0]
        training_X = training_X.reshape(num_samples, 5, 37632)
        model = Sequential()
        model.add(GRU(6, 
                      input_shape=(5, 37632),
                      return_sequences=True,
                      activation='tanh', 
                      recurrent_activation='hard_sigmoid', 
                      dropout=0.1,
                      recurrent_dropout=0.1))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(training_X, training_Y, epochs=num_epochs, batch_size=num_batch_size)
        
        return model
        
    def simpleNeuralNetwork(self, trainingX, trainingY, num_epochs=3, num_batch=16):
        input_dims = trainingX.shape[2]
        
        model = Sequential()
        model.add(Conv1D(32, 3, input_shape=(1, input_dims), padding='same'))
#        model.add(Conv1D(32, 3, padding='same'))
        
        model.add(Bidirectional(LSTM(32, 
#                      input_shape=(1, input_dims),
                      return_sequences=True,
                      activation='tanh', 
                      recurrent_activation='hard_sigmoid', 
                      dropout=0.1,
                      recurrent_dropout=0.1)))
                      
#        model.add(Bidirectional(LSTM(16, 
#                      return_sequences=True,
#                      activation='tanh', 
#                      recurrent_activation='hard_sigmoid', 
#                      dropout=0.1,
#                      recurrent_dropout=0.1)))
        
        model.add(Flatten())        
        
#        model.add(Dense(units=2048, kernel_initializer='he_normal', bias_initializer='random_normal'))
#        model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        model.add(Dropout(0.4))        
#        
        model.add(Dense(units=1024, kernel_initializer='he_normal', bias_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(units=512, kernel_initializer='he_normal', bias_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(units=128, kernel_initializer='he_normal', bias_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(units=64, kernel_initializer='he_normal', bias_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainingX, trainingY, epochs=num_epochs, batch_size=num_batch)        
        
        return model
        
        
        
        
        
        
                
                
        
        
        
        
        
        
        