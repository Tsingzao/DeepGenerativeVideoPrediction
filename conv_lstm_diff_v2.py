import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

from keras.models import Model
from keras.layers import ConvLSTM2D, Input, merge, Convolution3D, Dense, MaxPooling3D, Flatten, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.activations import *
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
import time
import numpy as np
    
import h5py
h5file = h5py.File('/home/yutingzhao/CodeDemo/ConvLSTM/ConvLSTMDifference/ucf_aug_multi.h5','r') # mnist_aug_multi.h5','r') #
train_data = h5file['train_diff'][:]
train_fra  = h5file['train_fra'][:]
test_data  = h5file['test_data'][:]
h5file.close()
Cloud     = train_data.astype('float32')
Cloud_    = train_fra.astype('float32')
Noncloud  = test_data.astype('float32')
del train_data, train_fra, test_data
temp = []
for i in range(4):
    temp.append(Cloud_)
Cloud_ = np.asarray(temp, dtype='float32')
del temp
Cloud_ = Cloud_.transpose(1,0,2,3,4)
Cloud    /= 255
Cloud_   /= 255
Noncloud /= 255
	  	  
IMAGE_CHANNEL = 3

def trans_tensor(x):
    import tensorflow as tf
    return tf.transpose(x, perm=(0,4,1,2,3))

def generator():

    input1 = Input((4, 64, 64, IMAGE_CHANNEL))
    encode = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input1)
    # encode = LeakyReLU(0.2)(encode)
    encode = BatchNormalization(mode=2)(encode)
    encode = ConvLSTM2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(encode)
    # encode = LeakyReLU(0.2)(encode)
    encode = BatchNormalization(mode=2)(encode)
    encode = ConvLSTM2D(nb_filter=128, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(encode)
    # encode = LeakyReLU(0.2)(encode)
    encode = BatchNormalization(mode=2)(encode)
    encode = ConvLSTM2D(nb_filter=IMAGE_CHANNEL*4, nb_row=3, nb_col=3, border_mode='same', return_sequences=False)(encode)
    encode = Reshape((64, 64, IMAGE_CHANNEL, 4))(encode)
    encode = Lambda(trans_tensor, output_shape=(4,64,64,IMAGE_CHANNEL))(encode)
    input2 = Input((4, 64, 64, IMAGE_CHANNEL))
    output = merge([encode, input2], mode='sum')
    model  = Model([input1, input2], output)
    return model
    
def discriminator():
    input  = Input((4, 64, 64, IMAGE_CHANNEL))
    x = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(input)
    x = MaxPooling3D((1,2,2))(x)
    x = BatchNormalization(mode=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling3D((1,2,2))(x)
    x = BatchNormalization(mode=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling3D((1,2,2))(x)
    x = BatchNormalization(mode=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    y = Dense(2, activation='softmax')(x)
    model = Model(input, y)
    return model
    
def DCGAN(generator, discriminator_model):
    gen_input1 = Input((4, 64, 64, IMAGE_CHANNEL))
    gen_input2 = Input((4, 64, 64, IMAGE_CHANNEL))
    generated_image = generator([gen_input1, gen_input2])
    DCGAN_output = discriminator_model(generated_image)
    DCGAN = Model(input=[gen_input1, gen_input2], output=[generated_image, DCGAN_output])
    
    return DCGAN

def plot_loss(losses):  
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(losses["d"], label='adversarial loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()   
  
def plot_generated_batch(generator_model):
    import matplotlib.pyplot as plt
    index = np.arange(Cloud.shape[0])
    np.random.shuffle(index)
    index = list(np.sort(index[:10]))
    future_frames = generator_model.predict([Cloud[index],Cloud_[index]], batch_size=10, verbose=1)
    plt.figure(figsize=(20,20))
    t=0
    for i in index:
        ax = plt.subplot(10, 9, t*9+1)
        plt.imshow(Noncloud[i,0].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else Noncloud[i,0].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(10, 9, t*9+2)
        plt.imshow(Noncloud[i,1].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else Noncloud[i,1].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(10, 9, t*9+3)
        plt.imshow(Noncloud[i,2].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else Noncloud[i,2].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(10, 9, t*9+4)
        plt.imshow(Noncloud[i,3].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else Noncloud[i,3].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(10, 9, t*9+5)
        plt.imshow(future_frames[t,0].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else future_frames[t,0].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(10, 9, t*9+6)
        plt.imshow(future_frames[t,1].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else future_frames[t,1].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(10, 9, t*9+7)
        plt.imshow(future_frames[t,2].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else future_frames[t,2].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(10, 9, t*9+8)
        plt.imshow(future_frames[t,3].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else future_frames[t,3].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(10, 9, t*9+9)
        plt.imshow(Cloud_[i,0].reshape((64,64,IMAGE_CHANNEL)) if IMAGE_CHANNEL>1 else Cloud_[i,0].reshape((64,64)))
        plt.gray()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)
        t+=1
    plt.show()
     
batch_size = 16
###############################################################################
#    Both settings are proper for mnist, while the first is also proper for ucf
#    It seems that the second parameter can be used for segemtation 
###############################################################################
opt_dcgan = RMSprop() # seems satisfy while the reverse is bad
opt = SGD(lr=0.01)#Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
#==============================================================================
# opt_dcgan = SGD(lr=0.0001) # seems segement for ucf
# opt = Adadelta(lr=0.00001)
#==============================================================================

generator_model = generator()
generator_model.summary()
discriminator_model = discriminator()
discriminator_model.summary()
generator_model.compile(loss='mae', optimizer=opt)
discriminator_model.trainable = False

DCGAN_model = DCGAN(generator_model, discriminator_model)
DCGAN_model.summary()
loss = ['mae', 'binary_crossentropy']
loss_weights = [1, 1e2]
DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

discriminator_model.trainable = True
discriminator_model.compile(loss='binary_crossentropy', optimizer=opt)
gen_loss = 100
disc_loss = 100

#-----------------------------------------------------------------------------#
print("Start training")
nb_epoch = 400
n_batch_per_epoch = Cloud.shape[0] // batch_size
losses = {"d":[], "g":[]}
for e in range(nb_epoch):
    batch_counter = 1
    start = time.time()
    index = np.arange(Cloud.shape[0])
    np.random.shuffle(index)
    for i in range(0,n_batch_per_epoch,2):
        index_dis = list(np.sort(index[i*batch_size:(i+1)*batch_size]))
        cloud_batch = Cloud[index_dis]
        cloud_batch_= Cloud_[index_dis]
        noncloud_batch = Noncloud[index_dis]
        if batch_counter % 2 == 0:
            X_disc = generator_model.predict([cloud_batch, cloud_batch_])
            y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
            y_disc[:, 0] = 1
        else:
            X_disc = noncloud_batch
            y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
            
        # Update the discriminator
        disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
        losses["d"].append(disc_loss)
        print(disc_loss)
        
        # Create a batch to feed the generator model
        index_gen = list(np.sort(index[(i+1)*batch_size:(i+2)*batch_size]))
        X_gen_target = Noncloud[index_gen]
        X_gen = Cloud[index_gen]
        X_gen_= Cloud_[index_gen]
        y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1

        # Freeze the discriminator
        discriminator_model.trainable = False
        gen_loss = DCGAN_model.train_on_batch([X_gen,X_gen_], [X_gen_target, y_gen])
        losses["g"].append(gen_loss[0])
        print('G_tot = %f, G_L1 = %f, G_logloss = %f'%(gen_loss[0], gen_loss[1], gen_loss[2]))
        # Unfreeze the discriminator
        discriminator_model.trainable = True
        batch_counter += 1
        print(batch_counter)
        plot_loss(losses)
        if batch_counter % 200 == 0:
            plot_generated_batch(generator_model)                       
                 
    print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
