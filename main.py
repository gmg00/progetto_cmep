import os.path
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from math import *
# Fixing random state for reproducibility
np.random.seed(123)

def get_file(filename):
    '''Get matrix from file 'filename'.
    '''
    return np.load(filename)

def print_cov(cov):
    '''Print covariance matrix from the array of 15 values.
    '''
    print(cov[0:5])
    print((cov[1],cov[5:9]))
    print((cov[2], cov[6], cov[9:12]))
    print((cov[3], cov[7], cov[10], cov[12:14]))
    print((cov[4], cov[8], cov[11], cov[13], cov[14]))

def hist_res(res, n_bins=50, title='Residuals distribution', x_label='x label',
             y_label='Residuals'):
    '''Take residuals and plot histogram. 
    '''
    plt.hist(res, n_bins)
    plt.title(title, size=15)
    plt.xlabel(x_label, size=12)
    plt.ylabel(y_label, size=12)
    plt.grid()
    plt.show()

def plot_loss(loss, val_loss):
    plt.figure(figsize=(11, 6))
    plt.plot(loss, label='Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs', size=12)
    #x = [0,100,200,300,400,500,600,700,800,900,1000]
    plt.xticks(size=12)
    plt.ylabel('Loss functions', size=12)
    plt.yticks(size=12)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.title('Loss vs Validation Loss', size=15)
    
class Log:
    def __init__(self, description, model, train_settings, training_time, loss, val_loss):
        self.description = description
        self.model = model
        self.train_settings = train_settings
        self.training_time = training_time 
        self.loss = loss
        self.val_loss = val_loss

def print_log(obj, only_last = False):
    print(f'There are {len(obj)} objects.')
    i = 0
    for o in obj:
        print(f'Object n. {i}----------------------------------------')
        print(o.description)
        print('Model summary:')
        o.model.summary()
        print(o.train_settings)
        print(f'Final loss: {o.loss}')
        print(f'Final val_loss: {o.val_loss}')
        plot_loss(o.loss,o.val_loss)
        print(f'Training time: {o.training_time}')
        print(f'End object n. {i}------------------------------------')

def save_instance(filename, inst):
    try:
        if not os.path.exists(filename):
            arr = []
            arr.append(inst)
            with open(filename, 'xb') as inst_file:
                pickle.dump(arr, inst_file)
        else:
            with open(filename, 'rb') as inst_file:
                arr = pickle.load(inst_file)
            arr.append(inst)
            with open(filename, 'wb') as inst_file:
                pickle.dump(arr, inst_file)
    except: print("Error in save_instance.")

def generator(X1_data, X2_data, batch_size):

    samples_per_epoch = X1_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    ids = np.arange(len(cov_train))
    counter=0
    X1_batch = []
    X2_batch = []
    while 1:
        if counter == 0:
            np.random.shuffle(ids)
            X1_data = X1_data[ids]
            X2_data = X2_data[ids]
        X1_batch = np.array(X1_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X2_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        #np.random.shuffle(X2_batch) #!!!!!!!!!!!!
        counter += 1
        yield [X1_batch,X2_batch],X1_batch

        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

def predict_generator(X1_data, X2_data, batch_size):
    counter=0
    X1_batch = []
    X2_batch = []
    while 1:
        X1_batch = np.array(X1_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X2_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        counter += 1
        yield [X1_batch, X2_batch],1

def my_metric(y_true, y_pred):
    '''Computes loss = mean(sqrt(y_true - y_pred)**2 / y_true), axis=-1)
    '''
    
    #temp = tf.math.xdivy(tf.math.sqrt(tf.square(y_true-y_pred)),tf.square(y_true))
    index = [0,5,9,12,14]
    y_true = tf.gather(y_true, indices=index)
    y_pred = tf.gather(y_pred, indices=index)
    temp = tf.math.sqrt(tf.math.xdivy(tf.square(y_true-y_pred),tf.square(y_true)))

    return tf.reduce_mean(temp, axis=-1) 

def my_metric2(y_true, y_pred):
    '''Computes loss = mean(abs((y_true - y_pred) / y_true)), axis=-1)
    '''
    
    temp = tf.math.sqrt(tf.math.xdivy(tf.square(y_true-y_pred),tf.square(y_true)))

    return tf.reduce_mean(temp, axis=-1) 

def my_metric3(y_true, y_pred):
    '''Computes loss = mean(abs((y_true - y_pred) / y_true)), axis=-1)
    '''
    epsilon = 10**(-4)
    index = [0,5,9,12,14]
    y_true = tf.gather(y_true, indices=index)
    y_pred = tf.gather(y_pred, indices=index)
    temp = tf.math.sqrt(tf.math.xdivy(tf.square(y_true-y_pred),tf.square(y_true + epsilon)))

    return tf.reduce_mean(temp, axis=-1) 

 

#---------------------------------------------------------------------MAIN
if __name__ == "__main__":

    dir_name = '/mnt/c/Users/HP/Desktop/progetto_cmep'
    model_name = 'enc8'

    #Get covariance matrix.
    filename = dir_name + '/data/inputs/cov_tot.npy'
    cov = get_file(filename)

    print(f'Dataset size: {cov.size}')
    print(f'Dataset shape: {cov.shape}')

    #Get parameters for conditioning.
    filename = dir_name + '/data/inputs/par_tot.npy'
    par = get_file(filename)

    print(f'Dataset size: {par.size}')
    print(f'Dataset shape: {par.shape}')

    #Normalize dataset
    #cov = cov/np.max(cov)
    #par = par/np.max(par)
    #Standardize dataset
    scaler_cov = preprocessing.Normalizer().fit(cov)
    scaler_par = preprocessing.Normalizer().fit(par)

    #cov = scaler_cov.transform(cov)
    #par = scaler_par.transform(par)

    print('Mean of each column in matric covariance dataset:')
    print(cov.mean(axis=0))

    print('Standard deviation of each column in matric covariance dataset:')
    print(cov.std(axis=0))

    #Split dataset in training and test
    cov_train0, cov_test0, par_train0, par_test0 = train_test_split(cov, par, test_size=0.5, random_state=42)
    print(f'cov_train shape = {cov_train0.shape}')
    print(f'par_train shape = {par_train0.shape}')

    cov_train = scaler_cov.transform(cov_train0)
    cov_test = scaler_cov.transform(cov_test0)
    par_train = scaler_par.transform(par_train0)
    par_test = scaler_par.transform(par_test0)


    #Get normalization factors for every row.
    scal = []
    for i in range(cov_test.shape[0]):
        scal.append(cov_test0[i,0]/cov_test[i,0])
        #if i%10000 == 0:
            #print(i)

    np.save(dir_name + '/data/outputs/scal.npy', np.array(scal))
    

    ##Create an autoencoder model.
    #Input data.
    input_data = Input(shape=(15,))  
    #Input parameters for conditioning.
    input_params = Input(shape=(5,))

    #Encoder.
    concat = Concatenate()([input_data, input_params])

    hidden = Dense(300,activation='relu')(concat)

    hidden = Dense(200,activation='relu')(hidden)

    hidden = Dense(100,activation='relu')(hidden)

    #hidden = BatchNormalization()(hidden)

    code=Dense(8,activation='sigmoid')(hidden) #Encoded.

    encodedstate = Activation('linear',dtype='float16')(code)

    #Decoder.
    concat = Concatenate()([encodedstate, input_params])

    hidden = Dense(100,activation='relu')(concat)

    hidden = Dense(200,activation='relu')(hidden)

    hidden = Dense(300, activation='relu')(hidden)

    
    #Output.
    outputs = Dense(15, activation='linear')(hidden)
    

    model = Model(inputs=[input_data, input_params], outputs=outputs)
    model.compile(loss='MSE', optimizer='adam', metrics=[my_metric, my_metric2, my_metric3])
    model.summary()
    #tf.keras.utils.plot_model(model, "/mnt/c/Users/HP/Desktop/history/model_enc8.png",show_shapes=True)
    
    
    #Dictionary for training settings
    dt = {
        "validation_split" : 0.5,
        "epochs" : 1000,
        "batch_size" : 200,
        #Early stopping settings.
        "EarlyStopping_monitor" : "val_loss",
        "EarlyStopping_patience" : 65,
        #Reduce learning rate on plateau settings.
        "ReduceLROnPlateau_monitor" : "val_loss",
        "ReduceLROnPlateau_factor" : 0.25,
        "ReduceLROnPlateau_patience" : 30
        }

    ##Training.
    
    #Split cov_train and par_train in train and validation datasets.
    cov_train, cov_val, par_train, par_val = train_test_split(cov_train, par_train, test_size=0.5, random_state=42)

    print(f'len/batch_size = {cov_train.shape[0]//dt["batch_size"]}')

    input("Press a button to start training")
    
    gen = generator(cov_train,par_train,dt['batch_size'])
    print(type(gen))
    history = model.fit(
        gen,
        epochs=dt['epochs'],
        steps_per_epoch = cov_train.shape[0]//dt['batch_size'],
        validation_data = generator(cov_val,par_val,dt['batch_size']),
        validation_steps = cov_val.shape[0]//dt['batch_size'],
        #use_multiprocessing = True,
        callbacks = [EarlyStopping(monitor=dt["EarlyStopping_monitor"], patience=dt["EarlyStopping_patience"], verbose=1),
                     ReduceLROnPlateau(monitor=dt["ReduceLROnPlateau_monitor"], factor=dt["ReduceLROnPlateau_factor"],
                                       patience=dt["ReduceLROnPlateau_patience"], verbose=1)])
    
    #history = model.fit([cov_train, par_train], cov_train, validation_split=0.5, epochs=2, verbose=1, batch_size=128)
    ##Plot loss.
    plot_loss(history.history['loss'], history.history['val_loss'])
    plt.savefig(dir_name + '/data/models/loss_model_enc8.png')
    plt.show()
    
    plt.figure(12)
    plt.plot(history.history['my_metric'], label='my_metric')
    plt.plot(history.history['val_my_metric'], label='val_my_metric')
    plt.xlabel('Epochs', size=12)
    plt.xticks(size=12)
    plt.ylabel('Metrics', size=12)
    plt.yticks(size=12)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.title('My metric', size=15)
    plt.show()

    plt.figure(13)
    plt.plot(history.history['my_metric2'], label='my_metric2')
    plt.plot(history.history['val_my_metric2'], label='val_my_metric2')
    plt.xlabel('Epochs', size=12)
    plt.xticks(size=12)
    plt.ylabel('Metrics', size=12)
    plt.yticks(size=12)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.title('My metric2', size=15)
    plt.show()

    input("Press a button to save history")
    with open(dir_name + '/data/histories/history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    input("Press a button to continue")

    ##Encoder-decoder division.
    encoder = Model(inputs=[input_data, input_params], outputs=encodedstate)
    decoder = Model(inputs=[encodedstate, input_params] , outputs=outputs)
    print("Encoder predictions")
    cov_enc = encoder.predict(predict_generator(cov_test,par_test,dt['batch_size']),
                             steps = cov_test.shape[0]//dt['batch_size']+1,
                             verbose = 1)
    
    np.save(dir_name + '/data/outputs/cov_enc_' + model_name + '.npy', cov_enc)

    input("Press a button to continue")
   

    ##Test
    
    print(f'cov_test shape = {cov_test.shape}')
    #cov_pred = model.predict([cov_test, par_test],batch_size=1)
    
    cov_pred = model.predict(predict_generator(cov_test,par_test,dt['batch_size']),
                             steps = cov_test.shape[0]//dt['batch_size']+1,
                             verbose = 1)
    

    np.save(dir_name + '/data/outputs/cov_pred_' + model_name + '.npy', cov_pred)

    np.save(dir_name + '/data/outputs/cov_test_' + model_name + '.npy', cov_test)

    print(cov_test.shape)
    print(cov_pred.shape)
    
    
