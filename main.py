import os.path
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate
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
    plt.grid()

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
        yield [X1_batch, X2_batch],

 

#---------------------------------------------------------------------MAIN
if __name__ == "__main__":

    #Get covariance matrix.
    filename = '/mnt/c/Users/HP/Desktop/cov_tot.npy'
    cov = get_file(filename)

    print(f'Dataset size: {cov.size}')
    print(f'Dataset shape: {cov.shape}')

    #Get parameters for conditioning.
    filename = '/mnt/c/Users/HP/Desktop/par_tot.npy'
    par = get_file(filename)

    print(f'Dataset size: {par.size}')
    print(f'Dataset shape: {par.shape}')

    #Normalize dataset
    #cov = cov/np.max(cov)
    #par = par/np.max(par)
    #Standardize dataset
    scaler_cov = preprocessing.Normalizer().fit(cov)
    scaler_par = preprocessing.Normalizer().fit(par)

    cov = scaler_cov.transform(cov)
    par = scaler_par.transform(par)

    print('Mean of each column in matric covariance dataset:')
    print(cov.mean(axis=0))

    print('Standard deviation of each column in matric covariance dataset:')
    print(cov.std(axis=0))

    #Split dataset in training and test
    cov_train, cov_test, par_train, par_test = train_test_split(cov, par, test_size=0.5, random_state=42)
    print(f'cov_train shape = {cov_train.shape}')
    print(f'par_train shape = {par_train.shape}')


    ##Create an autoencoder model.
    #Input data.
    input_data = Input(shape=(15,))  
    #Input parameters for conditioning.
    input_params = Input(shape=(5,))

    '''
    #-----------------------------------MODEL TESTED -> OVERFITTING -> ?Too complex, too many layers?
    hidden = Dense(150, activation='relu')(input_data)
    concat = Concatenate()([hidden,input_params])
    hidden = Dense(100, activation='relu')(concat)
    hidden2 = Dense(50, activation='relu')(input_params)
    concat = Concatenate()([hidden,hidden2])
    hidden = Dense(50, activation='relu')(concat)

    hidden = BatchNormalization()(hidden)

    code = Dense(5, activation='sigmoid')(hidden)

    hidden = Dense(50, activation='relu')(code)
    hidden2 = Dense(50, activation='relu')(input_params)
    concat = Concatenate()([hidden, hidden2])
    hidden = Dense(100, activation='relu')(concat)
    concat = Concatenate()([hidden,input_params])
    hidden = Dense(150, activation='relu')(concat)

    outputs = Dense(15, activation='linear')(hidden)
    '''


    
    #-----------------------------------MODEL TESTED -> Val_loss can't go under 0.2
    #Encoder.
    concat = Concatenate()([input_data, input_params])

    hidden = Dense(300,activation='relu')(concat)

    hidden = Dense(200,activation='relu')(hidden)
    #hidden = Dropout(0.2)(hidden)

    hidden = Dense(100,activation='relu')(hidden)

    #hidden = BatchNormalization()(hidden)

    code=Dense(5,activation='sigmoid')(hidden) #Encoded.

    #Decoder.
    concat = Concatenate()([code, input_params])

    hidden = Dense(100,activation='relu')(concat)

    hidden = Dense(200,activation='relu')(hidden)
    #hidden = Dropout(0.2)(hidden)

    hidden = Dense(300,activation='relu')(hidden)

    
    #Output.
    outputs = Dense(15, activation='linear')(hidden)
    
    '''
    --------------------------------------MODEL TESTED -> not better than the last
      #Encoder.
    #hidden1 = Dense(50,activation='relu')(input_data)
    #hidden2 = Dense(50,activation='relu')(input_params)
    concat = Concatenate()([input_data, input_params])

    hidden = Dense(150,activation='relu')(concat)

    hidden = Dense(100,activation='relu')(hidden)

    hidden = Dense(50,activation='relu')(hidden)

    hidden = BatchNormalization()(hidden)

    code=Dense(5,activation='sigmoid')(hidden) #Encoded.

    #Decoder.
    #concat = Concatenate()([code, input_params])

    hidden = Dense(50,activation='relu')(code)

    hidden = Dense(100,activation='relu')(hidden)

    hidden = Dense(150,activation='relu')(hidden)

    concat = Concatenate()([hidden, input_params])

    #Output.
    outputs = Dense(15, activation='linear')(concat)
    '''
    '''
    #Encoder.
    #hidden = Dense(25, activation='relu')(input_data)
    #hidden1 = Dense(25, activation='relu')(input_params)
    concat = Concatenate()([input_data, input_params])

    hidden = Dense(150,activation='relu')(concat)

    #hidden = Dense(100,activation='relu')(hidden)

    hidden = Dense(50,activation='relu')(hidden)

    hidden = BatchNormalization()(hidden)

    code=Dense(5,activation='sigmoid')(hidden) #Encoded.

    #Decoder.
    #hidden1 = Dense(25, activation='relu')(input_params)
    concat = Concatenate()([code, input_params])

    hidden = Dense(50,activation='relu')(concat)

    #hidden = Dense(100,activation='relu')(hidden)

    hidden = Dense(150,activation='relu')(hidden)
    hidden = Dropout(0.2)(hidden)

    #Output.
    outputs = Dense(15, activation='linear')(hidden)
    '''

    model = Model(inputs=[input_data, input_params], outputs=outputs)
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    tf.keras.utils.plot_model(model, "/mnt/c/Users/HP/Desktop/history/model2_norm4.png",show_shapes=True)
    
    
    #Dictionary for training settings
    dt = {
        "validation_split" : 0.5,
        "epochs" : 500,
        "batch_size" : 200,
        #Early stopping settings.
        "EarlyStopping_monitor" : "val_loss",
        "EarlyStopping_patience" : 150,
        #Reduce learning rate on plateau settings.
        "ReduceLROnPlateau_monitor" : "val_loss",
        "ReduceLROnPlateau_factor" : 0.25,
        "ReduceLROnPlateau_patience" : 50
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

    ##Plot loss.
    plot_loss(history.history['loss'], history.history['val_loss'])
    plt.savefig('/mnt/c/Users/HP/Desktop/history/loss_model2_norm4.png')
    plt.show()

    input("Press a button to save history")
    with open('/mnt/c/Users/HP/Desktop/history/history_model2_norm4', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    input("Press a button to continue")

    '''
    ##Save instance in log.
    inp = input("Press 'y' if you want to save this instance.")
    if inp == "y":
        description = input("Insert a description for this instance:")
        inst = Log(description, model, dt, elapsed_time, history.history['loss'], history.history['val_loss'])
        save_instance("log", inst)
        print("Instance saved.")
    else: print("Instance not saved.")

    with open("log", 'rb') as inst_file:
        arr = pickle.load(inst_file)
    print_log(arr)

    ##Encoder-decoder division.
    encoder = Model(inputs=[input_data, input_params], outputs=code)
    decoder = Model(inputs=[code, input_params] , outputs=outputs)
    print("Prediction of ")
    encoded_cov = encoder.predict([cov_test[0:2, :], par_test[0:2, :]])
    print(encoded_cov)
    print(f'nbytes encoded matrix: {encoded_cov.nbytes}')
    print(f'nbytes original matrix: {cov_test[0:2,:].nbytes}')

    input("Press a button to continue")
    '''

    ##Test
    #cov_test, A, par_test, B = train_test_split(cov_test, par_test, test_size=0.95, random_state=42)
    
    #test_data = cov_test[0:1*10**6, :]
    #test_params = par_test[0:1*10**6, :]
    #cov_pred = model.predict([cov_test, par_test])
    cov_pred = model.predict(predict_generator(cov_test,par_test,dt['batch_size']),
                             steps = cov_test.shape[0]//dt['batch_size']+1,
                             verbose = 1)
    print('save?')
    np.save('/mnt/c/Users/HP/Desktop/cov_pred3.npy',cov_pred)
    print('saved')
    np.save('/mnt/c/Users/HP/Desktop/cov_test3.npy',cov_test)

    print(cov_test.shape)
    print(cov_pred.shape)
    
    index = [0, 5, 9, 12, 14]
    titles = ['qoverp', 'lambda', 'phi', 'dxy', 'dsz']
    # Da applicare taglio
    for i, elem in enumerate(index):
        print(f'i = {elem}')
        #Calculate residuals.
        res = (cov_test[:, elem] - cov_pred[:, elem])/cov_test[:, elem]
        #x = res[res < np.percentile(res, 95)]
        #Plot histogram.
        #hist_res(res, n_bins=int(np.max(res)-np.min(res)), title = titles[i], y_label='N', x_label = 'Norm. res.')
        hist_res(res[(res<10) & (res>-10)], n_bins=80, title = titles[i], y_label='N', x_label = 'Norm. res.')
        plt.show()
        

