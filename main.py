import os.path
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

    #Split dataset in training and test
    cov_train = cov[:1*10**6,:]
    cov_test = cov[1*10**6:,:]
    par_train = par[:1*10**6,:]
    par_test = par[1*10**6:,:]

    cov_train, cov_test, par_train, par_test = train_test_split(cov, par, test_size=0.92, random_state=42)
    print(f'cov_train size = {cov_train.size}')
    print(f'par_train size = {par_train.size}')
    
    ##Create an autoencoder model.
    #Input data.
    input_data = Input(shape=(15,))  
    #Input parameters for conditioning.
    input_params = Input(shape=(5,))
    #Concatenate the two input layers.

    #Encoder.
    hidden2 = Dense(50,activation='relu')(input_params)
    hidden = Dense(50,activation='relu')(input_data)
    concat = Concatenate()([hidden, input_params])
    hidden = Dense(100,activation='relu')(concat)
    concat = Concatenate()([hidden, hidden2])
    hidden = Dense(50,activation='relu')(concat)
    hidden = BatchNormalization()(hidden)

    code=Dense(5,activation='sigmoid')(hidden) #Encoded.

    #Decoder.
    hidden = Dense(30,activation='relu')(code)
    hidden3 = Dense(50,activation='relu')(input_params)
    concat = Concatenate()([hidden, input_params])
    hidden = Dense(100,activation='relu')(concat)
    concat = Concatenate()([hidden, hidden3])
    hidden = Dense(50,activation='relu')(concat)
    #Output.
    outputs = Dense(15, activation='linear')(hidden)
    
    model = Model(inputs=[input_data, input_params], outputs=outputs)
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    

    tf.keras.utils.plot_model(model, "model.png",show_shapes=True)

    input("Press a button to start training")

    
    #Dictionary for training settings
    dt = {
        "validation_split" : 0.5,
        "epochs" : 500,
        "batch_size" : 150,
        #Early stopping settings.
        "EarlyStopping_monitor" : "val_loss",
        "EarlyStopping_patience" : 75,
        #Reduce learning rate on plateau settings.
        "ReduceLROnPlateau_monitor" : "val_loss",
        "ReduceLROnPlateau_factor" : 0.25,
        "ReduceLROnPlateau_patience" : 50
        }

    ##Training.
    t_0 = time.time()
    
    history=model.fit([cov_train, par_train], cov_train, validation_split=dt["validation_split"],epochs=dt["epochs"],verbose=1, batch_size=dt["batch_size"],
                             callbacks = [EarlyStopping(monitor=dt["EarlyStopping_monitor"], patience=dt["EarlyStopping_patience"], verbose=1),
                                          ReduceLROnPlateau(monitor=dt["ReduceLROnPlateau_monitor"], factor=dt["ReduceLROnPlateau_factor"],
                                                            patience=dt["ReduceLROnPlateau_patience"], verbose=1)])
   
    #history=model.fit([cov_train, par_train], cov_train, validation_split=dt["validation_split"],epochs=500,verbose=1)
    elapsed_time = time.time() - t_0
    print(f'Training time: {elapsed_time}')
    ##Plot loss.
    plot_loss(history.history['loss'], history.history['val_loss'])
    plt.show()

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
    test_data, A, test_params, B = train_test_split(cov_test, par_test, test_size=0.95, random_state=42)
    
    #test_data = cov_test[0:1*10**6, :]
    #test_params = par_test[0:1*10**6, :]
    cov_pred = model.predict([test_data, test_params])

    np.save('/mnt/c/Users/HP/Desktop/cov_pred.npy',cov_pred)
    np.save('/mnt/c/Users/HP/Desktop/test_data.npy',test_data)

    print(test_data.shape)
    print(cov_pred.shape)
    
    index = [0, 5, 9, 12, 14]
    titles = ['qoverp', 'lambda', 'phi', 'dxy', 'dsz']
    # Da applicare taglio
    for i, elem in enumerate(index):
        print(f'i = {elem}')
        hist_res((test_data[:, elem] - cov_pred[:, elem]) / test_data[:, elem], n_bins=80, title = titles[i], y_label='N', x_label = 'Norm. res.')
        plt.show()
        

