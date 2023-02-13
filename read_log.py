import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

class Log:
    def __init__(self, description, model, train_settings, training_time, loss, val_loss):
        self.description = description
        self.model = model
        self.train_settings = train_settings
        self.training_time = training_time 
        self.loss = loss
        self.val_loss = val_loss

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

def print_log(obj, only_last = False):
    print(f'There are {len(obj)} objects.')
    i = 0
    for o in obj:
        print(f'Object n. {i}----------------------------------------')
        print(o.description)
        print('Model summary:')
        o.model.summary()
        print(o.train_settings)
        #print(f'Final loss: {o.loss}')
        #print(f'Final val_loss: {o.val_loss}')
        plot_loss(o.loss,o.val_loss)
        plt.show()
        print(f'Training time: {o.training_time}')
        print(f'End object n. {i}------------------------------------')

if __name__ == "__main__":
    filename = "log"
    with open(filename, 'rb') as inst_file:
        arr = pickle.load(inst_file)
    print_log(arr)
