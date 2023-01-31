import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import tensorflow as tf

from math import *
# Fixing random state for reproducibility
np.random.seed(123)

def get_cov(filename):
    '''Get covariance matrix from file 'filename'.
    '''
    return cov = np.load(filename)

def print_cov(cov):
    '''Print covariance matrix from the array of 15 values.
    '''
    print(cov[0:5])
    print((cov[1],cov[5:9]))
    print((cov[2], cov[6], cov[9:12]))
    print((cov[3], cov[7], cov[10], cov[12:14]))
    print((cov[4], cov[8], cov[11], cov[13], cov[14]))

def hist_res(res, n_bins=50, title='Residuals distribution', x_label='x label',
             y_label='y label'):
    '''Take residuals and plot histogram. 
    '''
    plt.hist(res, n_bins)
    plt.title(title, size=15)
    plt.xlabel(x_label, size=12)
    plt.ylabel(y_label, size=12)



inputs=Input(shape=(2,))
#create a encoding / decoding architecture with last layer having same dimensionality as the input
hidden=Dense(50,activation='relu')(inputs)
hidden=Dense(200,activation='relu')(hidden)
hidden=Dense(50,activation='relu')(hidden)

code=Dense(1,activation='sigmoid')(hidden) #let's compress to a single variable

hidden=Dense(50,activation='relu')(code)
hidden=Dense(200,activation='relu')(hidden)
hidden=Dense(50,activation='relu')(hidden)

outputs = Dense(2, activation='linear')(hidden)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='MSE', optimizer='adam')
model.summary()

tf.keras.utils.plot_model(model, "model.png",show_shapes=True)

input("Premi un tasto per avviare il training")

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

history=model.fit(X, X, validation_split=0.5,epochs=500,verbose=1, 
                         callbacks = [
                                             EarlyStopping(monitor='val_loss', patience=20, verbose=1),                                                              ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, verbose=1)
                                     ]
                 )
print(history.history.keys())
print(history.history['loss'])
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss functions")
plt.show()

input("Premi per continuare")

encoder = Model(inputs=inputs, outputs=code)
decoder = Model(inputs=code , outputs=outputs)
#encoder.summary()
#decoder.summary()

input("Premi per vedere i risultati")

print("Prediction of [0,1] and [1,0]")
print(model.predict(np.array(([0.0,1],[1,0]))))

print("Encoder values for [0,1], [1,0] and [-1,0]")
print(encoder.predict(np.array(([0.0,1],[1,0],[-1,0]))))

#let's now generate new samples by producing random states in the 
latent=np.linspace(0,1,100)
deco=decoder.predict(latent)
plt.scatter(deco[:, 0], deco[:, 1], c=latent, cmap=plt.cm.RdBu, edgecolors='k')
plt.show()
input("Premi per finire")


