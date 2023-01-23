import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import tensorflow as tf
import ROOT
import pandas
from math import *

rdf = ROOT.RDataFrame("Events", "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root")
print(rdf.GetColumnNames())

print(pandas.DataFrame(rdf_mass.AsNumpy(["Dimuon_mass", "event", "run"])))

'''
N=10000
phi = 1.0 * np.random.uniform(0,2.*pi,N)
r = 1.0 * np.random.uniform(1,1.05,N)
x1= r*np.cos(phi)
x2= r*np.sin(phi)

X=np.stack((x1,x2),axis=1)
print(f"X shape: {X.shape}")

plt.scatter(X[:, 0], X[:, 1])
plt.show()
input("Premi un tasto per continuare")


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
'''

