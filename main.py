"""Autoencoder for covariance matrix using 5 parameters as conditioning.
"""

def get_file(filename):
    '''Get matrix from file 'filename'.

    Args:
        filename (string): name of the file containing the matrix.

    Returns:
        cov (numpy.ndarray): array of the 15 significative elements of covariance matrix.
    '''
    return np.load(filename)

def print_cov(cov):
    '''Print covariance matrix from the array of 15 values.

    Args:
        cov (nupmy.ndarray): array of the 15 significative elements of covariance matrix.
    '''
    print(cov[0:5])
    print((cov[1],cov[5:9]))
    print((cov[2], cov[6], cov[9:12]))
    print((cov[3], cov[7], cov[10], cov[12:14]))
    print((cov[4], cov[8], cov[11], cov[13], cov[14]))

def preprocessing_dataset(cov, par, save_scal=False, random_state = 42, dir_name=None):
    """Normalize datasets and split them in training and test datasets.

    Args:
        cov: matrix containing 15 significative elements of covariance matrix.

        par: matrix containing 5 conditioning parameters.

        save_scal: Boolean. If True create an array with scaling factors and save it.
        Default = False.

        random_state: random_state for train_test_split function. Default = 42.

    Returns:
        cov_train: training dataset containing covariance matrix elements.

        cov_test: test dataset containing covariance matrix elements.

        par_train: training dataset containing conditioning parameters.

        par_test: test dataset containing conditioning parameters.
    """
    #Normalize dataset
    scaler_cov = preprocessing.Normalizer().fit(cov)
    scaler_par = preprocessing.Normalizer().fit(par)

    #Split dataset in training and test
    cov_train0, cov_test0, par_train0, par_test0 = train_test_split(cov,
                                                                    par,
                                                                    test_size=0.5,
                                                                    random_state=random_state)

    cov_train = scaler_cov.transform(cov_train0)
    cov_test = scaler_cov.transform(cov_test0)
    par_train = scaler_par.transform(par_train0)
    par_test = scaler_par.transform(par_test0)

    if save_scal:
        #Get normalization factors for every row of test dataset.
        scal = []
        for i in range(cov_test.shape[0]):
            scal.append(cov_test0[i,0]/cov_test[i,0])

        #np.save(dir_name + '/data/outputs/scal.npy', np.array(scal))
        np.save(f'{dir_name}/data/outputs/scal.npy', np.array(scal))

    return cov_train, cov_test, par_train, par_test

def hist_res(res, n_bins=50,
             title='Residuals distribution', x_label='x label', y_label='Residuals'):
    '''Take residuals and plot histogram.

    Args:
        res: residuals array.

        n_bins: number of bins. Deafault = 50.

        title: title of histogram. Default = 'Residuals distribution'

        x_label: label of x axis. Default = 'x label'

        y_label: label of y axis. Default = 'Residuals'
    '''
    plt.hist(res, n_bins)
    plt.title(title, size=15)
    plt.xlabel(x_label, size=12)
    plt.ylabel(y_label, size=12)
    plt.grid()
    plt.show()

def plot_loss(loss, val_loss):
    """Plot loss functions.

    Args:
        loss: points of loss function.

        val_loss: points of validation loss function.
    """
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

def plot_metrics(metric, val_metric, title='Metric functions'):
    """Plot metric functions.

    Args:
        metric: points of metric function.

        val_metric: points of validation metric function.

        title: string. Title of figure. Default = 'Metric functions'
    """
    plt.figure(figsize=(11, 6))
    plt.plot(metric, label='Metric')
    plt.plot(val_metric, label='Validation Metric')
    plt.xlabel('Epochs', size=12)
    #x = [0,100,200,300,400,500,600,700,800,900,1000]
    plt.xticks(size=12)
    plt.ylabel('Metric functions', size=12)
    plt.yticks(size=12)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.title(title, size=15)

def generator(x1_data, x2_data, batch_size):
    """Create a generator for model training. Training dataset is [covariance matrix, parameters]
    and target is [covariance matrix].

    Args:
        x1_data: train dataset composed by the 15 significative elements of covariance matrix.
        x1_data is also the target.

        x2_data: 5 parameters used for conditioning during training.

        batch_size: integer. Number of samples per gradient update.

    Yields:
        [x1_batch,x2_batch], x1_batch:
        inputs batch of size = batch_size and target batch of size = batch_size.
    """
    samples_per_epoch = x1_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    ids = np.arange(len(x1_data))
    counter=0
    x1_batch = []
    x2_batch = []
    while 1:
        if counter == 0:
            np.random.shuffle(ids)
            x1_data = x1_data[ids]
            x2_data = x2_data[ids]
        x1_batch = np.array(x1_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        x2_batch = np.array(x2_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')

        counter += 1
        yield [x1_batch,x2_batch],x1_batch

        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

def predict_generator(x1_data, x2_data, batch_size):
    """Create a generator for model.predict() function.
    Test dataset is [covariance matrix, parameters].

    Args:
        x1_data: test dataset composed by the 15 significative elements of covariance matrix.

        x2_data: 5 parameters used for conditioning.

        batch_size: integer. Number of samples per gradient update.

    Yields:
        [x1_batch,x2_batch]: inputs batch of size = batch_size.
    """
    counter=0
    x1_batch = []
    x2_batch = []
    while 1:
        x1_batch = np.array(x1_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        x2_batch = np.array(x2_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        counter += 1
        yield [x1_batch, x2_batch],

def my_metric(y_true, y_pred):
    '''Compute loss = mean(sqrt(y_true - y_pred)**2 / y_true**2), axis=-1)
    only on diagonal elements.

    Args:
        y_true: array of true values.

        y_pred: array of predicted values.

    Returns:
        metric: tensorflow.tensor. Mean quadratic relative error metric on diagonal elements.
    '''

    index = [0,5,9,12,14]
    y_true = tf.gather(y_true, indices=index)
    y_pred = tf.gather(y_pred, indices=index)
    temp = tf.math.sqrt(tf.math.xdivy(tf.square(y_true-y_pred),tf.square(y_true)))

    return tf.reduce_mean(temp, axis=-1)

def my_metric2(y_true, y_pred):
    '''Compute loss = mean(abs((y_true - y_pred) / y_true)), axis=-1)

    Args:
        y_true: array of true values.

        y_pred: array of predicted values.

    Returns:
        metric: tensorflow.tensor. Mean absolute relative error metric.
    '''

    temp = tf.math.sqrt(tf.math.xdivy(tf.square(y_true-y_pred),tf.square(y_true)))

    return tf.reduce_mean(temp, axis=-1)

def my_metric3(y_true, y_pred, epsilon = 10**(-5)):
    '''Compute loss = mean(abs((y_true - y_pred) / (y_true + epsilon))), axis=-1)
    only on diagonal elements.

    Args:
        y_true: array of true values.

        y_pred: array of predicted values.

        epsilon: float. Parameter used to avoid division by a number close to 0.

    Returns:
        metric: tensorflow.tensor. Mean absolute relative error metric on diagonal elements.
    '''
    index = [0,5,9,12,14]
    y_true = tf.gather(y_true, indices=index)
    y_pred = tf.gather(y_pred, indices=index)
    temp = tf.math.sqrt(tf.math.xdivy(tf.square(y_true-y_pred),tf.square(y_true + epsilon)))

    return tf.reduce_mean(temp, axis=-1)

def main(dir_name):
    """Main function. Create a model, fit that model and save the results.

    Args:
        dir_name: name of directory in which is contained input datasets.
    """

    model_name = 'enc3'

    #Get covariance matrix.

    cov = get_file(f'{dir_name}/data/inputs/cov_tot.npy')

    #Get parameters for conditioning.

    par = get_file(f'{dir_name}/data/inputs/par_tot.npy')

    cov_train, cov_test, par_train, par_test = preprocessing_dataset(cov, par)

    #model = my_model(5)

    #Create an autoencoder model.
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

    code=Dense(3,activation='sigmoid')(hidden) #Encoded.

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
    tf.keras.utils.plot_model(model, f"{dir_name}/data/models/model_{model_name}.png",
                              show_shapes=True)


    #Dictionary of training settings
    train_dt = {
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
    cov_train, cov_val, par_train, par_val = train_test_split(cov_train,
                                                              par_train,
                                                              test_size=0.5,
                                                              random_state=42)

    #print(f'len/batch_size = {cov_train.shape[0] // dt["batch_size"]}')

    input("Press a button to start training")

    gen = generator(cov_train, par_train, train_dt['batch_size'])

    history = model.fit(
        gen,
        epochs=train_dt['epochs'],
        steps_per_epoch = cov_train.shape[0]//train_dt['batch_size'],
        validation_data = generator(cov_val,par_val,train_dt['batch_size']),
        validation_steps = cov_val.shape[0]//train_dt['batch_size'],
        callbacks = [EarlyStopping(monitor=train_dt["EarlyStopping_monitor"],
                                   patience=train_dt["EarlyStopping_patience"],
                                   verbose=1),
                     ReduceLROnPlateau(monitor=train_dt["ReduceLROnPlateau_monitor"],
                                       factor=train_dt["ReduceLROnPlateau_factor"],
                                       patience=train_dt["ReduceLROnPlateau_patience"],
                                       verbose=1)])

    ##Plot loss.
    plot_loss(history.history['loss'], history.history['val_loss'])
    #plt.savefig(dir_name + '/data/models/loss_' + model_name +'.png')
    plot_metrics(history.history['my_metric'], history.history['val_my_metric'])

    input("Press a button to save history")
    with open(dir_name + '/data/histories/history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    input("Press a button to continue")

    ##Encoder-decoder division.

    encoder = Model(inputs=[input_data, input_params], outputs=encodedstate)
    #decoder = Model(inputs=[encodedstate, input_params] , outputs=outputs)
    print("Encoder predictions")
    #cov_enc = encoder.predict([cov_test, par_test],batch_size=1)
    cov_enc = encoder.predict(predict_generator(cov_test,par_test,train_dt['batch_size']),
                             steps = cov_test.shape[0]//train_dt['batch_size']+1,
                             verbose = 1)

    np.save(dir_name + '/data/outputs/cov_enc_' + model_name + '.npy', cov_enc)

    input("Press a button to continue")

    ##Test

    print(f'cov_test shape = {cov_test.shape}')

    cov_pred = model.predict(predict_generator(cov_test,par_test,train_dt['batch_size']),
                             steps = cov_test.shape[0]//train_dt['batch_size']+1,
                             verbose = 1)

    np.save(dir_name + '/data/outputs/cov_pred_' + model_name + '.npy', cov_pred)

    np.save(dir_name + '/data/outputs/cov_test_' + model_name + '.npy', cov_test)

    print(cov_test.shape)
    print(cov_pred.shape)

#---------------------------------------------------------------------MAIN
if __name__ == "__main__":

    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from keras.layers import Dense, Input, Activation, Concatenate
    from keras.models import Model
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    # Fixing random state for reproducibility
    np.random.seed(123)

    DIR_NAME = '/mnt/c/Users/HP/Desktop/progetto_cmep'
    main(DIR_NAME)
