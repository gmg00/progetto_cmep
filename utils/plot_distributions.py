import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from math import *
# Fixing random state for reproducibility
np.random.seed(123)

def plot_hist(x, n_bins=50, title='', xlabel='element', ylabel='N'):
    plt.hist(x,n_bins)
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title, size=15)
    plt.grid()

def percentile_matrix(cov,p):
    s = cov.sum()
    if s >= 0:
        pcov = cov[abs(cov) < np.percentile(cov,p)]
        return pcov
    elif s < 0:
        pcov = cov[abs(cov) > -np.percentile(-cov,p)]
        return pcov

def hist_res(res, n_bins=50, title='', x_label='x label',
             y_label='Residuals'):
    '''Take residuals and plot histogram. 
    '''
    plt.hist(res, n_bins)
    plt.title(title, size=15)
    plt.xlabel(x_label, size=12)
    plt.ylabel(y_label, size=12)
    plt.grid()

def matrix_error_graph(path, enc_lim=(5,15),tot_graph=True, compare_diag_matrix=True):
    index = [0, 5, 9, 12, 14]
    n = enc_lim[1] - enc_lim[0] + 1 
    mean_relative_absolute_error = np.zeros((n,5))
    mean_absolute_error = np.zeros((n,5))
    tot_abs_error = np.zeros(n)
    tot_rel_abs_error = np.zeros(n)
    scal = np.load(f'{path}scal.npy')

    for i in range(enc_lim[0],enc_lim[1]+1):
        print(f'Opening enc_{i}')
        cov_pred = np.load(f'{path}cov_pred_enc{str(i)}.npy')
        cov_test = np.load(f'{path}cov_test_enc{str(i)}.npy')
        cov_pred = np.transpose(np.transpose(cov_pred)*scal)
        cov_test = np.transpose(np.transpose(cov_test)*scal)
        k = 0
        for o in index:
            print(f'Starting mean on the {k}° diagonal element')
            a = np.abs((cov_test[:,o] - cov_pred[:,o])/(cov_test[:,o]+10**(-4)))
            b = np.abs(cov_test[:,o] - cov_pred[:,o])
            mean_relative_absolute_error[i-5,k] = a.mean()
            mean_absolute_error[i-5,k] = b.mean()
            k += 1
        if tot_graph:
            a = np.abs((cov_test - cov_pred)/(cov_test+10**(-4)))
            b = np.abs(cov_test - cov_pred)
            tot_rel_abs_error[i-5] = a.mean()
            tot_abs_error[i-5] = b.mean()
    print(mean_absolute_error)
    print(mean_relative_absolute_error)

    titles = ['qoverp', 'lambda', 'phi', 'dxy', 'dsz']
    if tot_graph:
        plt.figure(f'Mean Absolute Error entire matrix Graph')
        plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),tot_abs_error,'-',label='mean absolute error')
        plt.ylabel('Errors', size=12)
        plt.xlabel('Encoding nodes', size=12)
        plt.title(f'Mean Absolute Error entire matrix Graph', size=15)

        if compare_diag_matrix:
            plt.figure(f'Compare entrire matrix and diagonal relative error')
            plt.subplot(2,1,1)
            plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_relative_absolute_error.mean(axis=1),'-',label='mean absolute error')
            plt.ylabel('Digonal mean relative absolute error', size=12)
            plt.xlabel('Encoding nodes', size=12)
            plt.title(f'Compare entrire matrix and diagonal relative error', size=15)
            plt.grid()
            plt.subplot(2,1,2)
            plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),tot_rel_abs_error,'-',label='mean absolute error')
            plt.ylabel('Entire matrix mean relative absolute error', size=12)
            plt.xlabel('Encoding nodes', size=12)
            plt.grid()

            plt.figure(f'Compare entrire matrix and diagonal absolute error')
            plt.subplot(2,1,1)
            plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_absolute_error.mean(axis=1),'-',label='mean absolute error')
            plt.ylabel('Digonal mean absolute error', size=12)
            plt.xlabel('Encoding nodes', size=12)
            plt.title(f'Compare entrire matrix and diagonal absolute error', size=15)
            plt.grid()
            plt.subplot(2,1,2)
            plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),tot_abs_error,'-',label='mean absolute error')
            plt.ylabel('Entire matrix mean absolute error', size=12)
            plt.xlabel('Encoding nodes', size=12)
            plt.grid()


    plt.figure(f'Mean Absolute Error entire diagonal Graph')
    plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_relative_absolute_error.mean(axis=1),'-',label='mean absolute error')
    plt.ylabel('Errors', size=12)
    plt.xlabel('Encoding nodes', size=12)
    plt.title(f'Mean Absolute Error entire diagonal Graph', size=15)

    if 0:
        for i in range(len(titles)):

            plt.figure(f'Mean Absolute Error Graph {titles[i]}')
            plt.subplot(2,1,1)
            plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_absolute_error[:,i],'-',label='mean absolute error')
            plt.ylabel('Errors', size=12)
            plt.xlabel('Encoding nodes', size=12)
            plt.title(f'Mean Absolute Error Graph {titles[i]}', size=15)
            plt.subplot(2,1,2)
            plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_relative_absolute_error[:,i],'-',label='mean relative absolute error')
            plt.ylabel('Errors', size=12)
            plt.xlabel('Encoding nodes', size=12)
            #plt.title(f'Mean Absolute Error Graph {titles[i]}', size=15)


if __name__ == "__main__":
    cov_pred = np.load('/mnt/c/Users/HP/Desktop/cov_pred_enc7.npy')
    cov_test = np.load('/mnt/c/Users/HP/Desktop/cov_test_enc7.npy')
    scal = np.load('/mnt/c/Users/HP/Desktop/scal.npy')
    #cov_pred = np.transpose(np.transpose(cov_pred)*scal)
    #cov_test = np.transpose(np.transpose(cov_test)*scal)
    #scaler = preprocessing.StandardScaler().fit(cov)
    #cov = scaler.transform(cov)
    '''
    i=0
    
    for i in range(15):
        plt.figure(i)
        plt.subplot(2,1,1)
        pcov = percentile_matrix(cov[:,i],95)
        #cov[cov[:,i] < np.percentile(cov[:,1],95)]
        plot_hist(cov[:,i], n_bins = 100, title=f'Histogram column {i}', xlabel=f'Matrix element n. {i}', ylabel='N')
        plt.subplot(2,1,2)
        plot_hist(cov[:,i], n_bins = 100, title='', xlabel=f'Matrix element n. {i}', ylabel='N')
        plt.ylim([0.,10.])
        plt.show()
    '''
    if 0:
        mean_relative_absolute_error = np.zeros(15)
        mean_absolute_error = np.zeros(15)
        for i in range(15):
            a = np.abs((cov_test[:,i] - cov_pred[:,i])/cov_test[:,i])
            b = np.abs(cov_test[:,i] - cov_pred[:,i])
            mean_relative_absolute_error[i] = a.mean()
            mean_absolute_error[i] = b.mean()

            print(f'Relative Absolute Error = {mean_relative_absolute_error[i]}\nAbsolute Error = {mean_absolute_error[i]}\n')


    matrix_error_graph('/mnt/c/Users/HP/Desktop/progetto_cmep/data/outputs/')

    index = [0, 5, 9, 12, 14]
    titles = ['qoverp', 'lambda', 'phi', 'dxy', 'dsz']
    # Da applicare taglio
    if 0: 
        for i, elem in enumerate(index):
            print(f'i = {elem}')
            res = (cov_test[:, elem] - cov_pred[:, elem])
            res_norm = (cov_test[:, elem] - cov_pred[:, elem])/cov_test[:, elem]
            #Calculate residuals.
            plt.title(titles[i],size=15)

            plt.subplot(2,1,1)
            perc = np.percentile(abs(res),95)
            hist_res(res[(res<perc) & (res>-perc)], n_bins=80, y_label='N', x_label = 'x_decompresso - x_originale')
            plt.title(titles[i],size=15)

            plt.subplot(2,1,2)
            perc = np.percentile(abs(res_norm),95)
            hist_res(res_norm[(res_norm<perc) & (res_norm>-perc)], n_bins=80, y_label='N', x_label = '(x_decompresso - x_originale)/x_originale')
    plt.show()
    
    '''
    l = []
    i = 0
    for row in cov_pred:
        x = 0
        i = i+1
        for e in row:
            x = x + e**2
        l.append(x)
        if i == 10**5: break

    print(l)
    plt.hist(l,bins=50)
    plt.show()
    '''
    '''
    for i, elem in enumerate(index):
        plt.subplot(2,1,1)
        plt.title(titles[i]+' distributions', size=15)
        plt.ylabel('test data',size=12)
        plt.hist(cov_test[:,elem],100)
        plt.subplot(2,1,2)
        plt.ylabel('predicted data',size=12)
        plt.hist(cov_pred[:,elem],100)
        plt.xlabel(titles[i],size=12)
        plt.show()
    '''

    #fare metrica solo sulla diagonale
    #econding di 2,3,4
    #Dx/(x+epsilon) -> questo perché il problema può essere che x è ~ 0. epsilon ~ 10^-5 (giusto per non avere una divisione per 0).
    #fare un grafico con errore quadratico medio relativo su ogni elemento della diagonale
    #Fare confronto con quello che usa CMS di default.
    #hyperparameter scan, hyperparameter optimization.