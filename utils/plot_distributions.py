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
    


if __name__ == "__main__":
    cov_pred = np.load('/mnt/c/Users/HP/Desktop/cov_pred_enc5_1.npy')
    cov_test = np.load('/mnt/c/Users/HP/Desktop/cov_test_enc5_1.npy')
    scal = np.load('/mnt/c/Users/HP/Desktop/scal.npy')
    cov_pred = np.transpose(np.transpose(cov_pred)*scal)
    cov_test = np.transpose(np.transpose(cov_test)*scal)
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
    mean_relative_absolute_error = np.zeros(15)
    mean_absolute_error = np.zeros(15)
    for i in range(15):
        a = np.abs((cov_test[:,i] - cov_pred[:,i])/cov_test[:,i])
        b = np.abs(cov_test[:,i] - cov_pred[:,i])
        mean_relative_absolute_error[i] = a.mean()
        mean_absolute_error[i] = b.mean()

        print(f'Relative Absolute Error = {mean_relative_absolute_error[i]}\nAbsolute Error = {mean_absolute_error[i]}\n')


    index = [0, 5, 9, 12, 14]
    titles = ['qoverp', 'lambda', 'phi', 'dxy', 'dsz']
    # Da applicare taglio
    
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