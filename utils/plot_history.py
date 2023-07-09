import numpy as np
import matplotlib.pyplot as plt
import pickle

from math import *
# Fixing random state for reproducibility
np.random.seed(123)

def import_histories(filenames):
    '''
    '''
    histories = []
    for filename in filenames:
        with open(filename, "rb") as file_pi:
            histories.append(pickle.load(file_pi))
    return histories

def plot_history(histories, filenames):
    '''
    '''
    plt.figure(1)
    i = 0
    colors = ['b','g','r','c','m','y','k','w']
    for history in histories:
        plt.plot(history['loss'], label=filenames[i],color = colors[i])
        plt.plot(history['val_loss'],'--',color = colors[i])
        i += 1
    plt.xlabel('Epochs', size=12)
    #x = [0,100,200,300,400,500,600,700,800,900,1000]
    plt.xticks(size=12)
    plt.ylabel('Loss functions', size=12)
    #plt.ylim([-0.05,0.4])
    plt.yticks(size=12)
    plt.legend(loc='best', fontsize=12)
    plt.title('Loss vs Validation Loss', size=15)
    plt.grid()
        
def smoothing_metric(arr, n):
        l = len(arr)
        c = l // n
        mean = []
        k = 0
        while 1:
            if k == c:
                if l%n != 0:
                    mean.append(arr[k*n:k*n+l%n].mean())
                return mean
            mean.append(arr[k*n:k*n+n].mean())
            k += 1

if __name__ == "__main__":

    plot_hist = True
    plot_metrics = True

    directory = '/mnt/c/Users/HP/Desktop/history/'
    filenames = ['history_model_enc6', 'history_model_enc5_4']
    filenamesdir = []

    for i in range(len(filenames)):
        filenamesdir.append(directory + filenames[i])
    histories = import_histories(filenamesdir)
    
    if plot_hist:
        plot_history(histories, filenames)
        plt.show()
    
    n = 10

    if plot_metrics:
        
        m = []
        val_m = []

        for history in histories:
            m.append(np.array(history['my_metric2']))
            val_m.append(np.array(history['val_my_metric2']))

        n = 30
        m_smooth = []
        val_m_smooth = []
        for temp in m:
            m_smooth.append(smoothing_metric(temp,n))
        for temp in val_m:
            val_m_smooth.append(smoothing_metric(temp,n))

        plt.figure(2)
        i = 0
        colors = ['b','g','r','c','m','y','k','w']
        for i in range(len(m)):
            #print(history)
            plt.plot(m_smooth[i], label=filenames[i],color = colors[i])
            plt.plot(val_m_smooth[i],'--',color = colors[i])
            i += 1
        plt.xlabel('Epochs', size=12)
        #x = [0,100,200,300,400,500,600,700,800,900,1000]
        plt.xticks(size=12)
        plt.ylabel('Metric ', size=12)
        #plt.ylim([-0.05,0.4])
        plt.yticks(size=12)
        plt.legend(loc='best', fontsize=12)
        plt.title('Metric 1', size=15)
        plt.grid()


        m = []
        val_m = []

        for history in histories:
            m.append(np.array(history['my_metric3']))
            val_m.append(np.array(history['val_my_metric3']))

        n = 40
        m_smooth = []
        val_m_smooth = []
        for temp in m:
            m_smooth.append(smoothing_metric(temp,n))
        for temp in val_m:
            val_m_smooth.append(smoothing_metric(temp,n))

        plt.figure(3)
        i = 0
        colors = ['b','g','r','c','m','y','k','w']
        for history in histories:
            plt.plot(m_smooth[i], label=filenames[i],color = colors[i])
            plt.plot(val_m_smooth[i],'--',color = colors[i])
            i += 1
        plt.xlabel('Epochs', size=12)
        #x = [0,100,200,300,400,500,600,700,800,900,1000]
        plt.xticks(size=12)
        plt.ylabel('Metric ', size=12)
        #plt.ylim([-0.05,0.4])
        plt.yticks(size=12)
        plt.legend(loc='best', fontsize=12)
        plt.title('Metric 2', size=15)
        plt.grid()
        '''
        plt.figure(4)
        i = 0
        colors = ['b','g','r','c','m','y','k','w']
        for history in histories:
            plt.plot(history['my_metric3'], label=filenames[i],color = colors[i])
            plt.plot(history['val_my_metric3'],'--',color = colors[i])
            i += 1
        plt.xlabel('Epochs', size=12)
        #x = [0,100,200,300,400,500,600,700,800,900,1000]
        plt.xticks(size=12)
        plt.ylabel('Metric ', size=12)
        #plt.ylim([-0.05,0.4])
        plt.yticks(size=12)
        plt.legend(loc='best', fontsize=12)
        plt.title('Metric 3', size=15)
        plt.grid()
        '''
        plt.show()

