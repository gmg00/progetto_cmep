"""This module is used to analyze loss and metric functions.
"""

def import_histories(filenames):
    '''Import history files.

    Args: 
        filenames: array of strings containing filenames for every history file.
    '''
    histories = []
    for filename in filenames:
        with open(filename, "rb") as file_pi:
            histories.append(pickle.load(file_pi))
    return histories

def plot_history(histories, names):
    '''Plot loss and validation loss functions.

    Args:
        histories: matrix with all loss and validation loss functions.
        names: array of strings containing labels for every set of loss and val_loss.
    '''
    plt.figure(1)
    i = 0
    colors = ['b','g','r','c','m','y','k','darkgreen','midnightblue','maroon','gold']
    for history in histories:
        plt.plot(history['loss'], label=names[i],color = colors[i])
        plt.plot(history['val_loss'],'--',color = colors[i])
        i += 1
    plt.xlabel('Epochs', size=15)
    plt.xticks(size=15)
    plt.ylabel('Loss functions', size=15)
    plt.yticks(size=15)
    plt.legend(loc='best', fontsize=15)
    plt.title('Loss vs Validation Loss', size=18)
    plt.grid()

def smoothing_metric(arr, n):
    """Compute smoothing of n elements of array arr.

    Args:
        arr: points of metrics that will be smoothed.
        n: number of points mediated during smoothing.
    """
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

def plot_metric(histories, metric_name, n, names):
    '''Plot metric functions.

    Args:
        histories: matrix with all metrics functions.
        metric_name: name of the metric.
        names: array of strings containing labels for every run.
        n: number of points mediated during smoothing.
    '''
    m = []
    val_m = []
    for history in histories:
        m.append(np.array(history[metric_name]))
        val_m.append(np.array(history[f'val_{metric_name}']))

    m_smooth = []
    val_m_smooth = []
    for temp in m:
        m_smooth.append(smoothing_metric(temp,n))
    for temp in val_m:
        val_m_smooth.append(smoothing_metric(temp,n))

    plt.figure(f'{metric_name} graph')
    i = 0
    colors = ['b','g','r','c','m','y','k','darkgreen','midnightblue','maroon','gold']
    for i in range(len(m)):
        plt.plot(m_smooth[i], label=names[i],color = colors[i])
        plt.plot(val_m_smooth[i],'--',color = colors[i])
        i += 1
    plt.xlabel('Epochs', size=15)
    plt.xticks(size=15)
    plt.ylabel('Metric ', size=15)
    #plt.ylim([-0.05,0.4])
    plt.yticks(size=15)
    plt.legend(loc='best', fontsize=15)
    plt.title(f'{metric_name}', size=18)
    plt.grid()

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    # Fixing random state for reproducibility
    np.random.seed(123)

    directory = '/mnt/c/Users/HP/Desktop/progetto_cmep/data/histories/'
    filenames = ['history_enc5_bs40','history_enc5_bs80','history_enc5_bs100',
                 'history_enc5', 'history_enc5_bs300','history_enc5_bs400']
    filenamesdir = []
    names = ['batch_size = 40', 'batch_size = 80', 'batch_size = 100',
             'batch_size = 200', 'batch_size = 300', 'batch_size = 400']

    for i in range(len(filenames)):
        filenamesdir.append(directory + filenames[i])
    histories = import_histories(filenamesdir)

    plot_history(histories, names)
    plot_metric(histories, 'my_metric2', 30, names)

    plt.show()

