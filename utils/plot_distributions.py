"""This module is used to analyze datasets before and after being processed by the model.
It contains functions useful to graph the evolution of some important metrics
as encoding space dimension and batch_size dimension change.
"""

def plot_hist(x_arr, n_bins=50, title='', xlabel='element', ylabel='N'):
    '''Take residuals and plot histogram.

    Args:
        x_arr: array to plot.

        n_bins: number of bins. Deafault = 50.

        title: title of histogram. Default = ''.

        xlabel: label of x axis. Default = 'element'.

        ylabel: label of y axis. Default = 'N'.
    '''
    plt.hist(x_arr,n_bins)
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title, size=15)
    plt.grid()

def hist_res(res, n_bins=50, title='', x_label='x label', y_label='Residuals'):
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

def matrix_error_graph(path, enc_lim=(3,15)):
    """Produce graphs about absolute error and relative absolute error of matrix elements.

    Args:
        path: directory path in which predicted and test matrix are contained.
        enc_lim: tuple. Minimum and maximum of encoding space dimension. Default = (3,15)
    """
    index = [0, 5, 9, 12, 14]
    n_dim = enc_lim[1] - enc_lim[0] + 1
    mean_relative_absolute_error = np.zeros((n_dim,5))
    mean_absolute_error = np.zeros((n_dim,5))
    tot_abs_error = np.zeros(n_dim)
    tot_rel_abs_error = np.zeros(n_dim)
    scal = np.load(f'{path}scal.npy')
    eps=0

    for i in range(enc_lim[0],enc_lim[1]+1):
        print(f'Opening enc_{i}')
        cov_pred = np.load(f'{path}cov_pred_enc{str(i)}.npy')
        cov_test = np.load(f'{path}cov_test_enc{str(i)}.npy')
        cov_pred = np.transpose(np.transpose(cov_pred)*scal)
        cov_test = np.transpose(np.transpose(cov_test)*scal)
        k = 0
        for ind in index:
            print(f'Starting mean on the {k}° diagonal element')
            temp_a = np.abs((cov_test[:,ind] - cov_pred[:,ind])/(cov_test[:,ind]+eps))
            temp_b = np.abs(cov_test[:,ind] - cov_pred[:,ind])
            mean_relative_absolute_error[i-enc_lim[0],k] = temp_a.mean()
            mean_absolute_error[i-enc_lim[0],k] = temp_b.mean()
            k += 1
        temp_a = np.abs((cov_test - cov_pred)/(cov_test+eps))
        temp_b = np.abs(cov_test - cov_pred)
        tot_rel_abs_error[i-enc_lim[0]] = temp_a.mean()
        tot_abs_error[i-enc_lim[0]] = temp_b.mean()

    plt.figure('Compare entrire matrix and diagonal relative error')
    plt.subplot(2,1,1)
    plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_relative_absolute_error.mean(axis=1),
             'ko--',label='mean absolute error')
    plt.ylabel('Digonal MRAE', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.xticks(np.arange(enc_lim[0],enc_lim[1]+1))
    plt.title('Mean relative absolute error (MRAE)', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),tot_rel_abs_error,
             'ko--',label='mean absolute error')
    plt.ylabel('Entire matrix MRAE', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xticks(np.arange(enc_lim[0],enc_lim[1]+1))
    plt.grid()

    plt.figure('Compare entrire matrix and diagonal absolute error')
    plt.subplot(2,1,1)
    plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),mean_absolute_error.mean(axis=1),
             'ko--',label='mean absolute error')
    plt.ylabel('Diagonal MAE', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.title('Mean absolute error (MAE)', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(np.arange(enc_lim[0],enc_lim[1]+1),tot_abs_error,
             'ko--',label='mean absolute error')
    plt.ylabel('Entire matrix MAE', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()

def element_error(path, enc):
    """Produce graphs about MRAE and MAE of every element given the encoding space of the test.

    Args:
        path: directory path in which predicted and test matrix are contained.
        enc: int. Encoding space dimension.
    """
    scal = np.load(f'{path}scal.npy')
    cov_pred = np.load(f'{path}cov_pred_enc{enc}.npy')
    cov_test = np.load(f'{path}cov_test_enc{enc}.npy')
    cov_pred = np.transpose(np.transpose(cov_pred)*scal)
    cov_test = np.transpose(np.transpose(cov_test)*scal)

    mean_relative_absolute_error = np.zeros(15)

    for i in range(15):
        temp = (np.abs((cov_test[:,i] - cov_pred[:,i])/(cov_test[:,i])))
        mean_relative_absolute_error[i] = temp.mean()

    plt.figure('Elements errors')
    plt.plot(range(15),mean_relative_absolute_error)
    plt.xlabel('Element', size=15)
    plt.ylabel('MRAE', size=15)

def element_error_tot(path, enc_lim=(3,15)):
    """Produce graphs about MRAE and MAE of every element mediated on every test with encoding space
    from enc_lim[0] to enc_lim[1].

    Args:
        path: directory path in which predicted and test matrix are contained.
        enc_lim: tuple. Minimum and maximum of encoding space dimension. Default = (3,15)
    """
    n_dim = enc_lim[1] - enc_lim[0] + 1
    mean_relative_absolute_error = np.zeros((n_dim,15))
    mean_absolute_error = np.zeros((n_dim,15))
    scal = np.load(f'{path}scal.npy')
    eps=0

    for i in range(enc_lim[0],enc_lim[1]+1):
        print(f'Opening enc_{i}')
        cov_pred = np.load(f'{path}cov_pred_enc{str(i)}.npy')
        cov_test = np.load(f'{path}cov_test_enc{str(i)}.npy')
        cov_pred = np.transpose(np.transpose(cov_pred)*scal)
        cov_test = np.transpose(np.transpose(cov_test)*scal)
        for j in range(15):
            print(f'Starting mean on the {j}° diagonal element')
            temp_a = np.abs((cov_test[:,j] - cov_pred[:,j])/(cov_test[:,j]+eps))
            temp_b = np.abs(cov_test[:,j] - cov_pred[:,j])
            mean_relative_absolute_error[i-enc_lim[0],j] = temp_a.mean()
            mean_absolute_error[i-enc_lim[0],j] = temp_b.mean()

    mrae = mean_relative_absolute_error.mean(axis=0)
    mae = mean_absolute_error.mean(axis=0)
    plt.figure('Element errors 2')
    plt.subplot(2,1,1)
    plt.stem(range(15),mrae, basefmt=' ',markerfmt='k.',linefmt='k--')
    plt.title('MRAE and MAE on elements', size=18)
    plt.xticks(range(15), size=15)
    plt.yticks(size=15)
    plt.xlabel('Element index',size=15)
    plt.ylabel('MRAE',size=15)
    plt.yscale('log')
    plt.grid()
    plt.subplot(2,1,2)
    plt.stem(range(15),mae, basefmt=' ',markerfmt='k.',linefmt='k--')
    plt.xlabel('Element index',size=15)
    plt.xticks(range(15),size=15)
    plt.ylabel('MAE',size=15)
    plt.yticks(size=15)
    plt.grid()

def matrix_error_graph_bs(path, filenames):
    """Produce graphs about absolute error and relative absolute error of matrix elements.

    Args:
        path: directory path in which predicted and test matrix are contained.
        filenames: last part of file name.
        Default = ['5_bs40','5_bs80','5_bs100', '5', '5_bs300','5_bs400'].
    """
    index = [0, 5, 9, 12, 14]
    n_dim = len(filenames)
    mean_relative_absolute_error = np.zeros((n_dim,5))
    mean_absolute_error = np.zeros((n_dim,5))
    tot_abs_error = np.zeros(n_dim)
    tot_rel_abs_error = np.zeros(n_dim)
    scal = np.load(f'{path}scal.npy')

    for i in range(n_dim):
        print(f'Opening enc_{i}')
        cov_pred = np.load(f'{path}cov_pred_enc{filenames[i]}.npy')
        cov_test = np.load(f'{path}cov_test_enc{filenames[i]}.npy')
        cov_pred = np.transpose(np.transpose(cov_pred)*scal)
        cov_test = np.transpose(np.transpose(cov_test)*scal)
        print(cov_pred.shape)
        k = 0
        for j in index:
            print(f'Starting mean on the {k}° diagonal element')
            temp_a = np.abs((cov_test[:,j] - cov_pred[:,j])/(cov_test[:,j]+10**(-4)))
            temp_b = np.abs(cov_test[:,j] - cov_pred[:,j])
            mean_relative_absolute_error[i,k] = temp_a.mean()
            mean_absolute_error[i,k] = temp_b.mean()
            k += 1
        temp_a = np.abs((cov_test - cov_pred)/(cov_test+10**(-4)))
        temp_b = np.abs(cov_test - cov_pred)
        tot_rel_abs_error[i] = temp_a.mean()
        tot_abs_error[i] = temp_b.mean()

    plt.figure('Compare entrire matrix and diagonal relative error')
    plt.subplot(2,1,1)
    plt.plot([40,80,100,200,300,400], mean_relative_absolute_error.mean(axis=1),
             'ko--', label='mean absolute error')
    plt.ylabel('Digonal MRAE', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('Mean relative absolute error (MRAE)', size=18)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot([40,80,100,200,300,400],tot_rel_abs_error,'ko--',label='mean absolute error')
    plt.ylabel('Entire matrix MRAE', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()

    plt.figure('Compare entrire matrix and diagonal absolute error')
    plt.subplot(2,1,1)
    plt.plot([40,80,100,200,300,400],mean_absolute_error.mean(axis=1),
             'ko--',label='mean absolute error')
    plt.ylabel('Digonal mean absolute error', size=15)
    plt.xlabel('Encoding nodes', size=15)
    plt.title('Compare entrire matrix and diagonal absolute error', size=15)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot([40,80,100,200,300,400],tot_abs_error,'ko--',label='mean absolute error')
    plt.ylabel('Entire matrix mean absolute error', size=12)
    plt.xlabel('Encoding nodes', size=12)
    plt.grid()

def print_mean_std(filename):
    """Print mean values and standard deviation of every element in the covariance matrix.

    Args: 
        filename: string. Filename of covariance matrix.
    """
    cov = np.load(filename)
    for i in range(15):
        print(f'Element {i}')
        print(f'Mean = {cov[:,i].mean():.2e}, Std = {cov[:,i].std():.2e}')

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(123)

    matrix_error_graph('/mnt/c/Users/HP/Desktop/progetto_cmep/data/outputs/')
    element_error('/mnt/c/Users/HP/Desktop/progetto_cmep/data/outputs/', 15)
    element_error_tot('/mnt/c/Users/HP/Desktop/progetto_cmep/data/outputs/')
    matrix_error_graph_bs('/mnt/c/Users/HP/Desktop/progetto_cmep/data/outputs/',
                          ['5_bs40','5_bs80','5_bs100', '5', '5_bs300','5_bs400'])
    print_mean_std('/mnt/c/Users/HP/Desktop/progetto_cmep/data/inputs/cov_tot.npy')

    plt.show()
