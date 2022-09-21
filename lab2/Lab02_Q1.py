import numpy as np
import Lab02_myFunctions as myf
# Import data to take the standard deviation (std) of
carr = np.loadtxt('cdata.txt')

# Define function that takes in data and compute std using eqn (1)
def std_eqn1(data):
    n = len(data)
    s = 0
    data_mean = np.mean(data)
    for i in range(n):
        s += (data[i] - data_mean)**2
    return np.sqrt(s/(n-1))

# Define function that takes in data and compute std using eqn (2). Add a try-catch for possible sqrt(-) cases and print warning
def std_eqn2(data):
    n = len(data)
    s = 0
    data_mean = np.mean(data)
    for i in range(n):
        s += data[i]**2
    print(s, n*data_mean**2, data_mean)
    try:
        val = np.sqrt((s-n*data_mean**2)/(n-1))
    except RuntimeWarning:
        val = np.nan
        print('Square root of negative!')
    return val

def std_eqn3(data):
    n = len(data)
    s = 0
    data_mean = np.mean(data)
    for i in range(n):
        s += (data[i]+data_mean)*(data[i]-data_mean)
    return np.sqrt(s/(n-1))



# Using both functions, compute the relative error using np.std as the correct answer
if __name__ == '__main__':
    # 1 b), Relative error between the two methods
    expected_std = np.std(carr, ddof=1)
    rel_error_eqn1 = myf.rel_error(expected_std, std_eqn1(carr))
    rel_error_eqn2 = myf.rel_error(expected_std, std_eqn2(carr))
    rel_error_eqn3 = myf.rel_error(expected_std, std_eqn3(carr))
    print('Relative error for speed of light measurements using eqn (1): {0}'.format(rel_error_eqn1))
    print('Relative error for speed of light measurements using eqn (2): {0}'.format(rel_error_eqn2))
    print('Relative error for speed of light measurements using eqn (3): {0}\n'.format(rel_error_eqn3))

    # 1 c), Relative error between the two methods for normal distribution
    # Data set 1
    data_set1 = np.random.normal(0, 1., 2000) # Generate data set
    expected_std_ds1 = np.std(data_set1, ddof=1) # Get 'correct' std from data set
    rel_error_eqn1_ds1 = myf.rel_error(expected_std_ds1, std_eqn1(data_set1)) # Compute relative error between eqn (1) and np.std
    rel_error_eqn2_ds1 = myf.rel_error(expected_std_ds1, std_eqn2(data_set1)) # Compute relative error between eqn (2) and np.std
    rel_error_eqn3_ds1 = myf.rel_error(expected_std_ds1, std_eqn3(data_set1))  # Compute relative error between eqn (2) and np.std
    print('Relative error for data set (mean = 0, std = 1, n = 2000) using eqn (1): {0}'.format(rel_error_eqn1_ds1)) # Print results
    print('Relative error for data set (mean = 0, std = 1, n = 2000) using eqn (2): {0}'.format(rel_error_eqn2_ds1)) # Print results
    print('Relative error for data set (mean = 0, std = 1, n = 2000) using eqn (3): {0}\n'.format(rel_error_eqn3_ds1))  # Print results

    # Data set 2
    data_set2 = np.random.normal(1e7, 1., 2000)
    expected_std_ds2 = np.std(data_set2, ddof=1)
    rel_error_eqn1_ds2 = myf.rel_error(expected_std_ds2, std_eqn1(data_set2))
    rel_error_eqn2_ds2 = myf.rel_error(expected_std_ds2, std_eqn2(data_set2))
    rel_error_eqn3_ds2 = myf.rel_error(expected_std_ds2, std_eqn3(data_set2))
    print('Relative error for data set (mean = 1e7, std = 1, n = 2000) using eqn (1): {0}'.format(rel_error_eqn1_ds2))
    print('Relative error for data set (mean = 1e7, std = 1, n = 2000) using eqn (2): {0}'.format(rel_error_eqn2_ds2))
    print('Relative error for data set (mean = 1e7, std = 1, n = 2000) using eqn (3): {0}'.format(rel_error_eqn3_ds2))


