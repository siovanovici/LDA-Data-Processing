import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import fftpack
from scipy import signal
import LDA_Toolbox as LDA

# Set parameters
window_size = 5000      # Number of data points used in the moving window used to compute a time signal
interval_size = 5000     # Number of data point by which the window moves per computation step, e.g. if the first
# window is 0:window_size then the next window will be interval_size:window_size+interval_size
probe = 1               # Probe selection, 0 is horizontal and 1 is vertical
plot = 'together'       # together or individual, used to set graphing options

# Parameters used for FFT (can be ignored)
interpolate_count = 500 * 100
skip_point = 160 * 100

# Define dataset
data_path = 'data/processed_npy/1-2micron/water_glycerol/long/1-2mic_5sccm_wg_10.29mg/54.00_308.50_146.50.npy'
data = np.load(data_path)

subset = LDA.pre_processing(data, probe_num=probe)

plt.plot(subset[:, 0] / 1000, subset[:, 2], linestyle='', marker='+', markersize=5)
plt.title('')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [s]')
plt.show()

# Pre-build arrays to save performance
sample_count = np.shape(subset)[0]
divs = np.floor(sample_count / interval_size).astype(int)
data_rate = np.zeros((divs, 2))
data_rate_org = np.zeros_like(data_rate)
density_org = np.zeros_like(data_rate)
density = np.zeros_like(data_rate)
test1 = np.zeros_like(data_rate)
test2 = np.zeros_like(data_rate)
vel = np.zeros_like(data_rate)

count = 0
for i in np.arange(1, divs + 1):

    if i * interval_size < window_size:
        data_window = subset[0:interval_size * i, :]

    else:
        data_window = subset[i * interval_size - window_size:i * interval_size, :]

    # Calculate the optimal number DOF for GMM
    #n_opt = LDA.BIC_criterion(data_window[:, 2])
    n_opt = 2

    # Set the time stamp
    data_rate[i - 1, 0] = data_window[-1, 0] / 1000
    density[i - 1, 0] = data_rate[i - 1, 0]
    density_org[i - 1, 0] = data_rate[i - 1, 0]
    data_rate_org[i - 1, 0] = data_rate[i - 1, 0]

    test1[i - 1, 0] = data_rate[i - 1, 0]
    test2[i - 1, 0] = data_rate[i - 1, 0]
    vel[i - 1, 0] = data_rate[i - 1, 0]

    if n_opt == 1:
        data_rate[i - 1, 1] = 0
        density[i - 1, 1] = 0

    else:
        data_window_filtered = LDA.GMM_filter(data_window[:, 2], n_optimal=n_opt)
        dt = data_window[-1, 0] - data_window[0, 0]
        data_rate[i - 1, 1] = np.size(data_window_filtered) / (dt / 1000)
        density[i - 1, 1] = np.sum(np.abs(1 / data_window_filtered)) / (dt / 1000)

        data_rate_org[i - 1, 1] = np.size(data_window[:, 2]) / (dt / 1000)
        density_org[i - 1, 1] = np.sum(np.abs(1 / data_window[:, 2])) / (dt / 1000)

        test1[i - 1, 1] = np.size(data_window_filtered)
        test2[i - 1, 1] = dt / 1000

        vel[i - 1, 1] = np.mean(data_window_filtered)

    print(f' Progress: {count / np.max(np.arange(1, divs + 1)) * 100:.2f}% done')
    count += 1

mean_val = np.mean(np.sum(np.abs(1 / subset[0:17500, 2])) / subset[0:17500, 0][-1] * 1000)
density_mean = density_org[:, 1] - mean_val
density_mean[density_mean < 0] = 0
print(mean_val)

print(np.average(data_rate[:, 1]))

if plot == 'individual':
    plt.plot(data_rate[:, 0] / 60, data_rate[:, 1], linestyle='--', marker='+', markersize=5)
    plt.title('Data rate')
    plt.ylabel('Data Rate [1/s]')
    plt.xlabel('Time [min]')
    plt.show()

    plt.plot(density[:, 0] / 60, density[:, 1], linestyle='--', marker='+', markersize=5)
    plt.title('Data rate')
    plt.ylabel('Data Rate [1/s]')
    plt.xlabel('Time [min]')
    plt.show()

elif plot == 'together':
    fig, axs = plt.subplots(2, 1, sharex='all')
    axs[0].plot(data_rate[:, 0] / 60, data_rate[:, 1], linestyle='--', marker='+', markersize=5, label='GMM filter')
    axs[0].plot(data_rate_org[:, 0] / 60, data_rate_org[:, 1], linestyle='--', marker='+', markersize=5, label='unfiltered')
    #axs[0].plot(data_rate_org[:, 0] / 60, data_rate_org[:, 1], linestyle='--', marker='+', markersize=5,label='mean filter')

    axs[0].set_title('Data Rate (vert)')
    axs[0].set_ylabel('Data Rate [1/s]')
    axs[0].legend()

    axs[1].plot(density[:, 0] / 60, density[:, 1], linestyle='--', marker='+', markersize=5, label='GMM filter')
    axs[1].plot(density_org[:, 0] / 60, density_org[:, 1], linestyle='--', marker='+', markersize=5, label='unfiltered')
    axs[1].plot(density_org[:, 0] / 60, density_mean, linestyle='--', marker='+', markersize=5, label='mean filter')

    axs[1].set_title('Relative Particle Concentration (vert)')
    axs[1].set_xlabel('Time [m]')
    axs[1].set_ylabel('Relative Particle Concentration [-]')
    axs[1].legend()
    plt.show()


exit()

# Extra code that was used to de-trend & resample the signal and to perform a FFT, currently abandoned

plt.plot(test1[:, 0] / 60, test1[:, 1], linestyle='--', marker='+', markersize=5)
plt.title('Number of Data points (original = 3000)')
plt.ylabel('Data points [1/s]')
plt.xlabel('Time [min]')
plt.show()

plt.plot(test2[:, 0], test2[:, 1], linestyle='--', marker='+', markersize=5)
plt.title('dt')
plt.ylabel('Data Rate [1/s]')
plt.xlabel('Time [min]')
plt.show()

plt.plot(vel[:, 0], vel[:, 1], linestyle='--', marker='+', markersize=5)
plt.title('vel')
plt.ylabel('Data Rate [1/s]')
plt.xlabel('Time [min]')
plt.show()


f = interpolate.interp1d(data_rate[:, 0], window_size / data_rate[:, 1])

xnew = np.linspace(np.min(data_rate[:, 0]), np.max(data_rate[:, 0]), interpolate_count)
ynew = f(xnew)

#plt.plot(data_rate[:, 0], window_size / data_rate[:, 1], 'o', xnew, ynew, '-')
#plt.show()

plt.plot(xnew / 60, ynew, label='Resampled Data')
plt.plot(xnew[skip_point:-1] / 60, ynew[skip_point:-1], label='Resampled an Clipped Data')
plt.title('Resampling and Data Clipping')
plt.ylabel('Data Rate [1/s]')
plt.xlabel('Time [min]')
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, sharex='all')
axs[0].plot(xnew[skip_point:-1] / 60, ynew[skip_point:-1])
axs[0].set_title('Clipped Data')
axs[0].set_ylabel('')

axs[1].plot(xnew[skip_point:-1] / 60, signal.detrend(ynew[skip_point:-1]))
axs[1].set_title('Fluctuations')
axs[1].set_ylabel('')

axs[2].plot(xnew[skip_point:-1] / 60, ynew[skip_point:-1]-signal.detrend(ynew[skip_point:-1]))
axs[2].set_title('Trend')
axs[2].set_xlabel('Time [m]')
axs[2].set_ylabel('')



new_data_x = xnew[skip_point:-1]
new_data_y = ynew[skip_point:-1]

# Nearest size with power of 2
size = 2 ** np.ceil(np.log2(2*len(new_data_y) - 1)).astype('int')

# Normalized data
ndata_y = new_data_y - np.mean(new_data_y)
ndata_y = signal.detrend(new_data_y)

# Variance
var = np.var(ndata_y)

yf = fftpack.fft(ndata_y, new_data_x.size)
amp = np.abs(yf) # get amplitude spectrum
freq = fftpack.fftfreq(new_data_x.size, new_data_x[1] - new_data_x[0])
plt.figure(figsize=(10, 6))

plt.plot(freq[0:freq.size//2], (2/amp.size)*amp[0:amp.size//2])
plt.title('FFT of the Signal')
plt.ylabel('')
plt.xlabel('Frequency [Hz]')
plt.show()

acorr = np.correlate(ndata_y, ndata_y, 'full')[len(ndata_y)-1:]
acorr = acorr / var / len(ndata_y)

plt.plot(np.arange(np.size(acorr)) * (new_data_x[1] - new_data_x[0]), acorr)
#plt.plot(np.arange(np.size(acorr)) * (new_data_x[1] - new_data_x[0]), 1.95 / np.sqrt(new_data_x.size - np.arange(np.size(acorr))))
plt.title('Autocorrelation of the Signal')
plt.ylabel('')
plt.xlabel('Tau [s]')
plt.show()