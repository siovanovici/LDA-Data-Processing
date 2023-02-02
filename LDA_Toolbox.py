# import libraries
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM


def pre_processing(data, vel_max=0.15, vel_min=-0.15, probe_num=1):
    """Minor data processing to prepare data sets for use in graphing or other post-processing

    Parameters
    ----------
    data : numpy array
        Data file resulting from data_extraction_csv_auto.py
    vel_max : float
        Maximum allowed velocity in the data set
    vel_min : float
        Minimum allowed velocity in the data set
    probe_num : integer
        Selection of which probe data to use (0 or 1, 0 is horizontal and 1 in vertical)

    Returns
    -------
    numpy array
        Returns an array containing filtered data of a single probe, which is structured as: [Time, Transit Time, Velocity]
    """

    # Select the dataset corresponding to the selected probe output
    subset = data[:, probe_num * 3:probe_num * 3 + 3]

    # Remove zero measurements
    subset = subset[~np.all(subset == 0, axis=1)]
    subset = subset[np.nonzero(subset[:, -1])[0], :]
    subset = np.asarray(subset)

    # Apply a low pass filter to the velocity data to reduce signal noise
    subset = subset[subset[:, 2] < vel_max]
    subset = subset[subset[:, 2] > vel_min]

    return subset


def gap_filter(data, T_mingap, T_remove):
    """Filter based  on the gap time in between realizations, data is filtered on both sides of the gap to remove
    velocitiy gradients associated with a passing bubble

    Parameters
    ----------
    data : numpy array
        Output array from the pre_processing function
    T_mingap : float
        Minimum amount of time for something to be defined as a gap
    T_remove : float
        Size of the timeframe in which data point are removed before and after the gap


    Returns
    -------
    numpy array
        Numpy array containing the filtered data set

    """

    # Adjust for units
    T_mingap *= 1
    T_remove *= 1

    # Creates bool array for which values to keep
    bools = np.full(np.shape(data)[0], True)

    # Actual time gaps between realizations
    gaps = data[1::, 0] - data[0:-1, 0]

    for ind in np.argwhere(gaps > T_mingap):

        # Flag data for removal left of the gap
        mask = data[:, 0] - data[ind, 0]
        bools = np.where(~((mask >= -T_remove) & (mask <= 0)), bools, False)

        # Flag data for removal right of the gap
        mask = data[:, 0] - data[ind + 1, 0]
        bools = np.where(~((mask <= T_remove) & (mask >= 0)), bools, False)

    return data[bools]


def BIC_criterion(data, n_limit=4):
    """Computes the most likely number of Gaussian distributions (variables) based on the BIC criterion

    Parameters
    ----------
    data : numpy array
        Output array from the pre_processing function
    n_limit : float
        Maximum number of distribution to check for

    Returns
    -------
    n_optimal : float
        Most likely number of Gaussian distributions, to be used in the GMM_filter
    """

    data = data.reshape(-1, 1)

    # Finding the optimal number of components for the given limit
    bics = []
    min_bic = 0
    counter = 1
    for i in range(n_limit):  # test the AIC/BIC metric
        gmm = GMM(n_components=counter, max_iter=1000, random_state=0, covariance_type='full')
        labels = gmm.fit(data).predict(data)
        bic = gmm.bic(data)
        bics.append(bic)
        if bic < min_bic or min_bic == 0:
            min_bic = bic
            opt_bic = counter
        counter = counter + 1

    # Set the optimal number of components as the BIC optimal:
    n_optimal = opt_bic

    return n_optimal


def GMM_filter(data, n_optimal, select_mode='variance'):
    """Filter based on the GMM machine learning algorithm to separate data based on their velocity distributions

    Parameters
    ----------
    data : numpy array
        Output array from the pre_processing function
    n_optimal : float
        The number of expected Gaussian distributions, can be used together with BIC_criterion for automatic detection
    select_mode : string
        'variance' or 'mean', uses either the variance or the mean of the detected Gaussian distributions to select which distribution to keep

    Returns
    -------
    numpy array
        Numpy array containing the filtered data set
    """

    x = data.reshape(-1, 1)

    # create GMM model object
    gmm = GMM(n_components=n_optimal, max_iter=1000, random_state=10, covariance_type='full')
    # , weights_init=[0.1, 0.9]

    # find useful parameters
    mean = gmm.fit(x).means_
    covs = gmm.fit(x).covariances_
    weights = gmm.fit(x).weights_

    bins = np.arange(-0.5, 0.5, 0.005)

    if select_mode == 'mean':
        fit_num = np.argmax(mean)
    elif select_mode == 'variance':
        var = np.zeros(n_optimal)
        for n in np.arange(n_optimal):
            var[n] = np.sqrt(float(covs[n][0][0]))
        fit_num = np.argmin(var)
    else:
        print('Error: Gaussian fit selection mode not recognized, use either mean or variance (default)')
        exit()

    bin_middle = bins[:-1] + 0.5 * np.abs(bins[1] - bins[0])
    max_bin_counts = np.floor(norm.pdf(bin_middle, float(mean[fit_num][0]), np.sqrt(float(covs[fit_num][0][0]))) * weights[fit_num] * (
                bins[1] - bins[0]) * len(x))

    digit = np.digitize(x, bins)

    x_filtered = np.zeros_like(x)
    integer = 0
    for n in np.arange(1, np.max(digit) + 1):
        subset = x[digit == n]
        counts = max_bin_counts[n - 1].astype(int)

        if counts <= np.size(subset):
            rand_subset = np.random.choice(subset, counts, replace=False)
        else:
            rand_subset = subset

        if np.size(rand_subset) > 0:
            x_filtered[integer:integer + np.size(rand_subset)] = rand_subset[..., np.newaxis]
            integer += np.size(rand_subset)

    x_filtered = np.trim_zeros(x_filtered)
    return x_filtered


def mean_filter(subset, start_T=10, bub_T=120, part_T=180):
    """Simple filter that computes the average particle concentration corrected by a background measurement

    Parameters
    ----------
    subset : numpy array
        Output array from the pre_processing function
    start_T : float
        Start time for the data to be used in this function in seconds, all data < start_T is discarded
    bub_T : float
        Maximum timestamp at which only bubbles are included in the measurement in seconds (i.e. before adding particles)
    part_T : float
        Minimum timestamp at which solid particle have been added and have spread throughout the medium in seconds

    Returns
    -------
    filtered_mean : float
        Average filtered density reading
    """

    # Determines the closest data points to the given timestamps
    start_ind = np.searchsorted(subset[:, 0] / 1000, start_T)
    bub_ind = np.searchsorted(subset[:, 0] / 1000, bub_T)
    part_ind = np.searchsorted(subset[:, 0] / 1000, part_T)

    bub_mean = np.sum(np.abs(1 / subset[start_ind:bub_ind + 1, 2])) / (
            subset[bub_ind, 0] - subset[start_ind, 0]) * 1000
    part_mean = np.sum(np.abs(1 / subset[part_ind::, 2])) / (subset[-1, 0] - subset[part_ind, 0]) * 1000
    filtered_mean = part_mean - bub_mean

    return filtered_mean

