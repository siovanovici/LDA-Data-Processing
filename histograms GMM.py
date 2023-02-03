import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
import LDA_Toolbox as LDA

BIC = False

data_path = 'data/processed_npy/1-2micron/water_glycerol/curve3/1-2mic_10sccm_wg_16.26mg/54.00_308.50_146.50.npy'

x = LDA.pre_processing(np.load(data_path))

# create the data
#x = np.concatenate((np.random.normal(-0.05, 0.01, 500), np.random.normal(-0.09, 0.05, 500)))
#x = np.concatenate((np.random.normal(5, 4, 1000), np.random.normal(10, 2, 1000), np.random.normal(-5, 3, 1000)))
#x = np.random.normal(5, 5, 1000)
#x = np.random.normal(10, 2, 1000)

x = x[:,2].reshape(-1, 1)

# Plot 2
bins = np.arange(-0.5, 0.5, 0.005)
test = plt.hist(x, density=False, bins=bins)
plt.xlim(-0.5, 0.5)
plt.xlabel("Velocity [m/s]")
plt.ylabel("Density")
plt.show()


# BIC criterion computation
bics = []
min_bic = 0
counter=1
for i in range (10): # test the AIC/BIC metric between 1 and 10 components
  gmm = GMM(n_components = counter, max_iter=1000, random_state=0, covariance_type='full')
  labels = gmm.fit(x).predict(x)
  bic = gmm.bic(x)
  bics.append(bic)
  if bic < min_bic or min_bic == 0:
    min_bic = bic
    opt_bic = counter
  counter = counter + 1

# plot the evolution of BIC
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1,2,1)

plt.plot(np.arange(1,11), bics, 'o-', label='BIC')
plt.legend(frameon=False, fontsize=15)
plt.xlabel('Number of components')
plt.ylabel('Information criterion')
plt.xticks(np.arange(0,11, 2))
plt.title('Opt. components = '+str(opt_bic))
n_optimal = opt_bic
n_optimal = 2

gmm = GMM(n_components = n_optimal, max_iter=1000, random_state=10, covariance_type = 'full')

# find useful parameters
mean = gmm.fit(x).means_
covs  = gmm.fit(x).covariances_
weights = gmm.fit(x).weights_

x_axis = np.arange(-0.5, 0.5, 0.001)
y_axis = np.zeros((np.size(x_axis), n_optimal))

for n in np.arange(n_optimal):
  y_axis[:, n] = norm.pdf(x_axis, float(mean[n][0]), np.sqrt(float(covs[n][0][0])))*weights[n] # gaussians

ax = fig.add_subplot(1,2,2)
# Plot 2
bins = np.arange(-0.5, 0.5, 0.005)
test = plt.hist(x, density=True, bins=bins)
plt.plot(x_axis, y_axis, lw=3, c='C0')
plt.plot(x_axis, np.sum(y_axis, axis=1), lw=3, c='C2', ls='dashed')
plt.xlim(-0.5, 0.5)

plt.xlabel("X")
plt.ylabel("Density")

plt.subplots_adjust(wspace=0.3)
plt.show()
plt.close('all')

var = np.zeros(n_optimal)
for n in np.arange(n_optimal):
    var[n] = np.sqrt(float(covs[n][0][0]))
fit_num = np.argmin(var)
print(var)
print(mean)

print(fit_num)
bin_middle = bins[:-1] + 0.5 * np.abs(bins[1] - bins[0])
max_bin_counts = np.floor(
  norm.pdf(bin_middle, float(mean[fit_num][0]), np.sqrt(float(covs[fit_num][0][0]))) * weights[fit_num] * (
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

plt.hist(x, density=False, bins=bins, alpha=0.5, label='original data')
plt.hist(x_filtered, density=False, bins=bins, alpha=0.5, label='trimmed data')
plt.plot(x_axis, y_axis * (test[1][1] - test[1][0]) * len(x), lw=3, c='C0', label='GMM fits')
plt.plot(x_axis, np.sum(y_axis, axis=1)* (test[1][1] - test[1][0]) * len(x), lw=3, c='C2', ls='dashed', label='sum GMM fits')
plt.xlabel('X')
plt.ylabel('Counts')
plt.legend()
plt.xlim(-0.5, 0.5)
plt.show()