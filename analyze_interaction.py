"""
tool for analyzing interaction between two molecules. Implements the estimation of K and R from a given interaction
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import main


def estimate_K_and_R(trajectory1, trajectory2, num_bins=100):
    # Compute the distances between the two trajectories
    distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)

    # Create a histogram of the distances
    hist, bin_edges = np.histogram(distances, bins=num_bins)

    # Identify the peak(s) in the histogram
    peak_indices = np.argmax(hist)
    peak_distances = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2.0

    # Estimate the range constant (R)
    R = peak_distances.max()

    # Define the interaction model function (e.g., exponential decay)
    def interaction_model(distance, K):
        return np.exp(-distance / R) * K

    def get_ln_hist(hist, pseudocount_facotr=0.1):
        """
        return histogram using pseudo counts to avoid log(0)
        The pseudocount is computed as the minimum value in the histogram times the pseudocount factor
        """
        # compute a sensible value for the value of the pseudo count
        pseudo_count = np.min(hist[hist > 0]) * pseudocount_facotr
        return np.log(hist + pseudo_count)

    # Add a unit test for get_ln_hist, use assert() to check that the output is correct
    # assume get_ln_hist() takes the natural log, not log10
    def test_get_ln_hist():
        hist = np.array([0, 10, 100, 1000])
        ln_hist = get_ln_hist(hist)
        assert(np.allclose(ln_hist, np.array([-23.02585093, -2.30258509, 4.60517019, 6.90775528])))

    # Define the theoretical histogram function
    # \see scientific summary for the derivation of this function
    def get_theoretical_ln_hist_harmonic(d, k, R, C):
        # the first term is correction for the number of states of distance d
        # the second term is the spring potential that we're fitting
        # return np.log(d) - 0.5 * k * (d - R) ** 2 + C
        return np.log(d) - main.harmonic_potential(d, R, k) + C

        # Define the theoretical histogram function
        # \see scientific summary for the derivation of this function

    def get_theoretical_ln_hist_harmonic_bounded(d, k, R, R_max, C):
        # the first term is correction for the number of states of distance d
        # the second term is the spring potential that we're fitting
        # TODO: this is buggy - we must use the potential not the derivative
        return np.log(d) + main.get_harmonic_potential_derivative_upper_bounded() + C

    # plot theoretical ln hist as a function of bin centers
    # from 0.1 to 10.0 for R = 3.0 and k = 0.1
    d = np.linspace(0.1, 10.0, 100)

    # test_get_ln_hist()
    # Fit the interaction model to the histogram data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ln_hist = get_ln_hist(hist)
    # do curve fit
    # bound the parmeters to be positive
    popt, pcov = curve_fit(get_theoretical_ln_hist_harmonic,
                           bin_centers,
                           ln_hist)
    #, bounds=(0, 20))

    # Extract the spring constant (K) from the fitted parameters
    k = popt[0]
    R = popt[1]
    C = popt[2]
    print("K = %f" % k)
    print("R = %f" % R)
    print("C = %f" % C)
    print("Covariance matrix:")
    print(pcov)
    # plot ln hist as a function of bin centers
    plt.plot(bin_centers, ln_hist, '.', label="ln(hist)")
    plt.plot(d, get_theoretical_ln_hist_harmonic(d, k, R, C), label="theoretical ln(hist)")
    plt.plot(d, get_theoretical_ln_hist_harmonic(d, 0.1, 2, C), label="true ln(hist)")
    plt.legend()
    plt.show()

    return k, R


def label_interaction_frames(trajectory1, trajectory2, sklearn=None):
    """
    label frames as interacting or non-interacting
    :param trajectory1:
    :param trajectory2:
    :return:
    """
    distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)
    # for each frame, from 50 to the -50, calculate the std of the distances
    stds = []
    for i in range(50, len(distances) - 50):
        std = np.std(distances[i-50:i+50])
        stds.append(std)
    stds = np.array(stds)
    # run K-means on the stds, with 2 clusters
    k1, k2 = sklearn.cluster.KMeans(n_clusters=2).fit(stds.reshape(-1, 1)).cluster_centers_
    # get only the frames fro the lower cluster
    lower_cluster = stds < k1
    # take only the 75% of the frames with the lowest std
    lower_cluster = lower_cluster[:int(len(lower_cluster) * 0.75)]


            

if __name__ == '__main__':
    data = np.load("/Users/michaleldar/Documents/year3/3DStructure/3DBioHackaton/X_high_resolution.npy")
    trajectory1 = data[:80000:, 0:2]
    trajectory2 = data[:80000:, 2:4]
    distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)
    k, R = estimate_K_and_R(trajectory1, trajectory2)

