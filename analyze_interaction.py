"""
tool for analyzing interaction between two molecules. Implements the estimation of K and R from a given interaction
"""
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import main
from sklearn.cluster import KMeans

def Energy_func(d, k, R_0, R_max):
        R_th = 0.5 * (R_0 + R_max)
        result = np.zeros_like(d)

        mask1 = (d < R_th)
        result[mask1] = 0.5 * k * (d[mask1] - R_0) ** 2

        mask2 = (d >= R_th) & (d < R_max)
        result[mask2] = k * (R_th - R_0) ** 2 - 0.5 * k * (d[mask2] - R_max) ** 2

        mask3 = d >= R_max

        result[mask3] = k * (R_th - R_0) ** 2 

        return result

def get_ln_hist(hist, pseudocount_facotr=0.1):
    """
    return histogram using pseudo counts to avoid log(0)
    The pseudocount is computed as the minimum value in the histogram times the pseudocount factor
    """
    # compute a sensible value for the value of the pseudo count
    pseudo_count = np.min(hist[hist > 0]) * pseudocount_facotr
    return np.log(hist + pseudo_count)

def test_get_ln_hist():
        hist = np.array([0, 10, 100, 1000])
        ln_hist = get_ln_hist(hist)
        assert(np.allclose(ln_hist, np.array([-23.02585093, -2.30258509, 4.60517019, 6.90775528])))


def estimate_K_R0_R_max(distances, num_bins=100):
    # Create a histogram of the distances
    hist, bin_edges = np.histogram(distances, bins=num_bins)
    
    # Identify the peak(s) in the histogram
    # peak_indices = np.argmax(hist)
    # peak_distances = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2.0

    # Add a unit test for get_ln_hist, use assert() to check that the output is correct
    # assume get_ln_hist() takes the natural log, not log10
    
    # Define the theoretical histogram function
    # See scientific summary for the derivation of this function
    def get_theoretical_ln_hist_harmonic(d, k, R_0, R_max, C):
        # the first term is correction for the number of states of distance d
        # the second term is the spring potential that we're fitting
        # return np.log(d) - 0.5 * k * (d - R) ** 2 + C
        return np.log(d) - Energy_func(d, k, R_0, R_max) + C

    # plot theoretical ln hist as a function of bin centers
    # from 0.1 to 10.0 for R = 3.0 and k = 0.1
    # d = np.linspace(0.1, 10.0, 100)

    # test_get_ln_hist()
    # Fit the interaction model to the histogram data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ln_hist = get_ln_hist(hist)
    
    # TODO: this is ugly - make it nice
    # ln_hist = ln_hist[0:num_bins*4//5]
    # bin_centers = bin_centers[0:num_bins*4//5]

    # Check that energy function looks like we want
    # test_estimated_function_x_values = np.arange(0, 10.1, 0.1)
    # y = Energy_func(test_estimated_function_x_values, 0.2, 2.1,5)
    # plt.plot(test_estimated_function_x_values, y, label="U(d)")
    
    # do curve fit
    # bound the parmeters to be positive
    popt, _ = curve_fit(get_theoretical_ln_hist_harmonic, # Second variable is covariance matrix
                           bin_centers,
                           ln_hist,
                           p0 = (1,1,2,1),
                           bounds=(0.00001,20)) # TODO make sure p0 and bounds makes sense for the general case
   
    # Extract parameters
    k = popt[0]
    R_0 = popt[1]
    R_max = popt[2]
    # C = popt[3]
    
    # print(popt)
    # #Print parameters and covariance matrix
    # print("K = %f" % k)
    # print("R0 = %f" % R_0)
    # print("R_max = %f" % R_max)
    # print("C = %f" % C)
    # print("Covariance matrix:")
    # print(pcov)
    
    # # plot ln hist as a function of bin centers
    # plt.plot(bin_centers, ln_hist, '.', label="ln(hist)")
    # plt.plot(bin_centers, get_theoretical_ln_hist_harmonic(bin_centers, k, R_0, R_max, C), label="theoretical ln(hist)")
    # plt.plot(bin_centers, get_theoretical_ln_hist_harmonic(bin_centers, 0.1, 2, 5, C), label="true ln(hist)")
    # plt.legend()
    # plt.show()

    return k, R_0, R_max


def label_interaction_frames(trajectory1, trajectory2):
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
    
    # Cut the 50 first and last frames
    distances = distances[50:len(distances)-50]
    
    # Run K-means on the stds, with 2 clusters
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(stds.reshape(-1,1))
    # k1, _ = cluster.KMeans(n_clusters=2).fit(stds.reshape(-1, 1)).cluster_centers_
    
    # Get all the frames with the lower std(K1) = those sumples are suspects to be interaction frames
    cluster_assignments = kmeans.labels_
    interaction_frames = distances[cluster_assignments == 0]  
    
    # lower_cluster = stds < k1
    # interaction_frames = distances[lower_cluster]
    max_dist = np.max(interaction_frames)
    mask = interaction_frames <= 0.75 * max_dist

    # take only the 75% of the frames with the lowest std
    return interaction_frames[mask]

# -> Tuple(float, float, float)
def predict_parameters(trajectories_matrix:np.ndarray)-> Tuple[float,
float, float]:
    """
    Predicts parameters based on the data in a NumPy .npy file.

    Args:
        file_path (str): Path to the NumPy .npy file containing the data.

    Returns:
        object: K, R0, R_Max.
    """
    # Get data as a numpy array
    # data = np.load(file_path)

    # Get trajectories of molecules
    trajectory1 = trajectories_matrix[:, 0:2]
    trajectory2 = trajectories_matrix[:, 2:4]

    # Get the distances of interaction frames
    interaction_dists = label_interaction_frames(trajectory1,trajectory2)

    # Predict params based on heuristic that is detailed in the scientific report
    k, R_0, R_max = estimate_K_R0_R_max(interaction_dists)     

    return k, R_0, R_max

# if __name__ == '__main__':
# file_path = "/Users/alonbentzion/Desktop/university/3DBioHackaton/X_high_resolution.npy"
# #     data = np.load("/Users/alonbentzion/Desktop/university/3DBioHackaton/X_high_resolution.npy")
# #     trajectory1 = data[:, 0:2]
# #     trajectory2 = data[:, 2:4]
# #     # distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)
# #     distances = label_interaction_frames(trajectory1,trajectory2)
# k, R_0, R_max = predict_parameters(file_path)
#    # R_max = estimate_R_max(distances, k, R_0, num_bins=100)

