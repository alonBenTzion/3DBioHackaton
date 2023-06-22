import naive_model
import parametric_model
import numpy as np
import argparse
import analyze_interaction
import matplotlib.pyplot as plt

def plot_trajectories(X, complicated= False, title=''):
    '''
    plots the trajectories of the two particles.
    @param T - the time vector
    @param X - the position matrix
    '''

    plt.figure()
    if complicated:
        # plot both series on the same plot
        # use a gradient of colors to show the time evolution
        for i in range(X.shape[0]):
            plt.plot(X[i, 0], X[i, 1], 'o', color=plt.cm.Blues(i / X.shape[0]))
            plt.plot(X[i, 2], X[i, 3], 'o', color=plt.cm.Reds(i / X.shape[0]))
    else:
        # plot both series on the same plot
        plt.plot(X[:, 0], X[:, 1], label='particle 1')
        plt.plot(X[:, 2], X[:, 3], label='particle 2')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def estimate_parameters(r0, r_max, K):
    '''
    estimate the parameters of the model
    '''
    # read the data
    omit_factor_iteration = [2**i for i in range(11)]
    estimated_r0 = []
    estimated_r_max = []
    estimated_K = []
    for num_frames in omit_factor_iteration:
        # X_low_resolution = X_high_resolution[::num_frames+1]
        # compute the parameters
        predicted_r0, predicted_r_max, predicted_K = 2-0.1, 10-1, 0.1-0.01
        # r0, r_max, K = analyze_interaction.estimate_parameters(X_low_resolution)
        estimated_r0.append(abs(r0-predicted_r0))
        estimated_r_max.append(abs(r_max-predicted_r_max))
        estimated_K.append(abs(K-predicted_K))
    plt.figure()
    plt.plot(omit_factor_iteration, estimated_r0, label='r0')
    plt.xlabel('omit factor')
    plt.ylabel('parameter value')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(omit_factor_iteration, estimated_r_max, label='r_max')
    plt.xlabel('omit factor')
    plt.ylabel('parameter value')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(omit_factor_iteration, estimated_K, label='K')
    plt.xlabel('omit factor')
    plt.ylabel('parameter value')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # add command line arguments for low_res_filename, frame to complete, dt, D
    parser = argparse.ArgumentParser()
    parser.add_argument('--high_res_filename', type=str, default='X_high_resolution.npy')
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--D', type=float, default=100)
    args = parser.parse_args()

    # read npy files
    X_high_resolution = np.load(args.high_res_filename)
    plot_trajectories(X_high_resolution, complicated=False, title='high resolution data')
    estimate_parameters(2, 10, 0.1)
    # # return low resolution data, omit time steps
    # # loop over array with multiplied values 1,2,4,8,16,32,64,128,256,512,1024...
    # for num_frames in [2**i for i in range(11)]:
    #     X_low_resolution = X_high_resolution[::num_frames+1]
    #     plot_trajectories(X_low_resolution, complicated=False, title='low resolution data - number of omitted frames: '+ str(num_frames))
    #
    #     # predict the missing frames using the naive model
    #     X_high_naive_model = naive_model.get_high_resolution(X_low_resolution, num_frames,
    #                                                      args.dt, args.D)
    #     plot_trajectories(X_high_naive_model, complicated=False, title='naive model - number of omitted frames: '+ str(num_frames))
    #
    #
    # # X_high_parametric_model = parametric_model.get_high_resolution(X_low_resolution,args.num_of_frames_between,
    # #                                                      args.dt, args.D)
    # # plot_trajectories(X_high_parametric_model, complicated=False, title='parametric model')
    #

