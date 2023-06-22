import numpy as np


<<<<<<< HEAD
def compute_next_frame(position_i_minus_1, position_i, k):
    # compute the point (x1, y1) in the (1/omit_factor) * the distance
    # between the two points
    x1 = ((position_i[0] - position_i_minus_1[0]) / k) + position_i_minus_1[0]
    y1 = ((position_i[1] - position_i_minus_1[1]) / k) + position_i_minus_1[1]
    # compute the point (x2, y2) in the (1/omit_factor) * the distance
    # between the two points
    x2 = ((position_i[2] - position_i_minus_1[2]) / k) + position_i_minus_1[2]
    y2 = ((position_i[3] - position_i_minus_1[3]) / k) + position_i_minus_1[3]
    return np.array([x1, y1, x2, y2])

def compute_next_frame_brawnian_dymn(position_i_minus_1, position_i, k):
    # compute the point (x1, y1) in the (1/omit_factor) * the distance
    # between the two points
    x1 = ((position_i[0] - position_i_minus_1[0]) / k) + position_i_minus_1[0]
    y1 = ((position_i[1] - position_i_minus_1[1]) / k) + position_i_minus_1[1]
    # compute the point (x2, y2) in the (1/omit_factor) * the distance
    # between the two points
    x2 = ((position_i[2] - position_i_minus_1[2]) / k) + position_i_minus_1[2]
    y2 = ((position_i[3] - position_i_minus_1[3]) / k) + position_i_minus_1[3]
    return np.array([x1, y1, x2, y2])


def get_high_resolution(X_low, omit_factor):
=======
def compute_next_frame(position_i_minus_1, position_i, k, dt, D):
    # compute the random forces
    rng = np.random.default_rng()
    R1 = rng.normal(0, 1, 2)
    R2 = rng.normal(0, 1, 2)
    # compute the point (x1, y1) in the (1/omit_factor) * the distance
    # between the two points
    x1 = ((position_i[0] - position_i_minus_1[0]) / k) + \
         position_i_minus_1[0] + (np.sqrt(2 * D * dt) * R1)[0]
    y1 = ((position_i[1] - position_i_minus_1[1]) / k) + \
         position_i_minus_1[1] + (np.sqrt(2 * D * dt) * R1)[1]
    # compute the point (x2, y2) in the (1/omit_factor) * the distance
    # between the two points
    x2 = ((position_i[2] - position_i_minus_1[2]) / k) + \
         position_i_minus_1[2] + (np.sqrt(2 * D * dt) * R2)[0]
    y2 = ((position_i[3] - position_i_minus_1[3]) / k) + \
         position_i_minus_1[3] + (np.sqrt(2 * D * dt) * R2)[1]
    return np.array([x1, y1, x2, y2])


def get_high_resolution(X_low, omit_factor, dt, D):
>>>>>>> origin/main
    X_high = np.zeros((omit_factor * X_low.shape[0] - omit_factor + 1, 4))
    for i in range(0, X_low.shape[0] - 1):
        X_high[i * omit_factor, :] = X_low[i, :]
        for j in range(1, omit_factor):
            position_j_minus_1 = X_high[i * omit_factor + j - 1, :]
            position_j = X_low[i + 1, :]
<<<<<<< HEAD
            X_high[i * omit_factor + j] = compute_next_frame(position_j_minus_1,
                                                             position_j,
                                                             omit_factor -
                                                             j + 1)
=======
            X_high[i * omit_factor + j] = compute_next_frame(
                position_j_minus_1,
                position_j,
                omit_factor -
                j + 1, dt, D)
>>>>>>> origin/main
    X_high[-1, :] = X_low[-1, :]
    print(X_high)
    return X_high

#
# if __name__ == '__main__':
#     get_low_resolution()
