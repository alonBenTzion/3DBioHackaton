import numpy as np
import analyze_interaction

frame_max, frame_min = 10, -10
kT = 0.6 # Boltzmann constant kcal/mol at room temperature
D = 1.0 # diffusion coefficient

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

# compute the pseudo force that is applied to the particle at time step i-1 by
# the same particle at time step i normalized by a factor of k
def compute_pseudo_force(position_i_minus_1, position_i, k):
    return (position_i - position_i_minus_1) / k

# compute the force that is applied to the particle by the other particle based on their
# distance and the spring constant and boundry conditionsof the interaction
def compute_force(x1, x2, k, r0, upper_bounded=True, r_max=10):
    # compute the force on each particle
    r = np.linalg.norm(x1 - x2)
    if upper_bounded:
        F_magnitude = get_harmonic_potential_derivative_upper_bounded(r, r0, k,
                                                                      r_max)
    else:
        F_magnitude = harmonic_potential_derivative(r, r0, k)  # units of kT/A

    # compute the force on each particle
    F1 = -F_magnitude * (x1 - x2) / r
    F2 = -F_magnitude * (x2 - x1) / r
    return F1, F2

def harmonic_potential_derivative(r, r0, k):
    '''
    computes the derivative of the harmonic potential where k=1 kT/A^2.
    @param r - the position of the particle
    @param r0 - the equilibrium position of the particle (rest length)
    @param k - the spring constant
    '''
    return k * (r - r0)

def get_harmonic_potential_derivative_upper_bounded(r, r0, k, r_max):
    '''
    computes the derivative of the harmonic potential where k=1 kT/A^2.
    @param r - the position of the particle
    @param r0 - the equilibrium position of the particle (rest length)
    @param k - the spring constant
    @param r_min - the minimum value of the potential
    @param r_max - the maximum value of the potential
    '''
    decay_threshold = 0.5 * (r0 + r_max)
    F_decay_threshold = harmonic_potential_derivative(decay_threshold, r0, k)
    if  r > r_max:
        return 0
    elif r > decay_threshold:
        # interploate from decay threshold, where the potential is
        # F_decay_threshold to r_max, where the potential is zero
        return F_decay_threshold * (r_max - r) / (r_max - decay_threshold)
    else:
        return harmonic_potential_derivative(r, r0, k)

def compute_dx(dt, D, k, r0, x1, x2, upper_bounded=True, r_max=10):
    '''
    computes the change in position of the two particles based on the
    previous positions.
    @param dt - the time step
    @param D - the diffusion coefficient
    @param k - the spring constant
    @param r0 - the equilibrium position of the particle (rest length)
    @param x1 - the position of the first particle
    @param x2 - the position of the second particle
    '''
    # compute the force on each particle
    r = np.linalg.norm(x1 - x2)
    if upper_bounded:
        F_magnitude = get_harmonic_potential_derivative_upper_bounded(r, r0, k, r_max)
    else:
        F_magnitude = harmonic_potential_derivative(r, r0, k) # units of kT/A
    F1 = -F_magnitude * (x1 - x2) / r
    F2 = -F_magnitude * (x2 - x1) / r
    # compute the random forces
    rng = np.random.default_rng()
    R1 = rng.normal(0, 1, 2)
    R2 = rng.normal(0, 1, 2)
    # compute the new positions
    dx1 = dt * D / kT * F1 + np.sqrt(2 * D * dt) * R1
    dx2 = dt * D / kT * F2 + np.sqrt(2 * D * dt) * R2
    return dx1, dx2

def get_high_resolution(X_low, omit_factor, dt, D):
    # k = 0.1 # TODO: change to estimation of
    # r0 = 2
    # r_max = 10

    k, r0, r_max = analyze_interaction.predict_parameters(X_low)

    X_high = np.zeros((omit_factor * X_low.shape[0] - omit_factor + 1, 4))
    for i in range(0, X_low.shape[0] - 1):
        X_high[i * omit_factor, :] = X_low[i, :]
        for j in range(1, omit_factor):
            position_j_minus_1 = X_high[i * omit_factor + j - 1, :]
            position_j = X_low[i + 1, :]

            new_position_based_on_next_position = compute_next_frame(
                position_j_minus_1,
                position_j,
                omit_factor -
                j + 1, dt, D)


            dx1, dx2 = compute_dx(dt, D, k, r0, position_j_minus_1[0:2], position_j_minus_1[2:4], r_max)
            new_position_based_on_interaction = position_j_minus_1 + np.hstack((dx1, dx2))

            # compute element-wise average of the two positions
            X_high[i * omit_factor + j, :] = (new_position_based_on_next_position + new_position_based_on_interaction) / 2

            # pseudo_force = compute_pseudo_force(position_j_minus_1, position_j, omit_factor - j + 1)
            # F1, F2 = compute_force(position_j_minus_1[0:2], position_j_minus_1[2:4], k, r0, upper_bounded=True, r_max=r_max)

            # normalize F1 and F2 by the number of frames that they are acting on
            # F1 = F1 / (omit_factor - j + 1)
            # F2 = F2 / (omit_factor - j + 1)

            # sum the forces that are acting on the particle
            # F1 = F1 + pseudo_force[0:2]
            # F2 = F2 + pseudo_force[2:4]
            #
            # # compute the new positions and add a random component
            # # compute the random forces
            # rng = np.random.default_rng()
            # R1 = rng.normal(0, 1, 2)
            # R2 = rng.normal(0, 1, 2)
            # # compute the new positions
            # dx1 = dt * D / kT * F1 + np.sqrt(2 * D * dt) * R1
            # dx2 = dt * D / kT * F2 + np.sqrt(2 * D * dt) * R2

            # update the positions
            #X_high[i * omit_factor + j, :] = position_j_minus_1 + np.hstack((dx1, dx2))

            # update all coordinates if they are above a certain value
            X_high[i * omit_factor + j, :] = np.where(
                X_high[i * omit_factor + j, :] > frame_max, frame_max,
                X_high[i * omit_factor + j, :])
            X_high[i * omit_factor + j, :] = np.where(
                X_high[i * omit_factor + j, :] < frame_min, frame_min,
                X_high[i * omit_factor + j, :])
    X_high[-1, :] = X_low[-1, :]
    return X_high



