import numpy as np

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size

def Laplace(scaler_c_i, epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon / scaler_c_i
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(scaler_c_i, epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta / scaler_c_i)) *sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)
