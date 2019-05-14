import numpy as np


def sharpen_npy(x, T):
    numerator = x ** (1 / T)
    return numerator / numerator.sum(axis=1, keepdims=True)


def lin_comb(a, b, frac_a): return (frac_a * a) + (1 - frac_a) * b


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha, x1.shape[0])
    beta = np.maximum(beta, 1 - beta)
    return lin_comb(x1, x2, beta), lin_comb(y1, y2, beta)
