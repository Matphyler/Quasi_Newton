import autograd.numpy as np


def random_matrix_wishart(dim, loc=0.0, scale=1.0):
    x = np.random.normal(size=(dim, dim), loc=loc, scale=scale)
    return np.dot(x, x.transpose())


def random_vec_normal(dim, loc=0.0, scale=1.0):
    return np.random.normal(size=dim, loc=loc, scale=scale)


def my_abs(x):  # auto_grad would fail on x = 0 if np.abs() is used.
    if x >= 0:
        return x
    else:
        return -x

