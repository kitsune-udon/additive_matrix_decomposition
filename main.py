import numpy as np
import imageio,robust_pca

def separate_noise(X):
    n_samples, n_pixels = X.shape
    lambda1 = 5 * 1e-2
    max_iter = 10000
    return robust_pca.robust_pca(X, lambda1, max_iter, True)

def make_original_image(image_size, n_samples, noise_prob):
    width, height = image_size
    X = np.zeros((n_samples, width*height))
    for k in xrange(n_samples):
        t = float(k) / n_samples
        for j in xrange(height):
            y = float(j) / height
            for i in xrange(width):
                idx = j * width + i
                x = float(i) / width
                if np.random.random() < noise_prob:
                    X[k, idx] = 0.
                else:
                    #phase = 2.*np.pi*(x+y+t)
                    phase = 2.*np.pi*(x+y+t+0.25)
                    X[k, idx] = 0.5 + 0.45 * np.sin(phase)
    return X

def output(X, A, E, image_size, n_samples):
    def vertical_concat(Y, size, n):
        w, h = size
        v_sep = 5
        Z = 0.5 * np.ones((h*n+v_sep*(n-1), w))
        for i in xrange(n):
            y = h*i+v_sep*i
            Z[y:y+h, 0:w] = Y[i].reshape((h, w))
        return Z
    E0 = 0.5 + 0.5 * E
    X_all = vertical_concat(X, image_size, n_samples)
    A_all = vertical_concat(A, image_size, n_samples)
    E_all = vertical_concat(E0, image_size, n_samples)
    h_sep = 5
    SEP = 0.5 * np.ones((X_all.shape[0], h_sep))
    R = np.c_[X_all, SEP, A_all, SEP, E_all]
    loc = "results.png"
    imageio.write_image_L(loc, R)

image_size = (160, 90)
n_samples = 10
noise_prob = 0.005

np.random.seed(0)
X = make_original_image(image_size, n_samples, noise_prob)
A, E = separate_noise(X)
output(X, A, E, image_size, n_samples)
