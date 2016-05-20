import numpy as np

def robust_pca(X, lambda1, max_iter, verbose=False):
    def J(X):
        n = np.linalg.norm(X)
        m = np.max(np.fabs(X))
        return max(n, m / lambda1)

    def soft_threshold(c, M):
        f0 = M < -c
        f1 = M > c
        f2 = np.logical_not(np.logical_or(f0, f1))
        M[f0] += c
        M[f1] -= c
        M[f2] = 0.

    mu_cur, rho = 1.25, 1.6
    Y = X / J(X)
    norm_X = np.linalg.norm(X)
    E_cur = np.zeros(X.shape)
    c0,c1 = 1e9, 1e9
    eps1,eps2 = 1e-7, 1e-5
    i = 0

    while True:
        if not i < max_iter:
            break

        # stopping criteria
        if c0 < eps1 and c1 < eps2:
            break

        mu_inv = 1. / mu_cur
        C = X + mu_inv * Y

        # A update
        U, s, V = np.linalg.svd(C - E_cur, full_matrices=False)
        soft_threshold(mu_inv, s)
        A = np.dot(U, np.dot(np.diag(s), V))

        # E update
        E_succ = C - A
        soft_threshold(lambda1 * mu_inv, E_succ)

        # Y update
        Yd = X - A - E_succ
        Y += mu_cur * Yd

        # update stopping criterion constants
        c0 = np.linalg.norm(Yd) / norm_X
        c1 = mu_cur * np.linalg.norm(E_succ-E_cur) / norm_X

        # mu update
        if c1 < eps2:
            mu_succ = rho * mu_cur
        else:
            mu_succ = mu_cur

        mu_cur = mu_succ
        E_cur = E_succ
        i += 1

    if verbose:
        print "{} iterated".format(i)

    return (X-E_cur, E_cur)

if __name__ == "__main__":
    X = np.array([[1,9,1],[9,2,9],[3,9,3]])
    lambda1 = 0.5
    max_iter = 1000
    A, E = robust_pca(X, lambda1, max_iter, verbose=True)
    print A
    print E
