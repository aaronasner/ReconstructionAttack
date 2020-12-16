# Author : Aaron Asner
# Reconstruction attack on binary mechanism for normalized L0 norm of a private vector of bits,
#  using L-BFGS method of Szegedy et al. 2013 with varying initial guesses and n subset queries on X[1:i] for i=1,...,n
#  asking 2^n subset queries was attempted, but for even small n it is too expensive in memory and time (n=20 required 1TB of dynamic memory and crashed)
#  l-bfgs-b is used for implementation: the limited memory, bounded box ([0,1]) variant of Broyden–Fletcher–Goldfarb–Shanno algorithm
#  due to instability of l-bfgs-b, reconstruction attack performed 20 times and average is plotted
#  plots show fraction of private dataset recovered for increasing n from 100 to 1000 by hundreds
#  reconstruction attack for datasets larger than 1000 takes too long to compute



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from math import exp


# privacy budget
eps = .5


def query(X, n):
    # calculate all noisy counts for n subset queries: f(X[0:i + 1]) for i = 1,...,n
    return np.array([noise_injection(i, X) for i in range(n)])


def noise_injection(t, X):
    # return noisy count to query on subset X[0:t + 1] using eps-DP binary mechanism
    return np.sum(np.array([x if np.random.uniform() < exp(eps) / (1 + exp(eps)) else 1 - x for x in X[0:(t + 1)]]))


def attack(Y, x0, n):
    f = get_f(np.tri(n).T, Y)
    ans = fmin_l_bfgs_b(f, x0, bounds=[(0, 1)] * n, approx_grad=True)
    # if >.5 count as a 1, else a 0
    return np.where(ans[0] > 0.5, 1, 0)


def get_f(A, Y):
    def f(x):
        # Hamming distance between current iterate and reconstruction
        return np.linalg.norm(Y - np.dot(x, A), 1)
    return f


def main():

    a1, a2, a3, a4 = ([] for i in range(4))
    sizes = [(i + 1) * 100 for i in range(10)]

    # loop through varying sizes of datasets
    for n in sizes:

        c1, c2, c3, c4 = ([] for i in range(4))
        print(n)
        for i in range(20):

            # generate random bitvector of size n (private)
            X = np.random.randint(2, size=n)

            # results of n subset queries (public)
            Y = query(X, n)

            # reconstruct with initial guess of [.5 for i in range(n)]
            r1 = attack(Y, np.repeat(0.5, n), n)
            c1.append(np.linalg.norm(X - r1, 1))

            # reconstruct with initial guess of random binary vector of size n
            r2 = attack(Y, np.random.randint(2, size=n), n)
            c2.append(np.linalg.norm(X - r2, 1))

            # reconstruct with initial guess of random vector of size n with entries taken uniformly from [0, 1)
            r3 = attack(Y, np.random.rand(n), n)
            c3.append(np.linalg.norm(X - r3, 1))

            # reconstruct with initial guess of gradient
            r4 = attack(Y, np.gradient(Y), n)
            c4.append(np.linalg.norm(X - r4, 1))

        # compute average L0 norm (number of 1s) and normalize
        a1.append((sum(c1) / len(c1)) / n)
        a2.append((sum(c2) / len(c2)) / n)
        a3.append((sum(c3) / len(c3)) / n)
        a4.append((sum(c4) / len(c4)) / n)

    # plot results
    plt.plot(sizes, [1 - a for a in a1], 'b-', label='initial guess of {0,1}^n')
    plt.plot(sizes, [1 - a for a in a2], 'r-', label='initial guess of {.5}^n')
    plt.plot(sizes, [1 - a for a in a3], 'g-', label='initial guess of [0,1]^n')
    plt.plot(sizes, [1 - a for a in a4], 'k-', label='initial guess of gradient')
    plt.legend(loc='upper right')
    plt.title('Binary Mechanism Reconstruction eps={}'.format(eps))
    plt.xlabel('Dataset size n')
    plt.ylabel('Fraction of bits reconstructed')
    plt.show()


if __name__ == '__main__':
    main()