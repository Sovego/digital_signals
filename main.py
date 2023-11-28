import scipy
from numpy import cos, sin, sign
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def a_0(T, t, x, A):
    return 2/T*quad(lambda t: x(t, A,T), t, t+T)[0]


def a_n(t, T, n, w, x, A):
    return 2/T*quad(lambda t: x(t, A,T) * cos(n * 2 * np.pi / T * t), t, t+T)[0]


def b_n(t, T, n, w, x, A):
    return 2/T*quad(lambda t: x(t, A,T) * sin(n * 2 * np.pi / T * t), t, t+T)[0]


def furie(N, t_0, T, function_x, num_t, mas_t, A):
    """
    :param N: Length of the sum of the series
    :param t_0: Initial moment of time
    :param T: Period
    :param function_x: Function for approximation
    :param num_t: Number of t
    :param mas_t: massive of t
    :param A: Amplitude
    :return: List of dots
    """
    a = []
    w = 2 * np.pi / T
    for t in range(num_t):
        result = 0
        for n in range(1, N):
            result += (a_n(t_0, T, n, w, function_x, A) * cos(n * w * mas_t[t]) + b_n(t_0, T, n, w, function_x, A) *
                       sin(n * w * mas_t[t]))
        a.append((a_0(T, t_0, function_x, A))/2 + result)
    return a


def square_signal(t, A, T):
    return A * scipy.signal.square(2 * np.pi / T * t)


x = np.linspace(-10, 10, 50)
plt.plot(x, (furie(10, -10, 20, square_signal, 50, x, 2)))
plt.plot(x, square_signal(x, 2,20))
plt.show()
