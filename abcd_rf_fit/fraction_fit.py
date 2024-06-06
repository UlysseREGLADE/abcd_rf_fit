import numpy as np
import matplotlib.pyplot as plt

from scipy.special import comb
from utils import complex_fit

def fit_rational_fraction_pole_basis(x, y, n, poles = None, den = None):
    if den is None:
        den = np.ones(x.shape)
    
    if poles is None:
        poles = np.linspace(np.min(x), np.max(x), n+2, dtype=complex)
        poles = 0.5*(poles[1:] + poles[:-1])
        poles += np.mean(np.diff(poles))*1j*(-1)**(np.arange(n+1))
    
    pole_mesh, x_mesh = np.meshgrid(poles, x)

    x_design = (1/(x_mesh-pole_mesh)).T

    xx = (x_design * np.abs(den) ** -2) @ x_design.T
    xdyx = (y * x_design * np.abs(den) ** -2) @ x_design.T
    xdycx = (np.conj(y) * x_design * np.abs(den) ** -2) @ x_design.T
    xdy2x = (np.abs(y) ** 2 * x_design * np.abs(den) ** -2) @ x_design.T

    up_right = np.linalg.inv(xx) @ xdyx
    bottom_left = np.linalg.inv(xdy2x) @ xdycx

    to_diag = np.zeros((2*(n+1), 2*(n+1)), dtype=complex)
    to_diag[:n+1, n+1:] = up_right
    to_diag[n+1:, :n+1] = bottom_left

    v, w = np.linalg.eig(to_diag)

    solution = w[:, np.argmin(np.abs(1 - v))]

    num = x_design.T @ solution[:n+1]
    den = x_design.T @ solution[n+1:]

    return num, den

def fit_rational_fraction_polynomial_basis(x, y, n, den = None):
    if den is None:
        den = np.ones(x.shape)

    n_mesh, x_mesh = np.meshgrid(np.arange(n+1), x)

    x_design = (x_mesh**n_mesh).T

    xx = (x_design * np.abs(den) ** -2) @ x_design.T
    xdyx = (y * x_design * np.abs(den) ** -2) @ x_design.T
    xdycx = (np.conj(y) * x_design * np.abs(den) ** -2) @ x_design.T
    xdy2x = (np.abs(y) ** 2 * x_design * np.abs(den) ** -2) @ x_design.T

    up_right = np.linalg.inv(xx) @ xdyx
    bottom_left = np.linalg.inv(xdy2x) @ xdycx

    to_diag = np.zeros((2*(n+1), 2*(n+1)), dtype=complex)
    to_diag[:n+1, n+1:] = up_right
    to_diag[n+1:, :n+1] = bottom_left

    v, w = np.linalg.eig(to_diag)

    solution = w[:, np.argmin(np.abs(1 - v))]

    num = x_design.T @ solution[:n+1]
    den = x_design.T @ solution[n+1:]

    return num, den, solution

def convergence_criteria(array, convergance_precision, min_converged_passes):
    array_convergence = np.diff(array)/array[:-1]

    if np.prod(np.abs(array_convergence[-min_converged_passes:]) < convergance_precision):
        return True
    return False

def iterative_solver(x, y, n, min_rec_depth=10, max_rec_depth=100, convergance_precision=0.01, min_converged_passes=5):
    residuals = []

    center = 0.5*(np.max(x) + np.min(x))
    span = np.max(np.abs(x-center))
    x_norm = (x-center)/span
    
    num, den = fit_rational_fraction_pole_basis(x_norm, y, n=n)
    num, den, solution = fit_rational_fraction_polynomial_basis(x_norm, y, n=n, den=den)

    for _ in range(max(min_rec_depth, min_converged_passes)):
        num, den = fit_rational_fraction_pole_basis(x_norm, y, n=n, den=den)
        num, den, solution = fit_rational_fraction_polynomial_basis(x_norm, y, n=n, den=den)

        residuals.append(np.sum(np.abs(y-num/den)))

    if convergence_criteria(np.array(residuals), convergance_precision, min_converged_passes):
        return num, den, solution, residuals
    
    for _ in range(max(max_rec_depth - min_rec_depth, 0)):
        num, den = fit_rational_fraction_pole_basis(x_norm, y, n=n, den=den)
        num, den, solution = fit_rational_fraction_polynomial_basis(x_norm, y, n=n, den=den)

        residuals.append(np.sum(np.abs(y-num/den)))

        if convergence_criteria(np.array(residuals), convergance_precision, min_converged_passes):
            return num, den, solution, residuals
    
    return num, den, solution, residuals

def convert_solution(x, solution):
    n = (len(solution) // 2) - 1
    scalled_solution = np.copy(solution)
    
    center = 0.5*(np.max(x) + np.min(x))
    span = np.max(np.abs(x-center))

    scalled_solution[:n+1] /= span**np.arange(n+1)
    scalled_solution[n+1:] /= span**np.arange(n+1)

    translated_solution = np.zeros_like(solution)

    for i in range(n+1):
        num_i = 0
        den_i = 0

        for j in range(i, n + 1):
            num_i += scalled_solution[j] * comb(j, i) * (-center)**(j - i)
            den_i += scalled_solution[n+1+j] * comb(j, i) * (-center)**(j - i)
        
        translated_solution[i] = num_i
        translated_solution[n+1+i] = den_i
    
    return translated_solution


def get_rationnal_fit(x, y, n, return_true_solution=True, return_residuals=False, *args, **kwargs):
    num, den, solution, residuals = iterative_solver(x, y, n, *args, **kwargs)
    if return_true_solution:
        solution = convert_solution(x, solution)

    if return_residuals:
        return solution, num/den, residuals
    else:
        return solution, num/den

def rationnal_function(x, coefs):
    n = (len(coefs) // 2) - 1

    n_mesh, x_mesh = np.meshgrid(np.arange(n+1), x)
    x_design = (x_mesh**n_mesh).T

    num = x_design.T @ coefs[:n+1]
    den = x_design.T @ coefs[n+1:]

    return num/den

def reflection_purcell(freq, f_a_0, f_b_0, kappa_a, kappa_b, g):
    delta_a = freq - f_a_0
    delta_b = freq - f_b_0

    num = (1j * delta_b - 0.5 * kappa_b) * (1j * delta_a + 0.5 * kappa_a) + g ** 2
    den = (1j * delta_b + 0.5 * kappa_b) * (1j * delta_a + 0.5 * kappa_a) + g ** 2

    return num / den

if __name__ == "__main__":

    noise = 0.1
    n = 2
    x = np.linspace(-5, 10, 1001)
    coefs = np.random.rand(2*(n+1)) + 1j*np.random.rand(2*(n+1))
    y = rationnal_function(x, coefs) + noise*np.random.normal(0, 1, (x.size)) + noise*1j*np.random.normal(0, 1, (x.size))

    # y = reflection_purcell(x, -2, -1, 0.1, 3, 0.5) + noise*np.random.normal(0, 1, (x.size)) + noise*1j*np.random.normal(0, 1, (x.size))

    def fit_func(x, *args):
        n = len(args)//2
        coefs = np.array(args[:n]) + 1j*np.array(args[n:])
        return rationnal_function(x, coefs)
    
    solution, fit = get_rationnal_fit(x, y, n=n)
    popt, pcov = complex_fit(fit_func, x, y, p0=np.array([*solution.real, *solution.imag], dtype=float))
    better_solution = np.array(popt[:len(popt)//2]) + 1j*np.array(popt[len(popt)//2:])

    print(popt)

    plt.figure()
    plt.plot(np.abs(coefs/coefs[0]), 'oC0')
    plt.scatter(np.arange(2*(n+1)), np.abs(solution/solution[0]), facecolor='none', edgecolors='C1', zorder=100)
    plt.scatter(np.arange(2*(n+1)), np.abs(better_solution/better_solution[0]), facecolor='none', edgecolors='C2', zorder=100)

    plt.figure()
    plt.plot(x, np.angle(y))
    plt.plot(x, np.angle(fit))
    plt.plot(x, np.angle(fit_func(x, *popt)))
    plt.show()
