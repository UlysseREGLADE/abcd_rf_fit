import numpy as np
import sympy as sy
from scipy.special import comb, factorial


def get_denominator_coeffs(degree: int, x: np.array, f: np.array) -> np.array:
    """Assuming f(x) = P(x) / Q(x) with P and Q polynoms of degree 'degree',
    and y_n = f(x_n) for n in [0, len(y)]"""
    dx = x[1:] - x[:-1]
    assert dx.max() - dx.min() < 1e-10, f"Only support evenly spaced points"
    assert degree > 0, "Cannot infer parameters for zero degree"
    assert x.shape == f.shape, "You must give all pairs or (x, f(x))"
    assert (
        len(x) > 2 * degree
    ), f"Need at {2 * degree} least degree points to infer the {degree} coefficients"

    dx = dx[0]
    y = sy.symbols("y")
    coefficients = sy.symbols(f"a:{degree+1}")
    f_derivatives = sy.symbols(f"f:{degree+1}")
    Q = sum([c * y**n for n, c in enumerate(coefficients)])

    f_Q_nth_derivative = 0 * y  # Trick so the IDE to infer it as Expr instead of int
    # Compute in sympy
    # \sum_{k=0}^n {n \choose k} f^{(n-k)} Q^{(k)}
    for k in range(degree + 1):
        term = comb(degree, k)
        term *= f_derivatives[degree - k]
        term *= Q.diff(y, k)
        f_Q_nth_derivative += term

    # Evaluation of all derivatives of f in each point of x
    f_numerical_derivatives = np.zeros((len(f), degree + 1))
    f_numerical_derivatives[:, 0] = f
    for n in range(0, degree):
        f_numerical_derivatives[:, n + 1] = np.gradient(
            f_numerical_derivatives[:, n], dx
        )

    # we drop the points on the side since their gradient will have errors
    x = x[degree:-degree]
    f_numerical_derivatives = f_numerical_derivatives[degree:-degree]

    # Build the system as a matrix
    system = []
    for i in range(f_numerical_derivatives.shape[0]):
        eq = f_Q_nth_derivative.subs(y, x[i])
        eq = eq.subs(zip(f_derivatives, f_numerical_derivatives[i]))
        eq = sy.Poly(eq, coefficients)
        eq = eq.coeffs()
        eq = list(map(float, eq))
        system.append(eq)
    system = np.array(system)

    # Compute the minimizing cost
    y = factorial(degree) * np.ones(system.shape[0])
    (result, _residuals, _rank, _singular_values) = np.linalg.lstsq(
        system, y, rcond=None
    )

    return result
