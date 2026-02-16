import numpy as np

def _default_initial_poles(s: np.ndarray, n_poles: int, alpha: float = 0.01) -> np.ndarray:
    """
    Heuristique type papier: p = (-alpha + j) * omega, omega linéairement espacé,
    puis conjugaison pour obtenir n_poles pôles (n_poles pair recommandé).
    """
    s = np.asarray(s, dtype=np.complex128).ravel()
    if n_poles < 1:
        raise ValueError("n_poles must be >= 1")
    if n_poles == 1:
        # un pôle proche de la 'bande' imaginaire
        w0 = np.median(np.abs(np.imag(s))) if np.any(np.imag(s)) else 1.0
        return np.array([-alpha * w0 + 1j * w0], dtype=np.complex128)

    # On essaie d'extraire une bande en imaginaire (cas fréquent: s = j*omega)
    w = np.abs(np.imag(s))
    wmax = float(np.max(w)) if np.any(w) else float(np.max(np.abs(s)))
    wmin = float(np.min(w[w > 0])) if np.any(w > 0) else (wmax / 100 if wmax > 0 else 1.0)
    wmin = max(wmin, 1e-6)

    # on fabrique n_poles/2 pôles complexes conjugués
    n_pairs = n_poles // 2
    omegas = np.linspace(wmin, wmax, n_pairs)

    poles_pos = (-alpha * omegas) + 1j * omegas
    poles = np.concatenate([poles_pos, np.conjugate(poles_pos)])

    # si n_poles impair, on ajoute un pôle réel négatif
    if n_poles % 2 == 1:
        poles = np.concatenate([poles, np.array([-alpha * wmax], dtype=np.complex128)])
    return poles[:n_poles].astype(np.complex128)


def _solve_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Moindres carrés robustes (SVD via lstsq)."""
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def _update_poles(poles: np.ndarray, w_coeffs: np.ndarray) -> np.ndarray:
    """
    Mise à jour des pôles via l’équation type (18) du papier:
      new_poles = eig( A - b * w^T )
    avec A = diag(poles), b = ones(N,1), w = [w1..wN]^T
    """
    N = len(poles)
    A = np.diag(poles)
    b = np.ones((N, 1), dtype=np.complex128)
    w = w_coeffs.reshape(1, N)  # (1,N)
    M = A - b @ w               # (N,N)
    return np.linalg.eigvals(M)


def _enforce_stability_lhp(poles: np.ndarray) -> np.ndarray:
    """Option simple: si Re(p)>0, on réfléchit dans le demi-plan gauche."""
    p = poles.copy()
    mask = np.real(p) > 0
    p[mask] = -np.real(p[mask]) + 1j * np.imag(p[mask])
    return p

def vector_fit_siso(
    s_k,
    y_k,
    n_poles: int,
    n_iter: int,
    init_poles=None,
    alpha: float = 0.01,
    enforce_stable: bool = False,
    final_refit: bool = True,
):
    """
    Vector Fitting SISO (une seule fonction à fitter), version complexe.

    Entrées:
      - s_k: (K,) points complexes où la fonction est échantillonnée
      - y_k: (K,) valeurs complexes correspondantes
      - n_poles: N
      - n_iter: nombre d'itérations VF (relocalisation des pôles)
      - init_poles: pôles initiaux (sinon heuristique)
      - alpha: amortissement des pôles initiaux
      - enforce_stable: force Re(p)<0 à chaque itération (option simple)
      - final_refit: refit final des résidus avec pôles figés (recommandé)

    Sorties:
      dict avec:
        poles, residues, r0, (num, den)  # (num,den) seulement si demandé via pole_residue_to_rational
    """
    s = np.asarray(s_k, dtype=np.complex128).ravel()
    y = np.asarray(y_k, dtype=np.complex128).ravel()
    if s.shape != y.shape:
        raise ValueError("s_k and y_k must have the same shape")
    K = len(s)
    if K < (n_poles + 1):
        raise ValueError("Not enough samples for the requested number of poles (need K >= n_poles+1 ideally).")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")

    poles = np.asarray(init_poles, dtype=np.complex128).ravel() if init_poles is not None else _default_initial_poles(s, n_poles, alpha)
    if len(poles) != n_poles:
        raise ValueError("init_poles must have length n_poles")

    r0 = 0.0 + 0.0j
    residues = np.zeros(n_poles, dtype=np.complex128)

    for _ in range(n_iter):
        # Matrice Phi0 = [1, 1/(s-p1), ..., 1/(s-pN)]
        Phi = np.empty((K, n_poles + 1), dtype=np.complex128)
        Phi[:, 0] = 1.0
        Phi[:, 1:] = 1.0 / (s[:, None] - poles[None, :])

        # Système (forme SISO de (21)):
        # [Phi  ,  -(diag(y) * Phi[:,1:])] * [r0,r1..rN, w1..wN]^T = y
        # (w0 fixé à 1 dans la version "classique")
        A_left = Phi
        A_right = -(y[:, None] * Phi[:, 1:])  # diag(y) @ Phi1
        A = np.hstack([A_left, A_right])
        x = _solve_ls(A, y)

        r0 = x[0]
        residues = x[1:n_poles + 1]
        w_coeffs = x[n_poles + 1:]  # (N,)

        poles = _update_poles(poles, w_coeffs)

        if enforce_stable:
            poles = _enforce_stability_lhp(poles)

    # Refit final: résoudre Phi * c_H = y avec pôles figés (équation type (29))
    if final_refit:
        Phi = np.empty((K, n_poles + 1), dtype=np.complex128)
        Phi[:, 0] = 1.0
        Phi[:, 1:] = 1.0 / (s[:, None] - poles[None, :])
        cH = _solve_ls(Phi, y)
        r0 = cH[0]
        residues = cH[1:]
    
    fit_function = np.ones_like(y_k).astype(complex) * r0
    for residue, pole in zip(residues, poles):
        fit_function = fit_function + residue/(s_k - pole)
    
    least_square_error = np.sum(np.abs(fit_function-y_k)**2)/s_k.size

    return {
        "poles": poles,
        "residues": residues,
        "r0": r0,
        "fit_function": fit_function,
        "least_square_error": least_square_error
    }
