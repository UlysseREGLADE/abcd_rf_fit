# Optimal method for resonator scattering parameter fit

## 0) Why spending time on this problem ?

Fitting resonators is really the bread and butter of the circuit QED engineer. Hence, it is absolutely crucial to have fast and reliable routines to perform this task.

To the best of my knowledge, there is currently no satisfying method in the literature to efficiently perform the fit of the scattering parameter of a resonator, apart from the one described in [this paper](https://arxiv.org/pdf/1410.3365.pdf) from 2014. This method is far from being perfect and has two main flaws:

Frist, it simply tries to fit a circle to the data in the complex plane. Hence, this method completely ignores the "cinematic" encoded in the data when $\omega$ travels through the resonance. In particular, we will see that the gradient of the data encodes a lot of information.

Secondly, if the fit of the electrical delay fails, then the data effectively lies on a circle of radius $\alpha$ centered around $0$ in the complex plane. As a consequence, the fit procedure will succeed, but the fitted parameters will be totally wrong.

In this note, we present an analytical method to perform this task.

## I) Derivation scattering parameter formula

We use the convention described in the [Gardiner](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.31.3761) for the input/output relation:

$$
\begin{align}
a^i_{out}-a^i_{in} &= \sqrt{\kappa_c}a \\
\frac{\partial a}{\partial t} &= -\frac{i}{\hbar}\frac{\partial H}{\partial a^\dagger} - \frac{\kappa_i}{2}a - \sum_i \left( \frac{\kappa^i_c}{2}a + \sqrt{\kappa^i_c}a^i_{in} \right)
\end{align}
$$

In the rotating frame at $\omega$, the Hamiltonian we consider the one of a bar linear resonator:

$$
H/\hbar = (\omega-\omega_0)a^\dagger a
$$

By definition, the scattering parameter is defined by:

$$
S_{ij}(\omega) = \frac{a^i_{out}}{a^j_{in}}
$$

### 1. Transmission

We consider:

$$
S_{21}(\omega) = S_T(\omega) = \frac{a^2_{out}}{a^1_{in}}
$$

We find:

$$
S_T(\omega) = \frac{2\sqrt{\kappa^1_c\kappa^2_c}}{2i(\omega-\omega_0) + \kappa_i+\kappa^1_c+\kappa^2_c}
$$

The only fittable parameters in this formula are $\omega_0$ and $\kappa = \kappa_i+\kappa^1_c+\kappa^2_c$. Since the numerator is constant, it will be entirely eaten by the amplification coefficient, hence we write:

$$
S_T(\omega) = \frac{1}{2i(\omega-\omega_0) + \kappa}
$$

### 2. Reflection

$$
S_{11}(\omega) = S_R(\omega) = \frac{a^1_{out}}{a^1_{in}}
$$

We find:

$$
S_R(\omega) = \frac{i(\omega-\omega_0) + (\kappa_i-\kappa_c)/2}{i(\omega-\omega_0) + (\kappa_i+\kappa_c)/2}
$$

In this case, both $\omega_0$, $\kappa_i$ and $\kappa_c$ are fittable, which makes a reflection measurement much more valuable than a transmission one. Both the transmission and reflection formula are derived in the thesis of Philippe Campagne-Ibarcq.

### 3. Hanger

To get the hanger scattering parameter, one should modify the input/output relation as follow:

$$
a^i_{out}-a^i_{in} = \frac{\sqrt{\kappa^i_c}}{2}a
$$

In this context, we can now compute:

$$
S_{11}(\omega) = S_H(\omega) = \frac{a^1_{out}}{a^1_{in}}
$$

We find:

$$
S_H(\omega) = \frac{i(\omega-\omega_0) + \kappa_i/2}{i(\omega-\omega_0) + (\kappa_i+\kappa_c)/2}
$$

Again, $\omega_0$, $\kappa_i$ and $\kappa_c$ are fittable.

### 4. Hanger with impedance mismatch

In [this paper](https://arxiv.org/pdf/1410.3365.pdf) from 2014 is described how to take into account the effect of an impedance mismatch in the context of the hanger geometry. In this case, $\kappa_c$ can be view as a complex number, in this situation, we write:

$$
\kappa_c = |\kappa_c|e^{i\phi_0}
$$

In this case, only the real part of the complex coupling rates contribute to the total loss-rate of the cavity $\kappa$ .

$$
\kappa = \kappa_i + \Re(\kappa_c)
$$

One should introduce the parameter $\phi_0$ in the scattering parameter of the hanger as follow:

$$
S_{HM}(\omega) = \frac{2i(\omega-\omega_0) + \kappa - \Re(\kappa_c)(1+i\tan(\phi_0))}{2i(\omega-\omega_0) + \kappa}
$$

From a geometrical point of view in the complex plan, it allows the rotation of the circle described by $S_H(\omega)$ around $1$. Empirically, one can also see it as a modification of the input/output relation:

$$
a^i_{out}-a^i_{in} = \frac{\sqrt{|\kappa^i_c| e^{i\phi_0}}}{2}a
$$

In practise, adding this degree of freedom to the formula loosens the precision of the fit of $\kappa_i$ and $\kappa_c$. Hence, one should be cautious when allowing for this offset.

### 5. Reflection with impedance mismatch

Taking inspiration for what was done for the hanger, we allow the rotation around $1$ in the complex plan by writing the input/output relation as follow. THERE IS NO PHYSICAL INTUITION BEHIND THIS FORMULA AND IT IS WRITTEN PURELY BY ANALOGY:

$$
a^i_{out}-a^i_{in} = \sqrt{|\kappa^i_c|e^{i\phi_0}}a
$$

The modified scattering parameter for the reflection geometry is as follow:

$$
S_{RM}(\omega) = \frac{2i(\omega-\omega_0) + \kappa -  2\Re(\kappa_c)(1+i\tan(\phi_0))}{2i(\omega-\omega_0) + \kappa}
$$

Again, the precision of the fit of $\kappa_i$ and $\kappa_c$ are loosened, and this formula is to be taken with the same caveat than the previous one. abcd_rf_fit will return a warning when the value of $\phi_0$ in greater than 0.25 .

### Effect of electrical delay and amplification chain

In a real life scenario of circuit QED, the scattering parameter of a resonator is always dressed.

First, since these resonators are meant to operate at very low energy scales, one always uses an extensive attenuation chain to send signal in, and a powerful amplification chain to retrieve the outputted signal. This results in an arbitrary complex multiplicative prefactor $\alpha \in \mathbb{C}$.

Second, the resonator is always at a finite distance from the instrument used to measure it. As a consequence, the phase of the outputted signal will vary as a function of its frequency depending on this distance and the speed of light in the medium that carries the signal. This has the effect to multiply the scattering parameter by a factor $e^{2i\pi\lambda\omega}$, where $\lambda \in \mathbb{R}$ is a typical time called the electrical delay that encodes for cables length and the speed of light among them.

Hence, what is actually to be fitted in most cases is:

$$
\begin{align}
S(\omega) &= \alpha \times S_X(\omega) \times e^{2i\pi\lambda\omega} \\
X &= T, R, H, RM, HM
\end{align}
$$

## II) Fit of a rationnal function of degree one

Ingoring the electrical for now, one can observe that all the scattering parameters we described can be written in the form:

$$
S_{X}(\omega) = \frac{a+b\omega}{c+d\omega}, (a, b, c, d) \in \mathbb{C}
$$

We propose a efficent procedure to extract these coefficents from a noised signal.

### 1. Side note: least square regression of an n-degree polynomial

As described in the [lecture notes](https://www.di.ens.fr/~fbach/mlclass/lecture2.pdf) of Francis Bach, given a set of points $(x_i, y_i), i \in$ &lobrk;1, N&robrk;, one can write the empirical risk $R(w)$ associated with the least square regression of an $n$-degree polynomial with coefficients $w_0, ..., w_n$ as follow:

$$
R_{LS}(w) = \frac{1}{N} || Xw - y ||_2^2
$$

with:

$$
X =
\begin{bmatrix}
1, x_1, \ldots, x_1^n \\
\vdots \\
1, x_N, \ldots, x_N^n \\
\end{bmatrix},
y =
\begin{bmatrix}
y_1 \\
\vdots \\
y_N \\
\end{bmatrix}
w =
\begin{bmatrix}
w_0 \\
\vdots \\
w_n \\
\end{bmatrix}
$$

We can now easily solve for $\nabla_w R(w) = 0$ to find the global minimum of this convex optimization problem:

$$
w = (X^T X)^{-1} Xy
$$

Not only is this formulation super elegant, it allows for very fast compution. It is actually the one implemented in the Python libreary numpy in the function `polyfit`. 

We take inspiration of this approach in the next section to find an efficient fitting procedure in our case.

### 2. Least square regression of a degree one rationnal function

We are given a set $(\omega_i, s_i), i \in$ &lobrk;1, N&robrk;, where $\omega_i$ is a frequency and $s_i$ the signal measured by the instrument, for instance a vectorial network analyser (VNA).

We whould like the find the global minimum of the least square risk. In this context it reads:

$$
R_{LS}(a, b, c, d) = \frac{1}{N}\sum_i \left| s_i - \frac{a+b\omega_i}{c+d\omega_i}\right|^2
$$

We write:

$$
X =
\begin{bmatrix}
1, \omega_1 \\
\vdots \\
1, \omega_N \\
\end{bmatrix},
y =
\begin{bmatrix}
s_1 \\
\vdots \\
s_N \\
\end{bmatrix}
w =
\begin{bmatrix}
a \\
b \\
\end{bmatrix},
m =
\begin{bmatrix}
c \\
d \\
\end{bmatrix}
$$

We now introduce the following empirical risk:

$$
R(w, m) = || \mathcal{D}(y)Xm - Xw||_2^2
$$

Where $\mathcal{D}$ is a short hand for the diagonal matrix with the elements of $y$ on its diagonal:

$$
\mathcal{D}(y) = \begin{bmatrix}
y_1 & & \\
& \ddots & \\
& & y_N
\end{bmatrix}
$$

Note that this risk is not yet the one associated with the least square regression, however its zeros includes the one of the least square risk. Indeed, if $R_{LS}(w, m) = 0$ then we have $R(w, m) = 0$.

We now solve for $\nabla_w R(w, m) = \nabla_m R(w, m) = 0$ :

$$
\begin{align}
X^\dagger X w - X^\dagger \mathcal{D}(y) X m = 0 \\
X^\dagger \mathcal{D}(|y|^2) Xw - X^\dagger \mathcal{D}(y^*) X m = 0
\end{align}
$$

We write:

$$
\begin{align}
A &= (X^\dagger X)^{-1} X^\dagger \mathcal{D}(y) X \\
B &= (X^\dagger \mathcal{D}(|y|^2) X)^{-1} X^\dagger \mathcal{D}(y^*) X \\
C &= \begin{bmatrix}
0_2, A \\
B, 0_2
\end{bmatrix}
\end{align}
$$

Leading to:

$$
\begin{bmatrix}
w \\
m
\end{bmatrix} = C \begin{bmatrix}
w \\
m
\end{bmatrix}
$$

Diagonalizing this $4\times4$ matrix and looking for its eigen vector associated with the eigen value $1$ allows to efficiently solve for $a, b, c, d$.

This is very nice and efficient to compute, though in the case of noised signals the fit while eventually fail since this risk dose not coincide with $R_{LS}(w, m)$.

To solve this issue, one can observe that the $R(w, m)$ we introduce corresponds to $R_{LS}(w, m)$ where each data point has been weighted by $|c+d\omega|$ which has the bad taste of being maximal away from the resonance. Fortunately for us, it happens that the empirical gradient of the signal encodes for this quantity:

$$
\sqrt{|\nabla_\omega S_X(\omega)|} = \sqrt{\left|\frac{b}{d}\left(\frac{c}{d}-\frac{a}{b} \right)\right|}\frac{1}{|c+d\omega|} \propto \frac{1}{|c+d\omega|}
$$

You can now pick your prefered empirical estimator of the gradient $\tilde\nabla_X(y)$ such as the convolution with the derivative of a gaussian kernel for instance. In the limite where this estimation of the gradient is accurate, we have $R_{LS}(w, m) = R^*(w, m)$, where:

$$
R^*(w, m) = \frac{1}{N}||\mathcal{D}\left(\sqrt{|\tilde{\nabla}_X(y)|}\right) (\mathcal{D}(y)Xm - Xw)||_2^2
$$

The corrected values for the matrices $A$ and $B$ are:

$$
\begin{align}
A &= (X^\dagger \mathcal{D}(|\tilde{\nabla}_X(y)|) X)^{-1} X^\dagger \mathcal{D}(y|\tilde{\nabla}_X(y)|) X \\
B &= (X^\dagger \mathcal{D}(|\tilde{\nabla}_X(y)||y|^2) X)^{-1} X^\dagger \mathcal{D}(y^*|\tilde{\nabla}_X(y)|) X
\end{align}
$$

Performing the conversion from the coefficients $a, b, c, d$ to the relevent quantities depending on the resonator geometry is now only a matter of simple algebra.

### 3. Estimation of the electrical delay

Assuming a flat frequencial landscape, the best estimator of the electrical delay one can write is the following:

$$
\lambda_\text{flat} = \frac{1}{N} \sum_{i=0}^{N-1} \frac{\arg(s_{i+1}/s_i)}{\omega_{i+1} - \omega_{i}}
$$

Please observe that this estimator should be much more robust to noised signals than performing an unwraping of the signal phase such as the one implemented in the Python library Numpy in the function `unwrap`.

In the case of a simple resonator, the signal performs at wrost a full revolution around the origin of the complex plane. Since on modern hardware the diagonalisation of a four by four matrix such as $C$ is so fast, one can brute force all the values of lambda in the range:

$$
\lambda \in \left[ \lambda_\text{flat} - \frac{1.5}{\max_{i}(\omega_i)-\min_{i}(\omega_i)}, \lambda_\text{flat} + \frac{1.5}{\max_{i}(\omega_i)-\min_{i}(\omega_i)} \right]
$$


This simple method proves to be really efficient.

## Contributing

Before you commit, run:
```
black .
ruff check . --fix
```