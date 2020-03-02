# Level-Set Method

A level set of a function $f:\R^n\rightarrow\R$ is defined as
$$ L_c(f) = \vset{x\in\R^n}{f(x) = c} $$

In the case of $n=2$ the level set corresponds to an implicit
cartesian representation of a curve, also called *isoline*.

The Level-Set Method (LSM) defines the curve $\Gamma$ as
the zero level set of a function $\phi:\R^2\rightarrow\R$
called the **level function**.
$$ \Gamma = \vset{x\in\R^2}{\phi(x) = 0}$$
The essence of the method is evolving the curve $\Gamma$
implicitly through its level function $\phi$.

LSM allows manipulating hypersurfaces in different contexts
without having their parametric representation.
It can be very useful in tracking changing topologies
and in contour detection.

In this part we will consider images defined over a domain
$[0,1]^2$ discretized in an $n\times n$ grid.
As a preliminary example, let's consider the level set represenation
of a circle of radius $r$ centered in $c\in\R^2$.

$$\phi_1(x) = \norm{x-c}_2 - r$$
where $\norm{\cdot}_2$ is the well-known euclidean norm $L^2$.

The level-set function divides the domain into three sets:

- $\Gamma=\vset{x\in\R^2}{\phi_1(x)=0} =\vset{x\in\R^2}{\norm{x-c}_2=r} 
=$ points lying on the circle.
- $L_{-}=\vset{x\in\R^2}{\phi_1(x)<0} =\vset{x\in\R^2}{\norm{x-c}_1<r}
=$ points lying inside the circle.
- $L_{+}=\vset{x\in\R^2}{\phi_1(x)<0} =\vset{x\in\R^2}{\norm{x-c}_1<r}
=$ points lying outside the circle.

Similarly we can define $\phi_2$ the level-set function
of a square centered in $c$ whose side is $2r$.
$$\phi_2(x) = \norm{x-c}_\infty - r$$
where $\norm{\cdot}_\infty$ is the maximum norm $L^\infty$.

Let's plot our level functions as a greyscale color mappings.


```{python}
# initialize grid
n = 200
x = y = np.arange(1, n + 1)
X, Y = np.meshgrid(x, y, sparse=True, indexing='xy')

# calculate phi1
r = n / 3
c = np.array([r,r]) + 10
phi1 = np.sqrt((X-c[0])**2 + (Y-c[1])**2) - r

# calculate phi2
c = n - c
phi2 = np.maximum(abs(X-c[0]), abs(Y-c[1])) - r

# plot mappings
from util import plot_levelset
_ = plt.figure(figsize = (10,5))
_ = plt.subplot(121)
plot_levelset(phi1, colorbar=True)
_ = plt.subplot(122)
plot_levelset(phi2, colorbar=True)
plt.show()
```

The level-set representation allows us to compute easily
the intersection and the union of two regions.

The level-set function of the union of domains is defined
as the minimum of the domains' functions.
$$\phi_{\cup} = \min(\phi_1, \phi_2)$$

Similarly, the level-set function of the intersection of domains
is defined by the maximum.
$$\phi_{\cap} = \max(\phi_1, \phi_2)$$

```{python}
_ = plt.figure(figsize = (10,5))

_ = plt.subplot(121)
_ = plt.title("Union")
plot_levelset(np.minimum(phi1, phi2), colorbar=True)

_ = plt.subplot(122)
_ = plt.title("Intersection")
plot_levelset(np.maximum(phi1, phi2), colorbar=True)

plt.show()
```

## Curve evolution

As mentioned before, LSM manipulates curves implicitly
through its level-set function.
Similarly to parametric curve evolution, LSM curve evolution
is a series of level-set functions $s\mapsto\phi_s$.

The evolution speed was defined for parametric curves
in function of the curve function, its normal and
its intrinsic curvature.
These terms can be redefined for curves expressed
as zero-level sets as follows:

- **Normal:** $n(x) = \frac{\nabla\phi(x)}{\norm{\nabla\phi(x)}}$
- **Curvature:** $\kappa(x) = \mathrm{div}{\pp{\frac{\nabla\phi}{\norm{\nabla\phi}}}}(x)$

The evolution PDE of the level-set function (called the *Level-Set Equation*) is

$$ \frac{\diff}{\ds} \phi_s = \beta(\phi_s(x),n_s(x),\kappa_s(x))
\cdot \norm{\nabla\phi_s} $$

## Mean curvature motion

As for parametric curves, mean curvature motion is based
on minimizing the normal energy flow which is characterized
by the curve's intrinsic curvature.

We have seen that the speed function is defined as
$\beta(\phi,n,\kappa) = \kappa$.

The level-set equation becomes
$$\frac{\diff}{\ds} \phi_s = \mathrm{div}\pp{\frac{\nabla\phi_s}{\norm{\nabla\phi_s}}} \cdot \norm{\nabla\phi_s}$$

Let's perform this method on the union of the
previous curves.

```{python}
from nt_toolbox.grad import *
from nt_toolbox.div import *

dt = 0.5                    # time step
Tmax = 200                  # stop time
niter = round(Tmax / dt)    # number of iterations
nplot = 4                   # number of plots
plot_interval = round(niter / nplot)

phi0 = np.minimum(phi1, phi2)   # union curve
phi = np.copy(phi0)         # initial curve
plot_iter = plot_interval   # plot iterator
subplot = 1                 # subplot counter


_ = plt.figure(figsize=(10,10))
eps = np.finfo(float).eps

for i in range(niter + 1):
    # g0 = grad(phi)
    g0 = grad(phi, order=2)
    # d = |grad(phi)|
    d = np.maximum(np.sqrt(np.sum(g0**2, 2)), eps)
    # g = grad(phi)/|grad(phi)|
    g = g0 / np.repeat(d[:,:,np.newaxis], 2, 2)
    # K = div(g)
    K = div(g[:,:,0], g[:,:,1], order=2)
    # calculate phi step
    G = K * d
    phi += dt * G

    # plot levelset
    if i == plot_iter and subplot <= 4:
        ax = _ = plt.subplot(2, 2, subplot)
        _ = ax.set_title(r'$\varphi_{' + str(i) + '}$')
        plot_levelset(phi)
        subplot += 1
        plot_iter += plot_interval

plt.show()
```

## Level-set redistancing

While the essential property of the level set function $\phi$
is the location of its zero isoline, in practice many applications
additionally require that it be a signed distance function
$\norm{\nabla\phi} = 1$ which ensures that the zero crossing
is sufficiently sharp.

Nevertheless, this property is not generally preserved by
the level-set evolution.
**Redistancing** is the process of recovering this property
without modifying the location of the zero isoline.

Redistancing essentially is computing a signed distance function
$\tilde\phi$ from an arbitrary non-signed distance function $\phi$
while preserving the zero isoline.
Mathematically, this process obeys the *eikonal equation*. [@hopf_lax]

$$\begin{cases}
\norm{\nabla \tilde\phi} = 1\\
\sign(\tilde\phi) = \sign(\phi)
\end{cases}$$

We can set $\phi$ initially to $\phi_0^3$ as $x\mapsto x^3$ preserves
the sign (consequently the isoline), we obtain the signed disntance
function $\tilde\phi$ from $\phi$ (which is a non-signed distance) by
performing redistancing with the help of the methods of `nt_toolbox`.

```{python}
phi = phi0**3

from nt_toolbox.perform_redistancing import *
phi1 = perform_redistancing(phi0)

_ = plt.figure(figsize=(10,5))

_ = plt.subplot(1,2,1)
_ = plt.title("Before redistancing")
plot_levelset(phi, colorbar=True)

_ = plt.subplot(1,2,2)
_ = plt.title("After redistancing")
plot_levelset(phi1, colorbar=True)

plt.show()
```

## Geodesic motion

Unlike geodesic motion of parametric curves, the evolution
of the zero level-set is computed along *local minima* of the weighted
geodesic distance attracting the curve toward the features
of the background image.

Given a background image $f_0$ to segment, we compute
the geodesic weight metric $W$.
The geodesic weight should be a decreasing function
of the blurred gradient magnitude.

$$ W(x) = \psi\underbrace{\pp{d_0\star h_a}}_{d}
\qtext{where} d_0 = \norm{\nabla f_0}$$
given the blurring kernel $h_a$ of size $a>0$.

Let's calculate first the blurred gradient magnitude.

```{python}
# load image of size n×n
f0 = rescale(load_image("nt_toolbox/data/cortex.bmp", n))

# compute the magnitude of the gradient
g = grad(f0, order=2)
d0 = np.sqrt(np.sum(g**2, 2))

# perform blurring
from nt_toolbox.perform_blurring import *
a = 5
d = perform_blurring(d0, np.asarray([a]), bound="per")
```

The decreasing function $\psi$ can be defined
as $\psi:s\mapsto \alpha + \frac{\beta}{\eps + s}$.
Nevertheless, the function `rescale` from the library `nt_toolbox`
can adjust the overall values of $W$ as needed without having
to adjust the parameters $\alpha$ and $\beta$.

```{python}
# calculate weight
epsilon = 1e-1
W = 1./(epsilon + d)
W = rescale(-d, 0.1, 1)

# display
_ = plt.figure(figsize=(10,5))
imageplot(f0, "Image to segment", [1,2,1])
imageplot(W, "Weight", [1,2,2])
plt.show()
```

```{python}
# initialize centered square contour
r = n / 3
c = np.asarray([n,n]) / 2
phi0 = np.maximum(np.abs(X-c[0]), np.abs(Y-c[1])) - r

# display
_ = plt.figure(figsize=(5,5))
plot_levelset(phi0, img=f0)
plt.show()
```

We remind the weighted length defined for a parametric curve
$$L(\ga) = \int_0^1 W(\ga(t)) \norm{\ga'(t)} \dt$$
and the speed term of the evolution equation
$$\beta(x,n,\kappa)=W\cdot\kappa-\dotp{\nabla W}{n}$$

The evolution equation for the level-set function is defined as
$$\frac{\diff}{\ds} \phi_s = \mathrm{div}\pp{W\frac{\nabla\phi_s}{\norm{\nabla\phi_s}}} \cdot \norm{\nabla\phi_s}$$

Let's apply the method on our initial square zero isoline.

```{python}
dt = 0.4                    # time step
Tmax = 1500                 # stop time
niter = round(Tmax / dt)    # number of iterations
nplot = 4                   # number of plots
plot_interval = round(niter / nplot)
n_redistancing = 30

phi = np.copy(phi0)         # initial curve
gW = grad(W, order=2)       # grad(W)
plot_iter = plot_interval   # plot iterator
subplot = 1                 # subplot counter

_ = plt.figure(figsize=(10,10))

for i in range(niter + 1):
    # g0 = grad(phi)
    g0 = grad(phi, order=2)
    # d = |grad(phi)|
    d = np.maximum(np.sqrt(np.sum(g0**2, 2)), eps)
    # g = grad(phi)/|grad(phi)|
    g = g0 / np.repeat(d[:,:,np.newaxis], 2, 2)
    # K = div(g)
    K = div(g[:,:,0], g[:,:,1], order=2)
    # calculate phi step
    G = W * d * K + np.sum(gW * g0, 2)
    phi += dt * G

    # perform redistancing
    if i % n_redistancing == 0:
        phi = perform_redistancing(phi)

    # plot levelset
    if i in [plot_iter, niter]:
        ax = _ = plt.subplot(2, 2, subplot)
        _ = ax.set_title(r'$\varphi_{' + str(i) + '}$')
        plot_levelset(phi, img=f0)
        subplot += 1
        plot_iter += plot_interval

plt.show()
```

## Region-based Chan-Vese segmentation

Chan-Vese active contours corresponds to a region-based energy
that looks for a piecewise constant approximation of the image.

The energy to be minimized is
$$ \min_{\phi}\pp{
    L(\phi) +
    \lambda \int_{\phi(x)>0} \abs{f_0(x)-c_1}^2 d x +
    \lambda \int_{\phi(x)<0} \abs{f_0(x)-c_2}^2 d x
}$$
where $L$ is the length of the zero level set of $\phi$.
Note that here $(c_1,c_2) \in \R^2$ are assumed to be known.

Let's initialize our regions.

```{python}
_ = plt.figure(figsize=(10,5))
k = 4               # number of circles
n_k = n / k         # n per region
r = 0.3 * n_k       # circles' radius

# initialize phi to infinity
phi0 = np.zeros([n,n]) + np.float("inf")

# calculate negative regions
for i in range(1, k+1):
    for j in range(1, k+1):
        c = n_k * (np.asarray([i,j]) - 0.5)
        phi0 = np.minimum(phi0, np.sqrt(abs(X-c[0])**2 + abs(Y-c[1])**2) - r)

# display regions
_ = plt.subplot(1,2,1)
plot_levelset(phi0)
_ = plt.subplot(1,2,2)
plot_levelset(phi0, img=f0)
plt.show()
```

The minimizing flow for the Chan-Vese energy
can be expressed as
$$ \frac{\diff}{\dt}\phi_t = -G(\phi_t)$$
where
$$ G(\phi) = -\norm{\nabla \phi}
    \text{div}\pp{\frac{\nabla \phi}{\norm{\nabla \phi}}}
    + \lambda (f_0-c_1)^2 - \lambda (f_0-c_2)^2$$

Now that we have our evolution equation well-defined
we can apply our gradient descent method.

```{python}
dt = 0.5                    # time step
Tmax = 100                  # stop time
niter = round(Tmax / dt)    # number of iterations
nplot = 4                   # number of plots
plot_interval = round(niter / nplot)
n_redistancing = 30

lmbda = 2
c1, c2 = (0.7, 0)

phi = np.copy(phi0)         # initial curve
plot_iter = plot_interval   # plot iterator
subplot = 1                 # subplot counter

_ = plt.figure(figsize=(10,10))

for i in range(niter + 1):
    # g0 = grad(phi)
    g0 = grad(phi, order=2)
    # d = |grad(phi)|
    d = np.maximum(np.sqrt(np.sum(g0**2, 2)), eps)
    # g = grad(phi)/|grad(phi)|
    g = g0 / np.repeat(d[:,:,np.newaxis], 2, 2)
    # K = div(g)
    K = div(g[:,:,0], g[:,:,1], order=2)
    # calculate energy term for non-zero phi
    energy = (f0-c1)**2 - (f0-c2)**2
    # calculate phi step
    G = d * K - lmbda * energy
    phi += dt * G

    # perform redistancing
    if i % n_redistancing == 0 and i > 0:
        phi = perform_redistancing(phi)

    # plot levelset
    if i in [plot_iter, niter]:
        ax = _ = plt.subplot(2, 2, subplot)
        _ = ax.set_title(r'$\varphi_{' + str(i) + '}$')
        plot_levelset(phi, img=f0)
        subplot += 1
        plot_iter += plot_interval

plt.show()
```