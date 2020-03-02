# Introduction

Recognising objects and identifying shapes in images is usually an easy
task for human, it is however difficult to automate.
The field of **computer vision** is concerned with automating such processes.
It aims to extract information from images (or video sequences of images)
in order to achieve what a human visual system can. [@wiki:computer_vision]

**Active contour models** (also called **snakes**) is a class of algorithms
for finding boundaries of shapes.
These methods formulate the problem as an optimisation process while
attempting to balance between matching to the image and ensuring
the result is smooth. [@snakes_icbe]

In the scope of this project, we will explore a few active contour models
with the help of *The Numerical Tours of Signal Processing* [@gpeyre].

## Snakes method

A snake is a smooth curve, similar to a spline.
The concept of a snakes method is to find smooth curves that match the image
features by iteratively minimizing the *energy* function of the snakes. [@wiki:snakes]

![Illustration of the snakes model](img/snakes.png)

The energy of the snakes is a combination of internal and external energy [@snakes_icbe]

- **Internal Energy:** a metric that measures the curve's smoothness or regularity.
- **External Energy:** a metric for measuring the data fidelity.

## Classification and representation

Curves can be divided into two types: open curves and closed curves.

- An *open curve* has two distinct ends and does not form a loop.
- A *closed curve* is a curve with no endpoints and which completely encloses an area. [@closed_curve]

![](img/curves.png)

A curve has two different representations: **parametric** or **cartesian**.

In the parametric form, the points of the curve are expressed as a function
of a real variable, conventionnally denoted $t$ representing *time*.

For instance, the parametric representation of circle in $\R^2$ is given by
$$ \gamma: t\mapsto \pmat{x_0+r\cos t\\y_0+r\sin t} $$
where the points of the curve are defined as $\Gamma = \Im(\gamma) = \sset{\gamma(t), t\in\R}$.

The cartesian representation is an equation (or a set of equations)
that describes the relations between the coordinates of the points of the curve.
Such representation can be explicit $y=f(x)$ or implicit $f(x,y)=0$.

In the case of a circle, we can express it implicitly by
$$ \Gamma = \vset{(x,y)\in\R^2}{(x-x_0)^2 + (y-y_0)^2 = r} $$

or explicitly
$$ \Gamma = \vset{(x,y)\in\R^2}{y=y_0\pm\sqrt{r-(x-x_0)^2}} $$

Generally speaking, the parametric representation can be more expressive than
cartesian equations.
It is also worth noting that tracking a curve's behaviour in a small neighborhood
of a point on the curve is much simpler as the derivative of the parametric
function $\gamma'(t)$ is easy to calculate and study.

In the following sections we will study active contours with respect to
their curve representation, as described by G. Peyré [-@gpeyre].

- Parametric active contours
- Implicit active contours