# Conclusion

Let's compare the execution time of each method
for detecting the contour of the medical image.

```{python}
results = "Time elapsed per method (in seconds):\n"
results += f"Parametric edge-based:\t{param_time}\n"
results += f"Implicit edge-based:\t{lsm_time}\n"
results += f"Implicit region-based:\t{cv_time}\n"

print(results)
```

We have seen two classes of active contour methods,
the parametric representation allows simple calculations
of evolution functions. However, the implicit representation
manipulates 2D functions so the calculations are significantly
more complex as they involve gradients and divergences of functions
as well as they require more storage.
Nevertheless, the implicit representations allows tracking
topological changes and can therefore be preferred in analysing
sequences of images (videos).

Thanks to *Numerical Tours* [@gpeyre] we were able to explore
a class of contour detection methods that can be applied
to different problems.
Peyré's tours have allowed us to understand and implement
these methods on medical images.

This projet has allowed me to explore the domain of image processing
for the first time, and have peaked my interest to seek further
understanding of current approaches of contour detection,
as well as other problems of image processing.

\pagebreak
# Références {-}