Python ML Minimal
=================

There are some programming situations where one
might not be allowed to use Numpy/Scipy and is
required to perform some simple regression or
classification tasks; programming competitions
are one example.

The code here has no dependencies that are not
in the Python 2.x core, so they may be used in
these situations with ease. Do note that not
using Numpy/Scipy optimization libraries results
in a decrease in both efficiency and accuracy;
do not use this code unless your hands are tied.

## Contents

   1. linear-regression-one-var: Univariate
      linear regression with feature scaling.

   2. linear-regression-mult-var: Multivariate
      linear regression with feature scaling.

   3. logistic-regression: Logistic regression
      with feature scaling.

## Credits

   * All the algorithms here have been derived from
     Octave code I wrote for the [ML-Class](http://www.ml-class.org)
     programming assignments.

   * Very useful insights and workarounds for issues
     with machine learning in minimal Python have been
     lifted from [OT's blog](http://nestedinfiniteloops.wordpress.com/).
