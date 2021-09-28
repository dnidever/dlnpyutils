***************
Getting Started
***************

There are currently 11 main modules in |dlnpyutils|.  They are described in detail below.


utils
=====

The utils module has many of the general purpose functions.

 - mad: median absolute deviation of array
 - minmax: minimum and maximum of an array
 - stat: many useful statistics of an array
 - strlen: number of characters in a string array or list
 - strip: strip whitespace from string array or list
 - strjoin: combine string arrays or scalars
 - strsplit: split string arrays
 - pathjoin: join two pathname components
 - first_el: return the first element of an array or list
 - grep: grep on a string array
 - readlines: read a file into a string array
 - writelines: write a string array to a file
 - remove_indices: remove certain indices from an array
 - numlines: return the number of lines in a file
 - basiclogger: return a basic logger to the screen and optionally a file
 - remove: delete multiple files and allow for non-existence
 - lt: takes the lesser of x or limit
 - gt: takes the greater of x or limit
 - limit: require x to be within upper and lower limits
 - gaussian: return Gaussian plus constant
 - gaussfit: fit a 1-D Gaussian to X/Y data
 - poly: evaluate a polynomial function of a variable
 - slope: derivative or slope of an array
 - closest: find value in array closest to an input scalar

coords
======

The utils module has many of the general purpose functions.

 - rotsph: Convert coordinates into a new coordinate system given the coordinates of the pole and the "ascending node".
 - rotsphcen: Convert coordinates into a new coordinate system given the coordinates of the origin of the new equator.
 - doPolygonsOverlap: Returns True if two polygons are overlapping.
 - xmatch: Cross-match angular values between RA1/DEC1 and RA2/DEC2.
 - dist: Calculate Euclidian distance between two sets of points.
 - sphdist: Calculate the angular distance between two sets of points.
 - lbd2xyz:  Convert from LON, LAT and DISTANCE to galactocentric cartesian coordinates.
 - xyz2ldb: Convert galactocentric X/Y/Z coordinates to l,b,dist.

bindata
=======

The bindata module is a variant of the scipy.stats.binned_statistic module.  It adds the "mad" and "percentile" statistics.

 - binned_statistic: Compute a binned statistic for one or more sets of data.

astro
=====

Various astronomy-related tools and functions.

 - airtovac: Convert air wavelengths to vacuum wavelengths.
 - vactoair: Convert vacuum wavelengths to air wavelengths.
 - vgsr2vhelio: Convert Galactocentric velocies to heliocentric.
 - vgsr2vlsr: Convert Galactocentric velocies to Local Standard of Rest velocities.
 - galaxy_model: Model of the proper motions and radial velocites of a simple disk galaxy.

job_daemon
==========

The job_daemon module has the capability to ask as a simple python job manager.

 - job_daemon: Run a set of python "jobs" simultaneously.

plotting
========

A set of utility plotting functions.

 - zscaling:  Calculate good min/max scaling values for a data set.
 - hist2d: Plot 2D histogram of data (similar to plt.hist2d).
 - display: Display an image (similar to plt.imshow).
 - plot: Line or scatter plot of data.

db
===

Some tools to interact with with a sqlite3 or PostgresQL database.

 - writecat:  Write a catalog to a database table.
 - createindex: Create an index on a column.
 - analyzetable:  Run "analyze" on a table.
 - query: Run a query on a table.

spec
====

Some spectral processing and analysis tools.

 - trace: Trace the spectrum.  Spectral dimension is assumed to be on the horizontal axis.
 - boxcar: Boxcar extract the spectrum.
 - linefit: Fit Gaussian profile to data with center and sigma fixed
 - extract: Extract a spectrum.
 - emissionlines: Measure the emission lines in an arc lamp spectrum.
 - continuum: Derive the continuum of a spectrum.

robust
======

A collection of robust statistics functions.

 - biweight_mean: Calculate the mean of a data set using bisquare weighting.
 - mean: Robust estimator of the mean of a data set.  Based on the  resistant_mean function from the AstroIDL User's Library.
 - mode: Robust estimator of the mode of a data set using the half-sample mode.
 - std: Robust estimator of the standard deviation of a data set.  Based on the robust_sigma function from the AstroIDL User's Library.
 - checkfit:  Determine the quality of a fit and biweights.
 - linefit: Outlier resistance two-variable linear regression function.
 - polyfit: Outlier resistance two-variable polynomial function fitter.

ladfit
======

The ladfit module has some robust functions for estimating a linear model.

 - ladfit: Fits the paired data {X(i), Y(i)} to the linear model, y = A + Bx, using a "robust" least absolute deviation method.

least_squares
=============

The least_squares module is a variant of the scipy.optimize.least_squares module but with some modifications to curve_fit to
allow for more user inputs.

 - curve_fit: General purpose curve fitting of data.
