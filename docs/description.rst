***********
Description
***********

There are currently 11 main modules in |dlnpyutils|.  They are described in detail below.


utils
=====

The :mod:`~dlnpyutils.utils` module has many of the general purpose functions.

 - :func:`~dlnpyutils.utils.size`: Returns the number of elements.
 - :func:`~dlnpyutils.utils.mad`: median absolute deviation of array.
 - :func:`~dlnpyutils.utils.minmax`: minimum and maximum of an array.
 - :func:`~dlnpyutils.utils.where`: Wrapper around numpy.where() to be more like IDL-like.
 - :func:`~dlnpyutils.utils.stat`: many useful statistics of an array
 - :func:`~dlnpyutils.utils.strlen`: number of characters in a string array or list
 - :func:`~dlnpyutils.utils.strip`: strip whitespace from string array or list
 - :func:`~dlnpyutils.utils.strjoin`: combine string arrays or scalars
 - :func:`~dlnpyutils.utils.strsplit`: split string arrays
 - :func:`~dlnpyutils.utils.pathjoin`: join two pathname components
 - :func:`~dlnpyutils.utils.first_el`: return the first element of an array or list
 - :func:`~dlnpyutils.utils.grep`: grep on a string array
 - :func:`~dlnpyutils.utils.readlines`: read a file into a string array
 - :func:`~dlnpyutils.utils.writelines`: write a string array to a file
 - :func:`~dlnpyutils.utils.remove_indices`: remove certain indices from an array
 - :func:`~dlnpyutils.utils.numlines`: return the number of lines in a file
 - :func:`~dlnpyutils.utils.basiclogger`: return a basic logger to the screen and optionally a file
 - :func:`~dlnpyutils.utils.remove`: delete multiple files and allow for non-existence
 - :func:`~dlnpyutils.utils.exists`: Check if a list of files exists.
 - :func:`~dlnpyutils.utils.lt`: takes the lesser of x or limit
 - :func:`~dlnpyutils.utils.gt`: takes the greater of x or limit
 - :func:`~dlnpyutils.utils.limit`: require x to be within upper and lower limits
 - :func:`~dlnpyutils.utils.valrange`: returns range of values.
 - :func:`~dlnpyutils.utils.signs`: Return the sign of input.  Return +1.0 for 0.0.
 - :func:`~dlnpyutils.utils.scale`: Maps an array onto a new scale given two values on the old and new scales.
 - :func:`~dlnpyutils.utils.scale_vector`:  Scale a vector to minrange and maxrange.
 - :func:`~dlnpyutils.utils.quadratic_bisector`: Calculate the axis of symmetric or bisector of parabola.
 - :func:`~dlnpyutils.utils.quadratic_coefficients`: Calculate the quadratic coefficients from the three points.
 - :func:`~dlnpyutils.utils.wtmean`: Calculate weighted mean and error.
 - :func:`~dlnpyutils.utils.mediqrslope`: Calculate robust slope from median of first quartile and points in the 3+4th quartile and median of 4th quartile and points in the 1+2nd quartile.  The median is then found of all the slopes.
 - :func:`~dlnpyutils.utils.iqrslope`: Calculate robust slope from median of first quartile and last quartile of points.
 - :func:`~dlnpyutils.utils.medslope`: Calculate robust slope from median of first half and last half of points.
 - :func:`~dlnpyutils.utils.wtslope`: Calculate weighted slope and error.
 - :func:`~dlnpyutils.utils.robust_slope`: Calculate robust weighted slope.
 - :func:`~dlnpyutils.utils.wtmedian`: Weighted median by sorting the weighted values and finding the point of half the total weights.
 - :func:`~dlnpyutils.utils.iqrstdev`: Use the interquartile range to estimate the standard deviation robustly.
 - :func:`~dlnpyutils.utils.sigclipmean`: Sigma-clipped mean.
 - :func:`~dlnpyutils.utils.gausswtmean`: Compute weighted mean using a Gaussian with center of the median and sigma of the MAD.
 - :func:`~dlnpyutils.utils.gmean`: Compute geometric mean.   
 - :func:`~dlnpyutils.utils.skewquartile`: Measure the skewness robustly based on quartiles.   
 - :func:`~dlnpyutils.utils.skewgauss`:  Return a skewed Gaussian.
 - :func:`~dlnpyutils.utils.gaussian`: Return 1-D Gaussian.
 - :func:`~dlnpyutils.utils.gaussbin`: Return 1-D binned Gaussian.
 - :func:`~dlnpyutils.utils.gaussfit`: fit a 1-D Gaussian to X/Y data.
 - :func:`~dlnpyutils.utils.voigt`: Return the Voigt line shape at x with Lorentzian component HWHM gamma and Gaussian sigma.
 - :func:`~dlnpyutils.utils.voigtfit`: Fit a Voigt profile to data.
 - :func:`~dlnpyutils.utils.voigtarea`: Compute area of Voigt profile.
 - :func:`~dlnpyutils.utils.poly`: evaluate a polynomial function of a variable
 - :func:`~dlnpyutils.utils.poly_fit`: Fit a polynomial to X/Y data.
 - :func:`~dlnpyutils.utils.slope`: derivative or slope of an array 
 - :func:`~dlnpyutils.utils.smooth`: Boxcar smooth an array.
 - :func:`~dlnpyutils.utils.gsmooth`: Gaussian smooth an array or image.
 - :func:`~dlnpyutils.utils.savol`: Savitzky-Golay smoothing of data.
 - :func:`~dlnpyutils.utils.rebin`: Rebin data.
 - :func:`~dlnpyutils.utils.roi_cut`: Use cuts in a 2D plane to select points from arrays.
 - :func:`~dlnpyutils.utils.create_index`: Create an index of array values like reverse indices.
 - :func:`~dlnpyutils.utils.match`: Function to match values in two vectors.
 - :func:`~dlnpyutils.utils.interp`: Interpolate with extrapolation.
 - :func:`~dlnpyutils.utils.concatenate`: Concatenate two or more numpy structured arrays (or list of them).
 - :func:`~dlnpyutils.utils.addcatcols`: Add new columns to an existing numpty structured array catalog.
 - :func:`~dlnpyutils.utils.clicker`: Click on a plot and return the coordinates.
 - :func:`~dlnpyutils.utils.add_elements`: Add more elements to a catalog.
 - :func:`~dlnpyutils.utils.ellipsecoords`: Create coordinates of an ellipse.
 - :func:`~dlnpyutils.utils.closest`: Find value in array closest to an input scalar.   
 - :func:`~dlnpyutils.utils.sexig2ten`: Convert sexigesimal to decimal.
 - :func:`~dlnpyutils.utils.fread`: Read the values in a string into variables using a format string.
 - :func:`~dlnpyutils.utils.randf`: Pick random floats between low and high (inclusive).
 - :func:`~dlnpyutils.utils.isnumber`: Returns True if string is a number or float.

coords
======

The :mod:`~dlnpyutils.coords` module has coordinate-related tools.

 - :func:`~dlnpyutils.coords.rotsph`: Convert coordinates into a new coordinate system given the coordinates of the pole and the "ascending node".
 - :func:`~dlnpyutils.coords.rotsphcen`: Convert coordinates into a new coordinate system given the coordinates of the origin of the new equator.
 - :func:`~dlnpyutils.coords.doPolygonsOverlap`: Returns True if two polygons are overlapping.
 - :func:`~dlnpyutils.coords.xmatch`: Cross-match angular values between RA1/DEC1 and RA2/DEC2.
 - :func:`~dlnpyutils.coords.dist`: Calculate Euclidian distance between two sets of points.
 - :func:`~dlnpyutils.coords.sphdist`: Calculate the angular distance between two sets of points.
 - :func:`~dlnpyutils.coords.lbd2xyz`:  Convert from LON, LAT and DISTANCE to galactocentric cartesian coordinates.
 - :func:`~dlnpyutils.coords.xyz2lbd`: Convert galactocentric X/Y/Z coordinates to l,b,dist.

bindata
=======

The :mod:`~dlnpyutils.bindata` module is a variant of the scipy.stats.binned_statistic module.  It adds the "mad" and "percentile" statistics.

 - :func:`~dlnpyutils.bindata.binned_statistic`: Compute a binned statistic for one or more sets of data.

astro
=====

Various astronomy-related tools and functions.

 - :func:`~dlnpyutils.astro.airtovac`: Convert air wavelengths to vacuum wavelengths.
 - :func:`~dlnpyutils.astro.vactoair`: Convert vacuum wavelengths to air wavelengths.
 - :func:`~dlnpyutils.astro.vgsr2vhelio`: Convert Galactocentric velocies to heliocentric.
 - :func:`~dlnpyutils.astro.vgsr2vlsr`: Convert Galactocentric velocies to Local Standard of Rest velocities.
 - :func:`~dlnpyutils.astro.galaxy_model`: Model of the proper motions and radial velocites of a simple disk galaxy.

job_daemon
==========

The :mod:`~dlnpyutils.job_daemon` module has the capability to ask as a simple python job manager.

 - :func:`~dlnpyutils.job_daemon.job_daemon`: Run a set of python "jobs" simultaneously.

plotting
========

A set of utility plotting functions.

 - :func:`~dlnpyutils.plotting.zscaling`:  Calculate good min/max scaling values for a data set.
 - :func:`~dlnpyutils.plotting.hist2d`: Plot 2D histogram of data (similar to plt.hist2d).
 - :func:`~dlnpyutils.plotting.display`: Display an image (similar to plt.imshow).
 - :func:`~dlnpyutils.plotting.plot`: Line or scatter plot of data.

db
===

Some tools to interact with with a sqlite3 or PostgresQL database.

 - :func:`~dlnpyutils.db.writecat`:  Write a catalog to a database table.
 - :func:`~dlnpyutils.db.createindex`: Create an index on a column.
 - :func:`~dlnpyutils.db.analyzetable`:  Run "analyze" on a table.
 - :func:`~dlnpyutils.db.query`: Run a query on a table.

spec
====

Some spectral processing and analysis tools.

 - :func:`~dlnpyutils.spec.trace`: Trace the spectrum.  Spectral dimension is assumed to be on the horizontal axis.
 - :func:`~dlnpyutils.spec.boxcar`: Boxcar extract the spectrum.
 - :func:`~dlnpyutils.spec.linefit`: Fit Gaussian profile to data with center and sigma fixed
 - :func:`~dlnpyutils.spec.extract`: Extract a spectrum.
 - :func:`~dlnpyutils.spec.emissionlines`: Measure the emission lines in an arc lamp spectrum.
 - :func:`~dlnpyutils.spec.continuum`: Derive the continuum of a spectrum.

robust
======

A collection of robust statistics functions transflated from the AstroIDL User's Library.

 - :func:`~dlnpyutils.robust.biweight_mean`: Calculate the mean of a data set using bisquare weighting.
 - :func:`~dlnpyutils.robust.mean`: Robust estimator of the mean of a data set.  Based on the resistant_mean function from the AstroIDL User's Library.
 - :func:`~dlnpyutils.robust.mode`: Robust estimator of the mode of a data set using the half-sample mode.
 - :func:`~dlnpyutils.robust.std`: Robust estimator of the standard deviation of a data set.  Based on the robust_sigma function from the AstroIDL User's Library.
 - :func:`~dlnpyutils.robust.checkfit`:  Determine the quality of a fit and biweights.
 - :func:`~dlnpyutils.robust.linefit`: Outlier resistance two-variable linear regression function.
 - :func:`~dlnpyutils.robust.polyfit`: Outlier resistance two-variable polynomial function fitter.

ladfit
======

The :mod:`~dlnpyutils.ladfit` module has some robust functions for estimating a linear model.

 - :func:`~dlnpyutils.ladfit.ladfit`: Fits the paired data {X(i), Y(i)} to the linear model, y = A + Bx, using a "robust" least absolute deviation method.

minpack
=======

The :mod:`~dlnpyutils.minpack`, :mod:`~dlnpyutils.least_squares`, and :mod:`~dlnpyutils.trf` module is a variant of the scipy.optimize.least_squares module but with some modifications to curve_fit to
allow for more user inputs.

 - :func:`~dlnpyutils.minpack.curve_fit`: General purpose curve fitting of data.
