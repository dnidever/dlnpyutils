"""
This is from the LWA Software Library (LSL)
https://github.com/lwa-project/lsl
Redistributed under the GNU license.

Small collection of robust statistical estimators based on functions from
Henry Freudenriech (Hughes STX) statistics library (called ROBLIB) that have
been incorporated into the AstroIDL User's Library.  Function included are:
 * biweight_mean - biweighted mean estimator
 * mean - robust estimator of the mean of a data set
 * mode - robust estimate of the mode of a data set using the half-sample
          method
 * std - robust estimator of the standard deviation of a data set
 * checkfit - return the standard deviation and biweights for a fit in order 
              to determine its quality
 * linefit - outlier resistant fit of a line to data
 * polyfit - outlier resistant fit of a polynomial to data

For the fitting routines, the coefficients are returned in the same order as
numpy.polyfit, i.e., with the coefficient of the highest power listed first.

For additional information about the original IDL routines, see:
http://idlastro.gsfc.nasa.gov/contents.html#C17
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
import math
import numpy
from numpy.polynomial.polynomial import polyfit as npp_polyfit, polyval as npp_polyval


__version__ = '0.5'
__all__ = ['biweight_mean', 'mean', 'mode', 'std', 'checkfit', 'linefit', 'polyfit']

__max_iter = 25
__delta = 5.0e-7
__epsilon = 1.0e-20


def __stddev(x):
    return x.std()*numpy.sqrt(x.size/(x.size+1.0))


def biweight_mean(inputData, axis=None, dtype=None):
    """
    Calculate the mean of a data set using bisquare weighting.  
    
    Based on the biweight_mean routine from the AstroIDL User's 
    Library.
    
    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with numpy.mean()
    """
    
    if axis is not None:
        fnc = lambda x: biweight_mean(x, dtype=dtype)
        y0 = numpy.apply_along_axis(fnc, axis, inputData)
    else:
        y = inputData.ravel()
        if type(y).__name__ == "MaskedArray":
            y = y.compressed()
        if dtype is not None:
            y = y.astype(dtype)
            
        n = len(y)
        closeEnough = 0.03*numpy.sqrt(0.5/(n-1))
        
        diff = 1.0e30
        nIter = 0
        
        y0 = numpy.median(y)
        deviation = y - y0
        sigma = std(deviation)
        
        if sigma < __epsilon:
            diff = 0
        while diff > closeEnough:
            nIter = nIter + 1
            if nIter > __max_iter:
                break
            uu = ((y-y0)/(6.0*sigma))**2.0
            uu = numpy.where(uu > 1.0, 1.0, uu)
            weights = (1.0-uu)**2.0
            weights /= weights.sum()
            y0 = (weights*y).sum()
            deviation = y - y0
            prevSigma = sigma
            sigma = std(deviation, zero=True)
            if sigma > __epsilon:
                diff = numpy.abs(prevSigma - sigma) / prevSigma
            else:
                diff = 0.0
                
    return y0


def mean(inputData, cut=3.0, axis=None, dtype=None):
    """
    Robust estimator of the mean of a data set.  Based on the 
    resistant_mean function from the AstroIDL User's Library.
    
    .. versionchanged:: 1.2.1
        Added a ValueError if the distriubtion is too strange
    
    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with numpy.mean()
    """
    
    if axis is not None:
        fnc = lambda x: mean(x, cut=cut, dtype=dtype)
        dataMean = numpy.apply_along_axis(fnc, axis, inputData)
    else:
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
            
        data0 = numpy.median(data)
        maxAbsDev = numpy.median(numpy.abs(data-data0)) / 0.6745
        if maxAbsDev < __epsilon:
            maxAbsDev = (numpy.abs(data-data0)).mean() / 0.8000
            
        cutOff = cut*maxAbsDev
        good = numpy.where( numpy.abs(data-data0) <= cutOff )
        good = good[0]
        dataMean = data[good].mean()
        dataSigma = math.sqrt( ((data[good]-dataMean)**2.0).sum() / len(good) )

        if cut > 1.0:
            sigmacut = cut
        else:
            sigmacut = 1.0
        if sigmacut <= 4.5:
            dataSigma = dataSigma / (-0.15405 + 0.90723*sigmacut - 0.23584*sigmacut**2.0 + 0.020142*sigmacut**3.0)
            
        cutOff = cut*dataSigma
        good = numpy.where(  numpy.abs(data-data0) <= cutOff )
        good = good[0]
        dataMean = data[good].mean()
        if len(good) > 3:
            dataSigma = math.sqrt( ((data[good]-dataMean)**2.0).sum() / len(good) )
        else:
            raise ValueError("Distribution is too strange to compute mean")
            
        if cut > 1.0:
            sigmacut = cut
        else:
            sigmacut = 1.0
        if sigmacut <= 4.5:
            dataSigma = dataSigma / (-0.15405 + 0.90723*sigmacut - 0.23584*sigmacut**2.0 + 0.020142*sigmacut**3.0)
            
        dataSigma = dataSigma / math.sqrt(len(good)-1)
        
    return dataMean


def mode(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.
    
    .. versionadded: 1.0.3
    """
    
    if axis is not None:
        fnc = lambda x: mode(x, dtype=dtype)
        dataMode = numpy.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = data[-1] - data[0]
                N = data.size // 2 + data.size % 2 
                for i in range(0, N):
                    w = data[i+N-1] - data[i] 
                    if w < wMin:
                        wMin = w
                        j = i
                return _hsm(data[j:j+N])
                
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
            
        # The data need to be sorted for this to work
        data = numpy.sort(data)
        
        # Find the mode
        dataMode = _hsm(data)
        
    return dataMode


def std(inputData, zero=False, axis=None, dtype=None):
    """
    Robust estimator of the standard deviation of a data set.  
    
    Based on the robust_sigma function from the AstroIDL User's Library.
    
    .. versionchanged:: 1.2.1
        Added a ValueError if the distriubtion is too strange
    
    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with numpy.std()
    """
    
    if axis is not None:
        fnc = lambda x: std(x, zero=zero, dtype=dtype)
        sigma = numpy.apply_along_axis(fnc, axis, inputData)
    else:
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
            
        if zero:
            data0 = 0.0
        else:
            data0 = numpy.median(data)
        maxAbsDev = numpy.median(numpy.abs(data-data0)) / 0.6745
        if maxAbsDev < __epsilon:
            maxAbsDev = (numpy.abs(data-data0)).mean() / 0.8000
        if maxAbsDev < __epsilon:
            sigma = 0.0
            return sigma
            
        u = (data-data0) / 6.0 / maxAbsDev
        u2 = u**2.0
        good = numpy.where( u2 <= 1.0 )
        good = good[0]
        if len(good) < 3:
            raise ValueError("Distribution is too strange to compute standard deviation")
            
        numerator = ((data[good]-data0)**2.0 * (1.0-u2[good])**4.0).sum()
        nElements = (data.ravel()).shape[0]
        denominator = ((1.0-u2[good])*(1.0-5.0*u2[good])).sum()
        sigma = nElements*numerator / (denominator*(denominator-1.0))
        if sigma > 0:
            sigma = math.sqrt(sigma)
        else:
            sigma = 0.0
            
    return sigma


def checkfit(inputData, inputFit, epsilon, delta, bisquare_limit=6.0):
    """
    Determine the quality of a fit and biweights.  Returns a tuple
    with elements:
     0. Status
     1. Robust standard deviation analog
     2. Fractional median absolute deviation of the residuals
     3. Number of input points given non-zero weight in the calculation
     4. Bisquare weights of the input points
     5. Residual values scaled by sigma
    
    This function is based on the rob_checkfit routine from the AstroIDL 
    User's Library.
    """
    
    status = 0
    
    data = inputData.ravel()
    fit = inputFit.ravel()
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if type(fit).__name__ == "MaskedArray":
        fit = fit.compressed()

    deviation = data - fit
    sigma = std(deviation, zero=True)
    if sigma < epsilon:
        return (status, sigma, 0.0, 0, 0.0, 0.0)
    
    toUse = (numpy.where( numpy.abs(fit) > epsilon ))[0]
    if len(toUse) < 3:
        fracDev = 0.0
    else:
        fracDev = numpy.median(numpy.abs(deviation[toUse]/fit[toUse]))
    if fracDev < delta:
        return (status, sigma, fracDev, 0, 0.0, 0.0)
        
    status = 1
    scaledResids = numpy.abs(deviation)/(bisquare_limit*sigma)
    toUse = (numpy.where(scaledResids > 1))[0]
    if len(toUse) > 0:
        scaledResids[toUse] = 1.0
    nGood = len(data) - len(toUse)
    
    biweights = (1.0 - scaledResids**2.0)
    biweights = biweights / biweights.sum()
    
    return (status, sigma, fracDev, nGood, biweights, scaledResids)


def linefit(inputX, inputY, max_iter=25, bisector=False, bisquare_limit=6.0, close_factor=0.03):
    """
    Outlier resistance two-variable linear regression function.
    
    Based on the robust_linefit routine in the AstroIDL User's Library.
    """
    
    xIn = inputX.ravel()
    yIn = inputY.ravel()
    if type(yIn).__name__ == "MaskedArray":
        xIn = xIn.compress(numpy.logical_not(yIn.mask))
        yIn = yIn.compressed()
    n = len(xIn)
    
    x0 = xIn.mean()
    y0 = yIn.mean()
    x = xIn - x0
    y = yIn - y0
    
    cc = numpy.zeros(2)
    ss = numpy.zeros(2)
    sigma = 0.0
    yFit = yIn
    badFit = 0
    nGood = n
    
    lsq = 0.0
    yp = y
    if n > 5:
        s = numpy.argsort(x)
        u = x[s]
        v = y[s]
        nHalf = n//2 - 1
        x1 = numpy.median(u[0:nHalf+1])
        x2 = numpy.median(u[nHalf+1:])
        y1 = numpy.median(v[0:nHalf+1])
        y2 = numpy.median(v[nHalf+1:])
        if numpy.abs(x2-x1) < __epsilon:
            x1, x2 = u[0], u[-1]
            y1, y2 = v[0], v[-1]
        cc[1] = (y2-y1)/(x2-x1)
        cc[0] = y1 - cc[1]*x1
        yFit = cc[0] + cc[1]*x
        status, sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
        if nGood < 2:
            lsq = 1.0
            
    if lsq == 1 or n < 6:
        sx = x.sum()
        sy = y.sum()
        sxy = (x*y).sum()
        sxx = (x*x).sum()
        d = sxx - sx*sx
        if numpy.abs(d) < __epsilon:
            return (0.0, 0.0)
        ySlope = (sxy - sx*sy) / d
        yYInt = (sxx*sy - sx*sxy) / d
        
        if bisector:
            syy = (y*y).sum()
            d = syy - sy*sy
            if numpy.abs(d) < __epsilon:
                return (0.0, 0.0)
            tSlope = (sxy - sy*sx) / d
            tYInt = (syy*sx - sy*sxy) / d
            if numpy.abs(tSlope) < __epsilon:
                return (0.0, 0.0)
            xSlope = 1.0/tSlope
            xYInt = -tYInt / tSlope
            if ySlope > xSlope:
                a1 = yYInt
                b1 = ySlope
                r1 = numpy.sqrt(1.0+ySlope**2.0)
                a2 = xYInt
                b2 = xSlope
                r2 = numpy.sqrt(1.0+xSlope**2.0)
            else:
                a2 = yYInt
                b2 = ySlope
                r2 = numpy.sqrt(1.0+ySlope**2.0)
                a1 = xYInt
                b1 = xSlope
                r1 = numpy.sqrt(1.0+xSlope**2.0)
            yInt = (r1*a2 + r2*a1) / (r1 + r2)
            slope = (r1*b2 + r2*b1) / (r1 + r2)
            r = numpy.sqrt(1.0+slope**2.0)
            if yInt > 0:
                r = -r
            u1 = slope / r
            u2 = -1.0/r
            u3 = yInt / r
            yp = u1*x + u2*y + u3
            yFit = y*0.0
            ss = yp
        else:
            slope = ySlope
            yInt = yYInt
            yFit = yInt + slope*x
        cc[0] = yInt
        cc[1] = slope
        status, sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
        
    if nGood < 2:
        cc[0] = cc[0] + y0 - cc[1]*x0
        return cc[::-1]
        
    sigma1 = min([(100.0*sigma), 1e20])
    closeEnough = close_factor * numpy.sqrt(0.5/(n-1))
    if closeEnough < __delta:
        closeEnough = __delta
    diff = 1.0e20
    nIter = 0
    while diff > closeEnough and nIter < max_iter:
        nIter = nIter + 1
        sigma2 = sigma1
        sigma1 = sigma
        sx = (biweights*x).sum()
        sy = (biweights*y).sum()
        sxy = (biweights*x*y).sum()
        sxx = (biweights*x*x).sum()
        d = sxx - sx*sx
        if numpy.abs(d) < __epsilon:
            return (0.0, 0.0)
        ySlope = (sxy - sx*sy) / d
        yYInt = (sxx*sy - sx*sxy) / d
        slope = ySlope
        yInt = yYInt
        
        if bisector:
            syy = (biweights*y*y).sum()
            d = syy - sy*sy
            if numpy.abs(d) < __epsilon:
                return (0.0, 0.0)
            tSlope = (sxy - sy*sx) / d
            tYInt = (syy*sx - sy*sxy) / d
            if numpy.abs(tSlope) < __epsilon:
                return (0.0, 0.0)
            xSlope = 1.0/tSlope
            xYInt = -tYInt / tSlope
            if ySlope > xSlope:
                a1 = yYInt
                b1 = ySlope
                r1 = numpy.sqrt(1.0+ySlope**2.0)
                a2 = xYInt
                b2 = xSlope
                r2 = numpy.sqrt(1.0+xSlope**2.0)
            else:
                a2 = yYInt
                b2 = ySlope
                r2 = numpy.sqrt(1.0+ySlope**2.0)
                a1 = xYInt
                b1 = xSlope
                r1 = numpy.sqrt(1.0+xSlope**2.0)
            yInt = (r1*a2 + r2*a1) / (r1 + r2)
            slope = (r1*b2 + r2*b1) / (r1 + r2)
            r = numpy.sqrt(1.0+slope**2.0)
            if yInt > 0:
                r = -r
            u1 = slope / r
            u2 = -1.0/r
            u3 = yInt / r
            yp = u1*x + u2*y + u3
            yFit = y*0.0
            ss = yp
        else:
            yFit = yInt + slope*x
        cc[0] = yInt
        cc[1] = slope
        status, sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta, bisquare_limit=bisquare_limit)
        
        if status == 0:
            break
            
        if nGood < 2:
            badFit = 1
            break
        diff = min([numpy.abs(sigma1 - sigma)/sigma, numpy.abs(sigma2 - sigma)/sigma])
        
    cc[0] = cc[0] + y0 - cc[1]*x0
    return cc[::-1]


def __polyfit_rescale(coeffs, x0, y0):
    order = len(coeffs)-1
    
    if order == 1:
        coeffs[0] = coeffs[0]-coeffs[1]*x0 + y0
    elif order == 2:
        coeffs[0] = coeffs[0]-coeffs[1]*x0+coeffs[2]*x0**2 + y0
        coeffs[1] = coeffs[1]-2.*coeffs[2]*x0
    elif order == 3:
        coeffs[0] = coeffs[0]-coeffs[1]*x0+coeffs[2]*x0**2-coeffs[3]*x0**3 + y0
        coeffs[1] = coeffs[1]-2.*coeffs[2]*x0+3.*coeffs[3]*x0**2
        coeffs[2] = coeffs[2]-3.*coeffs[3]*x0
    elif order == 4:
        coeffs[0] = coeffs[0]-   coeffs[1]*x0+coeffs[2]*x0**2-coeffs[3]*x0**3+coeffs[4]*x0**4+ y0
        coeffs[1] = coeffs[1]-2.*coeffs[2]*x0+3.*coeffs[3]*x0**2-4.*coeffs[4]*x0**3
        coeffs[2] = coeffs[2]-3.*coeffs[3]*x0+6.*coeffs[4]*x0**2
        coeffs[3] = coeffs[3]-4.*coeffs[4]*x0
    elif order == 5:
        coeffs[0] = coeffs[0]-  coeffs[1]*x0+coeffs[2]*x0**2-coeffs[3]*x0**3+coeffs[4]*x0**4-coeffs[5]*x0**5+ y0
        coeffs[1] = coeffs[1]-2.*coeffs[2]*x0+ 3.*coeffs[3]*x0**2- 4.*coeffs[4]*x0**3+5.*coeffs[5]*x0**4
        coeffs[2] = coeffs[2]-3.*coeffs[3]*x0+ 6.*coeffs[4]*x0**2-10.*coeffs[5]*x0**3
        coeffs[3] = coeffs[3]-4.*coeffs[4]*x0+10.*coeffs[5]*x0**2
        coeffs[4] = coeffs[4]-5.*coeffs[5]*x0
    return coeffs[::-1]


def polyfit(inputX, inputY, order, max_iter=25):
    """
    Outlier resistance two-variable polynomial function fitter.
    
    Based on the robust_poly_fit routine in the AstroIDL User's 
    Library.
    """
    
    x = inputX.ravel()
    y = inputY.ravel()
    if type(y).__name__ == "MaskedArray":
        x = x.compress(numpy.logical_not(y.mask))
        y = y.compressed()
    n = len(x)
    
    if order < 6:
        x0 = x.mean()
        y0 = y.mean()
        u = x - x0
        v = y - y0
    else:
        u = x
        v = y
        
    minPts = order + 1
    if (n//4)*4 == n:
        need2 = 1
    else:
        need2 = 0
    n3 = 3*n//4
    n1 = n//4

    nSeg = order + 2
    if (nSeg//2)*2 == nSeg:
        nSeg = nSeg + 1
    min_pts = nSeg*3
    yp = y
    if n < 10000:
        lsqFit = 1
        cc = npp_polyfit(u, v, order)
        yFit = npp_polyval(u, cc)
    else:
        lsqfit = 0
        q = numpy.argsort(u)
        u = u[q]
        v = v[q]
        nPerSeg = numpy.zeros(nSeg, dtype=numpy.int64) + n//nSeg
        nLeft = n - nPerSeg[0]*nSeg
        nPerSeg[nSeg//2] = nPerSeg[nSeg//2] + nLeft
        r = numpy.zeros(nSeg, dtype=numpy.float64)
        s = numpy.zeros(nSeg, dtype=numpy.float64)
        r[0] = numpy.median(u[0:nPerSeg[0]])
        s[0] = numpy.median(v[0:nPerSeg[0]])
        i2 = nPerSeg[0]-1
        for i in range(1, nSeg):
            i1 = i2
            i2 = i1 + nPerSeg[i]
            r[i] = numpy.median(u[i1:i2])
            s[i] = numpy.median(v[i1:i2])
        cc = npp_polyfit(r, s, order)
        yFit = npp_polyval(u, cc)
        
    status,  sigma, fracDev, nGood, biweights, scaledResids = checkfit(v, yFit, __epsilon, __delta)
    if nGood == 0:
        return __polyfit_rescale(cc, x0, y0)
    if nGood < minPts:
        if lsqFit == 0:
            cc = npp_polyfit(u, v, order)
            yFit = npp_polyval(u, cc)
            status, sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
            if nGood == 0:
                return 0
            nGood = n - nGood
        if nGood < minPts:
            return 0
            
    closeEnough = 0.03*numpy.sqrt(0.5/(n-1))
    if closeEnough < __delta:
        closeEnough = __delta
    diff = 1.0e10
    sigma1 = min([100.0*sigma, 1e20])
    nIter = 0
    while diff > closeEnough and nIter < max_iter:
        nIter = nIter + 1
        sigma2 = sigma1
        sigma1 = sigma
        g = (numpy.where(biweights > 0))[0]
        if len(g) < len(biweights):
            u = u[g]
            v = v[g]
            biweights = biweights[g]
        cc = npp_polyfit(u, v, order, w=biweights**2)
        yFit = npp_polyval(u, cc)
        status, sigma, fracDev, nGood, biweights, scaledResids = checkfit(v, yFit, __epsilon, __delta)
        if status == 0:
            break
        if nGood < minPts:
            break
        diff = min([numpy.abs(sigma1 - sigma)/sigma, numpy.abs(sigma2 - sigma)/sigma])
        
    return __polyfit_rescale(cc, x0, y0)
