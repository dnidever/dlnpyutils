import os
import numpy as np
from . import utils,robust

# Sinc resample spectra.  Can since resample and average multiple
# spectra at the same time.

def standardize(x,y,yerr,res):
    """
    Standardize data.  Puts inputs into a "standard" format of a list
    of numpy arrays.

    Parameters
    ----------
    x : numpy array or list
       Input X-values.  This can be a list of arrays, i.e. multiple
         spectra.
    y : numpy array or list
       Input Y-values.  This can be a list of arrays, i.e. multiple
         sets of data/spectra.
    yerr : numpy array or list
       Uncertainties in Y.  This can be None.
    res : numpy array or list:
       Resolution width (FWHM).  This can be None.

    Returns
    -------
    x : list of numpy arrays
       Same as input, but standardized to a list of numpy arrays.
    y : list of numpy arrays
       Same as input, but standardized to a list of numpy arrays.
    yerr : list of numpy arrays
       Same as input, but standardized to a list of numpy arrays.
    res : list of numpy arrays
       Same as input, but standardized to a list of numpy arrays.

    Examples
    --------

    x,y,yerr,res = standardize(x,y,yerr,res)
    
    """

    if type(x) != type(y):
        raise ValueError('X and Y must have same type')
    if type(x) is not list:
        x = [x]
        y = [y]
        yerr = [yerr]
        res = [res]
    # Sort each one by X
    for i in range(len(x)):
        x1 = x[i]
        si = np.argsort(x1)
        x[i] = x[i][si]
        y[i] = y[i][si]
        if yerr[i] is not None:
            yerr[i] = yerr[i][si]
        if res[i] is not None:
            res[i] = res[i][si]
    return x,y,yerr,res

def getscale(x,y,yerr,kind='median',order=None):
    """
    Get the scale of the spectra so it is easier to average them.

    Parameters
    ----------
    x : list of numpy arrays
       Input X-values.  This can be a list of arrays, i.e. multiple
         spectra.
    y : list of numpy arrays
       Input y-values.  This can be a list of arrays, i.e. multiple
         spectra.
    yerr : list of numpy arrays
       Uncertainties in "y".  This can be None.
    kind : str, optional
       Type of scaling to use.  Options are: 'median', 'poly', 'medfilt',
         'gaussfilt'.  Default is 'median'.
    order : int, optional
       Information for determining scale.  For 'poly' this is the polynomial
         order (default 3).  For 'medfilt' and 'gaussfilt' this is the bin
         size and FWHM, respectively(default 51).

    Returns
    -------
    y : list of numpy arrays
       Same as input but rescaled.
    yerr : list of numpy arrays
       Same as input but rescaled.
    scales : list of numpy arrays
       Scaling for each spectrum.  This is either a scalar or
         array for each spectrum.    

    Examples
    --------
    y,yerr = rescale(x,y,yerr,kind='medfilt',order=40)
    
    """

    # Loop over the spectra
    scales = len(x)*[None]  # initalize the scales list
    for i in range(x):
        x1 = x[i]
        y1 = y[i]
        yerr1 = yerr[i]
        if yerr1 is None:
            yerr1 = np.ones(y1.shape,float)
        good = (np.isfinite(y1) & np.isfinite(yerr1) & (yerr1 < 1e10))
        # Determine scaling
        if kind=='median':
            scale1 = np.nanmedian(y1[good])
        elif kind=='poly':
            if order is None:
                order = 3
            coef1 = robust.polyfit(x1[good],y1[good],order)
            scale1 = np.polyval(coef1,x1)
        elif kind=='medfilt':
            if order is None:
                order = 51
            if order % 2 == 0: order+=1                
            if np.sum(~good)>0:
                temp = y1.copy()                
                temp[~good] = np.nan
                scale1 = utils.nanmedfilt(temp,order)
            else:
                scale1 = utils.median_filter(y1,order)
        elif kind=='gaussfilt':
            if order is None:
                order = 51
            temp = y1.copy()  # gsmooth() handles NaNs well
            temp[~good] = np.nan
            scale1 = utils.gsmooth(temp,order)
        else:
            raise ValueYerr('kind = '+str(kind)+' not supported')
        # Rescale the arrays
        y1 /= scale1
        yerr1 /= scale1
        # Stick the information back in the lists
        y[i] = y1
        if yerr[i] is not None:
            yerr[i] = yerr1
        scales[i] = scale1
    return y,yerr,scales

def mergearrays(x,y,yerr,res,scales):
    """
    Merge the arrays of multiple spectra into a single sorted array.

    Parameters
    ----------
    x : list of numpy arrays
       Input X-values.  This can be a list of arrays, i.e. multiple
         spectra.
    y : list of numpy arrays
       Input y-values.  This can be a list of arrays, i.e. multiple
         spectra.
    yerr : list of numpy arrays
       Uncertainties in "y".  This can be None.
    res : list of numpy arrays
       Resolution width (FWHM).  This can be None.
    scales : list of scalars/numpy arrays
       The values used to rescale the spectra.

    Returns
    -------
    mx : numpy array
       Merged and sorted x-values.
    my : numpy array
       Merged and sorted y values.
    myerr : numpy array
       Merged and sorted yerr values.
    mres : numpy array
       Merged and sorted res values.
    mscales : numpy array
       Merged and sorted scales.

    Examples
    --------

    mx,my,myerr,mres,mscales = mergearrays(x,y,yerr,res,scales)

    """

    # Initialize the merged arrays
    n = int([len(xi) for xi in x])
    mx = np.zeros(n,float)
    my = np.zeros(n,float)
    myerr = np.zeros(n,float)
    mres = np.zeros(n,float)
    mscales = np.zeros(n,float)
    count = 0
    # Loop over the spectra and merge the arrays
    for i in range(len(x)):
        n1 = len(x[i])
        mx[count:count+n1] = x[i]
        my[count:count+n1] = y[i]        
        if yerr[i] is not None:
            myerr[count:count+n1] = yerr[i]
        else:
            # Use yerr=1 for None
            myerr[count:count+n1] = np.ones(len(x[i]),float)
        if res[i] is not None:
            mres[count:count+n1] = res[i]
        else:
            # Use res=2*dx (Nyquist) for None
            dx = np.gradient(x[i])
            mres[count:count+n1] = 2*dx
        # Should handle both scalar and arrays well
        mscales[count:count+n1] = scales[i]
        count += n1
    # Sort the values
    si = np.argsort(mx)
    mx = mx[si]
    my = my[si]
    myerr = myerr[si]
    mres = mres[si]
    mscales = mscales[si]

    return mx,my,myerr,mres,mscales

def findclosest(mx,xi,start):
    """
    Find closest mx point to xi.
    """
    index = start
    if mx[index]==xi:
        return index
    # Go forward
    if mx[index]<xi:
        while (mx[index]<=xi):
            index += 1
        if index>0 and abs(mx[index-1]-xi)<abs(mx[index]-xi):
            index -= 1
    # Go backwards
    else:
        while (mx[index]>=xi):
            index -= 1
        if index<(len(mx)-1) and abs(mx[index+1]-xi)<abs(mx[index]-xi):
            index += 1
    return index
    
def getlowhigh(mx,xi,start,lim):
    """
    Get low and high index.
    """
    # -- Get Low index --
    low = start
    while (xi-mx[low] < lim):
        low -= 1
    if xi-mx[low] > lim:  # went over
        low += 1
    # -- Get high index --
    high = start
    while (mx[high]-xi < lim):
        high += 1
    if mx[high]-xi > lim:  # went over
        high -= 1
    return low,high
    

def findindex(mx,mres,ksize,xout):
    """
    Find low/high index of mx for each value of xout.

    Parameters
    ----------
    mx : numpy array
       Merged and sorted input X array.
    mres : numpy array
       Merged and sorted resolution width (FWHM) array.
    ksize : int
       Lanczos filter size.
    xout : numpy array
       Output X array.

    Returns
    -------
    indexes : numpy array
       Numpy array of low/high index values for each
          xout value.  
    resout : numpy array
       Average resolution width (FWHM) array for xout.
    
    Examples
    --------

    indexes,resout = findindex(mx,mres,ksize,xout)

    """

    # Lanczos interpolation
    # https://en.wikipedia.org/wiki/Lanczos_resampling
    # S(x) = Sum_(i=floor(x)-a+1)^(floor(x)+a) s(i) * L(x-i)
    # where a is the kernel filter size (ksize)
    #
    # Lanczos window
    # L(x) = a*sin(pi*x)*sin(pi*x/a)/(pi^2 * x^2)  -a<=x<=a and x!=0
    #        1                                     x=0
    #        0                                     otherwise
    
    # Loop over xout
    indexes = np.zeros((len(xout),2),int)
    low = 0
    high = 0
    closeind = 0
    resout = np.zeros(len(xout),float)
    for i in range(len(xout)):
        xi = xout[i]
        # Find closest input x value to this xout value
        closeind = findclosest(mx,xi,closeind)
        # Limit to use for upper/lower region of kernel (in units of x)
        # res is FWHM or 2 for Nyquist
        lim = ksize*mres[closeind]/2.0
        # Find low/high indexes
        low,high = getlowhigh()
        # Find average kernel size
        resi = np.mean(mres[low:high+1])
        lim = ksize*resi/2.0
        # Find low/high indexes
        low,high = getlowhigh(mx,xi,closeind,lim)
        indexes[i,0] = low
        indexes[i,1] = high
        resout[i] = resi
    return indexes,resout

def doresample(mx,my,myerr,mres,mscales,xout,indexes,kind='sinc',ksize=3):
    """
    Perform the Sinc/Lanczos resampling of the y/yerr arrays.

    Parameters
    ----------
    mx : numpy array
       Input merged and sorted X values.  This can be pixels, wavelength,
         or anything else.
    my : numpy array
       Input merged, sorted and rescaled y values.
    myerr : numpy array
       Input merged, sorted and rescaled yerr values.
    mres : numpy array
       Input resolution width (FWHM) values in units of X.
    mscales : numpy array
       Merged and sorted scaling values for each spectrum.
    xout : numpy array
       Output X array.
    indexes : numpy array
       Numpy array of low/high index values for each
          xout value.
    resout : numpy array
       Average resolution width (FWHM) array for xout.
    kind : str, optional
       Type of resampling.  Either 'sinc' (dampled sinc) or 'lanczos'.
          Default is 'sinc'.
    ksize : int, optional
       Sinc/Lanczos filter size. Generally values between 3 and 7.
          Default is 3.

    Returns
    -------
    yout : numpy array
       Output resampled y array.
    yerrout : numpy array
       Output resampled yerr array.
    scalesout : numpy array
       Output resampled scales array.

    Examples
    --------

    yout,yerrout,scalesout = doresample(mx,my,myerr,mres,mscales,indexes,resout,kind,ksize)

    """

    # Initialize the output errays
    yout = np.zeros(len(xout),float)
    yerrout = np.zeros(len(xout),float)
    scalesout = np.zeros(len(xout),float)
    
    # Loop over each xout value and calculate the sinc resampled value
    for i in range(xout):
        # Get low/high index values from indexes
        lo,hi = indexes[i,:]
        x1 = mx[lo:hi+1]
        y1 = my[lo:hi+1]
        yerr1 = myerr[lo:hi+1]
        scales1 = mscales[lo:hi+1]

        # Lanczos interpolation
        
        # Calculate the kernel
        # Sinc kernel
        #  sinc = sin(x)/x, with exponential dampening
        if kind=='sinc':
            # From Holtzman's sincint() function
            # https://github.com/sdss/apogee_drp/blob/daily/python/apogee_drp/apred/sincint.py
            # dampfac = 3.25*nres/2.
            # ksize = int(21*nres/2.)
            # if ksize % 2 == 0 : ksize +=1
            # nhalf = ksize//2 
            #
            # xkernel = np.arange(ksize)-nhalf - fx[i]
            # # in units of Nyquist
            # xkernel /= (nres/2.)
            # u1 = xkernel/dampfac
            # u2 = np.pi*xkernel
            #
            # kernel = np.exp(-(u1**2)) * np.sin(u2) / u2
            # kernel /= (nres/2.)
            # # the value at x = 0 is defined to be the limiting value
            # kernel[u2 == 0] = 1

            # ksize is the kernel filter size ("a" above), in units of Nyquist
            # units of Nyquist is res/2
            xkernel = (x1-xout[i])
            xkernel /= (resout[i]/2.0)  # in units of Nyquist
            dampfac = 3.25*resout[i]/2.0
            u1 = xkernel/dampfac
            u2 = np.pi*xkernel
            u2[xkernel==0] = 1
            kernel = np.exp(-(u1**2))*np.sin(u2)/u2
            kernel /= (resout[i]/2.0)
            kernel[xkernel==0] = 1
            
        # Lanczos kernel
        #   sinc mulitiplied by Lanczos windew function, wider sinc
        elif kind=='lanczos':
            # Lanczos interpolation
            # https://en.wikipedia.org/wiki/Lanczos_resampling
            # S(x) = Sum_(i=floor(x)-a+1)^(floor(x)+a) s(i) * L(x-i)
            # where a is the kernel filter size (ksize)
            #
            # Lanczos window
            # L(x) = a*sin(pi*x)*sin(pi*x/a)/(pi^2 * x^2)  -a<=x<=a and x!=0
            #        1                                     x=0
            #        0                                     otherwise

            # ksize is the kernel filter size ("a" above), in units of Nyquist
            # units of Nyquist is res/2
            xkernel = (x1-xout[i])
            xkernel /= (resout[i]/2.0)  # in units of Nyquist
            u1 = np.pi*xkernel
            u2 = np.pi*xkernel/ksize
            denom = np.pi**2*xkernel**2
            denom[xkernel==0] = 1
            kernel = ksize*np.sin(u1)*np.sin(u2)/denom
            kernel[xkernel==0] = 1

        # Calculate the values
        # should we ignore bad values?
        yout[i] = np.sum(kernel*y1)
        yerrout[i] = np.sqrt(np.sum(kernel**2*yerr1**2))
        scalesout[i] = np.sum(kernel*scales1)
        
    return yout,yerrout,scalesout

def rescale(yout,yerrout,scalesout):
    """
    Rescale the output y/yerr arrays.

    Parameters
    ----------
    yout : numpy array
       Resampled y array.
    yerrout : numpy array
       Resampled yerr array.
    scalesout : numpy array
       Resampled scales array.

    Returns
    -------
    yout : numpy array
       Rescaled and resampled y array.
    yerrout : numpy array
       Rescaled and resampled yerr array.

    Examples
    --------

    yout,yerrout = rescale(yout,yerrout,scalesout)

    """
    
    yout,yerrout = rescale(yout,yerrout,scalesout)

    return yout,yerrout

def resample(x,y,xout,yerr=None,res=None,ksize=3,scale=True):
    """
    Resample spectrum or list of spectra onto a new wavelength/pixel scale.

    Parameters
    ---------- 
    x : numpy array or list
       Input X-values.  This can be a list of arrays, i.e. multiple
         spectra.
    y : numpy array or list
       Input y-values.  This can be a list of arrays, i.e. multiple
         spectra.
    xout : numpy array
       Output x-values.
    yerr : numpy array or list, optional
       Uncertainties in "y".  Default is None.
    res : numpy array or list, optional
       Resolution width (FWHM) of data.  Default is res=2 (Nyquist).
    kind : str, optional
       Type of resampling.  Either 'sinc' (dampled sinc) or 'lanczos'.
          Default is 'sinc'.
    ksize : int, optional
       Sinc/Lanczos filter size. Generally values between 3 and 7.
          Default is 3.
    scale : bool, optional
       Scale the y values before combining and then rescale at the end.
         Default is True.

    Returns
    -------
    yout : numpy array
       Output resampled y array.
    yerrout : numpy array
       Output resampled uncertainty array.

    Examples
    --------

    yout,yerrout = resample(x,y,xout,yerr=yerr,scale=True)

    """

    if kind not in ['sinc','lanczos']:
        raise ValueError('kind '+str(kind)+' not supported.  Only sinc or lanczos')
    
    # Standardize data
    x,y,yerr,res = standardize(x,y,yerr,res)
    nspectra = len(x)
    
    # Get the scales
    if scale:
        x,y,yerr,scales = getscale(x,y,yerr)
    else:
        scales = np.ones(nspectra,float)
        
    # Merge arrays
    mx,my,myerr,mres,mscales = mergearrays(x,y,yerr,res,scales)
    
    # Find the low/high input index values for each output value
    indexes,resout = findindex(mx,mres,ksize,xout)
    
    # Perform the resampling
    yout,yerrout,scalesout = doresample(mx,my,myerr,mres,mscales,
                                        indexes,resout,kind,ksize)

    # Rescale
    if scale:
        yout,yerrout = rescale(yout,yerrout,scalesout)

    return yout,yerrout
