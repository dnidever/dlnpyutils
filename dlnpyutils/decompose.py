import numpy as np

# Decomposing data using a basis function

def standardize(x,y,yerr,res):
    """
    Standardize data.  Puts inputs into a "standard" format of a list
    of numpy arrays.

    Parameters
    ----------
    x : numpy array or list
       Input X-values.  This can be a list of arrays, i.e. multiple
         datasets.
    y : numpy array or list
       Input Y-values.  This can be a list of arrays, i.e. multiple
         sets of data..
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
    Find the scale of the data so it is easier to average them.

    Parameters
    ----------
    x : list of numpy arrays
       Input X-values.  This can be a list of arrays, i.e. multiple
         data.
    y : list of numpy arrays
       Input y-values.  This can be a list of arrays, i.e. multiple
         data.
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

    # Loop over the datasets
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
    Merge the arrays of multiple datasets into a single sorted array.

    Parameters
    ----------
    x : list of numpy arrays
       Input X-values.  This can be a list of arrays, i.e. multiple
         datasets.
    y : list of numpy arrays
       Input y-values.  This can be a list of arrays, i.e. multiple
         datasets.
    yerr : list of numpy arrays
       Uncertainties in "y".  This can be None.
    res : list of numpy arrays
       Resolution width (FWHM).  This can be None.
    scales : list of scalars/numpy arrays
       The values used to rescale the datasets.

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
    # Loop over the datasets and merge the arrays
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

    Examples
    --------

    indexes = findindex(mx,mres,ksize,xout)

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
    for i in range(len(xout)):
        xi = xout[i]
        
        # Find closest input x value to this xout value
        while (mx[closeind]<=x1):
            closeind += 1
        if closeind>0 and abs(mx[closeind-1]-xi)<abs(mx[closeind]-xi):
            closeind -= 1
        
        # Limit to use for upper/lower region of kernel (in units of x)
        # res is FWHM or 2 for Nyquist
        lim = ksize*mres[closeind]/2.0
        
        # -- Get Low index --
        low = closeind
        while (xi-mx[low] < lim):
            low -= 1
        if xi-mx[low] > lim:  # went over
            low += 1
        indexes[i,0] = low
        
        # -- Get high index --
        high = closeind
        while (mx[high]-xi < lim):
            high += 1
        if mx[high]-xi > lim:  # went over
            high -= 1
        indexes[i,1] = high

    return indexes

def fourier_decompose(x,y,yerr,n=None):
    """
    Fourier decompose data.

    Parameters
    ----------
    x : list or numpy array
       List or numpy array of X-values.  This can be a set of different
         data/spectra, etc.
    y : list or numpy array
       List or numpy array of Y-values.  This can be a set of different
         data/spectra, etc.
    yerr : numpy array or list
       Uncertainties in Y.  This can be None.

    Returns
    -------
    coef : numpy array
       Fourier basis coefficients.
    coeferr : numpy array
       Uncertainty in basis coefficients.

    Examples
    --------

    coef,coeferr = fourier_decompose(x,y)

    """

    # f(t) = a_0 + Sum_(k=1)^(N) a_k * cos(2*pi*k*t/P) + b_k * sin(2*pi*k*t/P) )
    # where P is the period or span of the data
    p = np.max(x)-np.min(x)
    a0 = np.mean(y)
    acoef = np.zeros(n,float)
    bcoef = np.zeros(n,float)
    k = np.arange(n)+1
    for i in range(n):
        acoef[i] = (1/p)*np.sum(y*np.cos(2*np.pi*k/p))
        bcoef[i] = (1/p)*np.sum(y*np.sin(2*np.pi*k/p))
    coef = np.concatenate((np.array([a0]),acoef,bcoef))
    return coef

def bspline_decompose(x,y):
    """
    B-spline decompose data.

    Parameters
    ----------
    x : list or numpy array
       List or numpy array of X-values.  This can be a set of different
         data/spectra, etc.
    y : list or numpy array
       List or numpy array of Y-values.  This can be a set of different
         data/spectra, etc.
    yerr : numpy array or list
       Uncertainties in Y.  This can be None.

    Returns
    -------
    coef : numpy array
       Basis coefficients.
    coeferr : numpy array
       Uncertainty in basis coefficients.

    Examples
    --------

    coef,coeferr = bspline_decompose(x,y)

    """

    pass

    return coef,coeferr

def pca_decompose(x,y):
    """
    Decomposing driver function.

    Parameters
    ----------
    x : list or numpy array
       List or numpy array of X-values.  This can be a set of different
         data/spectra, etc.
    y : list or numpy array
       List or numpy array of Y-values.  This can be a set of different
         data/spectra, etc.
    yerr : numpy array or list
       Uncertainties in Y.  This can be None.

    Returns
    -------
    coef : numpy array
       Basis coefficients.
    basis : numpy array
       Basis vectors.

    Examples
    --------

    coef,basis = pca_decompose(x,y)

    """

    pass


def sphereharmonics_decompose(x,y):
    """
    Decomposing driver function.

    Parameters
    ----------
    x : list or numpy array
       List or numpy array of X-values.  This can be a set of different
         data/spectra, etc.
    y : list or numpy array
       List or numpy array of Y-values.  This can be a set of different
         data/spectra, etc.
    yerr : numpy array or list
       Uncertainties in Y.  This can be None.

    Returns
    -------
    coef : numpy array
       Spherical harmonics Basis coefficients.

    Examples
    --------

    coef = sphereharmonics_decompose(x,y)

    """

    pass

def decompose(x,y,yerr=None,res=None,kind=None):
    """
    Decomposing driver function.

    Parameters
    ----------
    x : list or numpy array
       List or numpy array of X-values.  This can be a set of different
         data/spectra, etc.
    y : list or numpy array
       List or numpy array of Y-values.  This can be a set of different
         data/spectra, etc.
    yerr : numpy array or list
       Uncertainties in Y.  This can be None.
    res : numpy array or list:
       Resolution width (FWHM).  This can be None.
    kind : str
       Type of decomposition basis to use: 'fourier', 'bspline', 'pca',
         'sphereharmonics'.

    Returns
    -------
    coef : numpy array
       Basis coefficients.

    Examples
    --------

    coef = decompose(x,y,'fourier')

    """

    # Standardize data
    x,y,yerr,res = standardize(x,y,yerr,res)
    ndata = len(x)
    # Rescale
    if scale:
        x,y,yerr,scales = getscale(x,y,yerr)
    else:
        scales = np.ones(ndata,float)
    # Merge data
    mx,my,myerr,mres,mscales = mergearrays(x,y,yerr,res,scales)
    # Get indexes?

    if kind=='fourier':
        coef,coeferr = fourier_decompose(mx,my,myerr)
        coef,coeferr = rescale(coef,coeferr)
        out = (coef,coeferr)
    elif kind=='bspline':
        coef = bspline_decompose(mx,my,myerr)
        coef = rescale(coef)
        out = coef
    elif kind=='pca':
        coef,basis = pca_decompose(mx,my,myerr)
        coef = rescale(coef)
        out = coef
    elif kind=='sphereharmonics':
        coef = sphereharmonics_decompose(mx,my,myerr)
        coef = rescale(coef)
        out = coef
    else:
        raise ValueError('kind '+str(kind)+' not supported')
    return out
