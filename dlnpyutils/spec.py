#!/usr/env python

# Imports
import numpy as np
from scipy.special import erf,wofz
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks,argrelextrema
from astropy.io import fits
from scipy import ndimage
from scipy.interpolate import interp1d
from . import utils,robust


def trace(im,yestimate=None,yorder=2,sigorder=2,step=50):
    """ Trace the spectrum.  Spectral dimension is assumed to be on the horizontal axis."""
    ny,nx = im.shape
    y = np.arange(ny)
    if yestimate is None:
        ymed = np.nanmedian(im,axis=1)
        #yestimate = np.argmax(ytot)
        sig = utils.mad(ymed)
        ypeaks, = argrelextrema(ymed, np.greater)
        gd, = np.where(ymed[ypeaks] > (np.nanmedian(ymed)+5*sig))
        ypeaks = ypeaks[gd]
        #ypeaks,ptab = find_peaks(ymed,height=10*sig,width=1,prominence=10*sig,wlen=100,distance=10)
        print(len(ypeaks),' peaks found')
        
    # Smooth in spectral dimension
    # a uniform (boxcar) filter with a width of 50
    smim = ndimage.uniform_filter1d(im, 50, 1)
    nstep = nx//step
    out = np.zeros(len(ypeaks),dtype=np.dtype([('peak',int),('yestimate',int),('tpars',(float,yorder+1)),
                                               ('sigpars',(float,sigorder+1))]))
    mcat = []
    # Loop over the traces
    for t in range(len(ypeaks)):
        yestimate = ypeaks[t]
        # Loop over the columns in steps and fit Gaussians
        tcat = np.zeros(nstep,dtype=np.dtype([('x',float),('pars',float,4),('status',int)]))
        for i in range(nstep):
            try:
                ht = np.maximum(im[yestimate,step*i+step//2],100)
                estimate = [ht,yestimate,2.0,0.0]
                bounds = [np.zeros(4,float)-np.inf,np.zeros(4,float)+np.inf]
                bounds[0][0] = 0
                bounds[0][1] = yestimate-15
                bounds[1][1] = yestimate+15                
                bounds[0][2] = 0.1
                pars,cov = utils.gaussfit(y[yestimate-15:yestimate+15],im[yestimate-15:yestimate+15,step*i+step//2],
                                          initpar=estimate,bounds=bounds,binned=True)
                tcat['x'][i] = step*i+step//2
                tcat['pars'][i] = pars
                tcat['status'][i] = 1
            except:
                pass
        # Fit polynomail to y vs. x and gaussian sigma vs. x
        gd, = np.where(tcat['status']==1)
        ypars = np.polyfit(tcat['x'][gd],tcat['pars'][gd,1],yorder)
        sigpars = np.polyfit(tcat['x'][gd],tcat['pars'][gd,2],sigorder)
        print(t+1,yestimate,pars,sigpars)
        # Model
        mcat1 = np.zeros(nx,dtype=np.dtype([('x',float),('y',float),('sigma',float)]))
        xx = np.arange(nx)
        mcat1['x'] = xx
        mcat1['y'] = np.poly1d(ypars)(xx)
        mcat1['sigma'] = np.poly1d(sigpars)(xx)
        mcat.append(mcat1)
        # output
        out['peak'][t] = t
        out['yestimate'][t] = yestimate
        out['tpars'][t] = ypars
        out['sigpars'][t] = sigpars      
        
    return out,mcat,tcat

def boxcar(im,ytrace=None,imerr=None,off=20,backoff=50):
    """ Boxcar extract the spectrum"""
    ny,nx = im.shape
    if ytrace is None:
        ytot = np.sum(im,axis=1)
        yest = np.argmax(ytot)
        ytrace = np.arange(nx)+yest
    else:
        yest = np.median(ytrace)
    # Background subtract
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    med = np.median(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    # Sum up the flux
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    flux = np.sum(subim*mask,axis=0)
    if imerr is not None:
        # add uncertainties in quadrature
        fluxerr = np.sqrt(np.sum(imerr[yblo:ybhi,:]**2*mask,axis=0))
        return flux,fluxerr
    return flux

def linefit(x,y,initpar,bounds,err=None,binned=False):
    # Fit Gaussian profile to data with center and sigma fixed.
    # initpar = [height, center, sigma, constant offset]
    cen = initpar[1]
    sigma = initpar[2]
    #def gline(x, amp, const=0):
    #    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    #    return amp * np.exp(-(x-cen)**2 / (2*sigma**2)) + const
    #line_initpar = [initpar[0],initpar[3]]
    #lbounds, ubounds = bounds
    #line_bounds = ([lbounds[0],lbounds[3]],[ubounds[0],ubounds[3]])
    #return curve_fit(gline, x, y, p0=line_initpar, bounds=line_bounds, sigma=err)

    func = utils.gaussian
    if binned is True: func=utils.gaussbin
    line_initpar = [initpar[0],initpar[3]]
    line_bounds = ([lbounds[0],lbounds[3]],[ubounds[0],ubounds[3]])
    return curve_fit(func, x, y, p0=line_initpar, sigma=sigma, bounds=line_bounds)
    
    return curve_fit(gline, x, y, p0=line_initpar, bounds=line_bounds, sigma=err)

def extract(im,imerr=None,mcat=None,fixtrace=False,fixsigma=False,nobackground=False,
            verbose=False,off=10):
    """ Extract a spectrum"""
    ny,nx = im.shape
    x = np.arange(nx)
    y = np.arange(ny)
    # No trace information input, get it
    if mcat is None:
        tcat,ypars,sigpars,mcat=trace(im)
    # Loop over the columns and get the flux using the trace information
    tab = np.zeros(nx,dtype=np.dtype([('x',int),('pars',float,4),('perr',float,4),
                                      ('flux',float),('fluxerr',float),('chisq',float),('status',int)]))
    #lastpars = None
    for i in range(nx):
        line = im[:,i].copy()
        if imerr is not None:
            lineerr = imerr[:,i].copy()
        else:
            lineerr = np.ones(len(line))   # unweighted        
        bad = (~np.isfinite(line)) | (lineerr==0)
        if np.sum(bad)>0:
            med = np.nanmedian(line[~bad])
            if ~np.isfinite(med): med=0
            line[bad] = med
            lineerr[bad] = 1e30
        # Fit the constant offset and the height of the Gaussian
        #  fix the central position and sigma
        ycen = mcat['y'][i]
        ysigma = mcat['sigma'][i]
        ht0 = np.maximum(line[int(np.round(ycen))],0.01)
        initpars = [ht0,ycen,ysigma,np.median(line)]
        #if lastpars is not None:
        #    initpars = lastpars
        if nobackground is True:
            initpars[3] = 0
        # Only fit the region right around the peak
        ylo = int(np.maximum(ycen-off,0))
        yhi = int(np.minimum(ycen+off,ny))
        y1 = y[ylo:yhi]
        line1 = line[ylo:yhi]
        err1 = lineerr[ylo:yhi]
        #bnds = ([0,ycen-1e-4,ysigma-1e-7,np.minimum(0,initpar[3]-1)],
        #        [1.5*ht0,ycen,ysigma+1e-7,np.maximum(1.5*initpar[3],initpar[3]+1)])
        bnds = (np.zeros(4)-np.inf,np.zeros(4)+np.inf)
        bnds[0][0] = 0   # height
        bnds[0][1] = initpars[1]-2
        bnds[1][1] = initpars[1]+2        
        if fixtrace:
            bnds[0][1] = initpars[1]-1e-7  # fix the trace position
            bnds[1][1] = initpars[1]+1e-7
        bnds[0][2] = 0     # positive sigma
        if fixsigma:
            bnds[0][2] = initpars[2]-1e-7
            bnds[1][2] = initpars[2]+1e-7
        # background
        bnds[0][3] = np.minimum(0,initpars[3]-1)
        bnds[1][3] =  np.maximum(1.5*initpars[3],initpars[3]+1)
        if nobackground is True:
            bnds[0][3] = initpars[3]-1e-7
            bnds[1][3] = initpars[3]+1e-7
            #bnds = ([0,ycen-1e-7,ysigma-1e-7,0],[1.5*ht0,ycen+1e-7,ysigma+1e-7,0.1])

        #func = utils.gaussian
        #if binned is True: func=utils.gaussbin
        func = utils.gaussbin
        try:
            pars,cov = curve_fit(func,y1,line1,p0=initpars,bounds=bnds,sigma=err1)
            # reject outlier points and refit
            perr = np.sqrt(np.diag(cov))
            yfit = func(y1,*pars)
            diff = y1-yfit
            bd, = np.where(np.abs(diff) > 3*err1)
            if len(bad)>0:
                pars1 = pars
                line1[bd] = yfit[bd]
                err1[bd] = 1e30
                pars,cov = curve_fit(func,y1,line1,p0=pars1,bounds=bnds,sigma=err1)                   
                # reject outlier points and refit
                perr = np.sqrt(np.diag(cov))
                yfit = func(y1,*pars)                   
            #pars,cov = linefit(y[y0:y1],line[y0:y1],initpar=initpar,bounds=bnds,err=lineerr[y0:y1])
            flux = np.sum(yfit)
            # Gaussian area = ht*wid*sqrt(2*pi)
            #flux = pars[0]*ysigma*np.sqrt(2*np.pi)
            #fluxerr = perr[0]*ysigma*np.sqrt(2*np.pi)
            # propagation of errors
            fluxerr = flux*np.sqrt( (perr[0]/pars[0])**2 + (perr[2]/pars[2])**2 )
            chisq = np.sum((line1-yfit)/err1)
            if verbose:
                print(i+1,pars)
            tab['x'][i] = i
            tab['pars'][i] = pars
            tab['perr'][i] = perr
            tab['flux'][i] = flux
            tab['fluxerr'][i] = fluxerr
            tab['chisq'][i] = chisq
            tab['status'][i] = 1
            #lastpars = pars
        except:
            if verbose:
                print(i+1,' problem fitting')
    return tab

def extract_optimal(im,ytrace,imerr=None,verbose=False,off=10,backoff=50,smlen=31):
    """ Extract a spectrum using optimal extraction (Horne 1986)"""
    ny,nx = im.shape
    yest = np.median(ytrace)
    # Get the subo,age
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.median(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    suberr = imerr[yblo:ybhi,:]
    # Make sure the arrays are float64
    subim = subim.astype(float)
    suberr = suberr.astype(float)    
    # Mask other parts of the image
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    sim = subim*mask
    serr = suberr*mask
    badpix = (serr <= 0)
    serr[badpix] = 1e20
    # Compute the profile/probability matrix from the image
    tot = np.sum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = utils.medfilt(psf1[i,:],smlen)
        #psf[i,:] = utils.gsmooth(psf1[i,:],smlen)        
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.sum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    # Compute the weights
    wt = psf**2/serr**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.sum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1
    # Compute the flux and flux error
    flux = np.sum(psf*sim/serr**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)    
    fluxerr[badcol] = 1e30  # bad columns
    # Recompute the trace
    trace = np.sum(psf*yy,axis=0)+yblo
    
    # Check for outliers
    diff = (sim-flux*psf)/serr**2
    bad = (diff > 25)
    if np.sum(bad)>0:
        # Mask bad pixels
        sim[bad] = 0
        serr[bad] = 1e20
        # Recompute the flux
        wt = psf**2/serr**2
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1        
        flux = np.sum(psf*sim/serr**2,axis=0)/totwt
        fluxerr = np.sqrt(1/totwt)
        fluxerr[badcol] = 1e30  # bad columns
        # Recompute the trace
        trace = np.sum(psf*yy,axis=0)+yblo
        
    return flux,fluxerr,trace
        
def emissionlines(spec,thresh=None):
    """Measure the emission lines in an arc lamp spectrum. """
    nx = len(spec)
    x = np.arange(nx)
    
    # Threshold
    if thresh is None:
        thresh = np.min(spec) + (np.max(spec)-np.min(spec))*0.05
    
    # Detect the peaks
    sleft = np.hstack((0,spec[0:-1]))
    sright = np.hstack((spec[1:],0))
    peaks, = np.where((spec>sleft) & (spec>sright) & (spec>thresh))
    npeaks = len(peaks)
    print(str(npeaks)+' peaks found')
    
    # Loop over the peaks and fit them with Gaussians
    gcat = np.zeros(npeaks,dtype=np.dtype([('x0',int),('x',float),('xerr',float),('pars',float,4),('perr',float,4),
                                           ('flux',float),('fluxerr',float)]))
    resid = spec.copy()
    gmodel = np.zeros(nx)
    for i in range(npeaks):
        x0 = peaks[i]
        xlo = np.maximum(x0-6,0)
        xhi = np.minimum(x0+6,nx)
        initpar = [spec[x0],x0,1,0]
        bnds = ([0,x0-3,0.1,0],[1.5*initpar[0],x0+3,10,1e4])
        pars,cov = utils.gaussfit(x[xlo:xhi],spec[xlo:xhi],initpar,bounds=bnds,binned=True)
        perr = np.sqrt(np.diag(cov))
        gmodel1 = utils.gaussian(x[xlo:xhi],*pars)
        gmodel[xlo:xhi] += (gmodel1-pars[3])
        resid[xlo:xhi] -= (gmodel1-pars[3])
        # Gaussian area = ht*wid*sqrt(2*pi)
        flux = pars[0]*pars[2]*np.sqrt(2*np.pi)
        fluxerr = perr[0]*pars[2]*np.sqrt(2*np.pi)
        gcat['x0'][i] = x0
        gcat['x'][i] = pars[1]
        gcat['xerr'][i] = perr[1]
        gcat['pars'][i] = pars
        gcat['perr'][i] = perr
        gcat['flux'][i] = flux
        gcat['fluxerr'][i] = fluxerr
        
    return gcat, gmodel


def continuum(spec,bin=50,perc=60,norder=4):
    """ Derive the continuum of a spectrum."""
    nx = len(spec)
    x = np.arange(nx)
    # Loop over bins and find the maximum
    nbins = nx//bin
    xbin1 = np.zeros(nbins,float)
    ybin1 = np.zeros(nbins,float)
    for i in range(nbins):
        xbin1[i] = np.nanmean(x[i*bin:i*bin+bin])
        ybin1[i] = np.nanpercentile(spec[i*bin:i*bin+bin],perc)
    # Fit polynomial to the binned values
    coef1 = robust.polyfit(xbin1,ybin1,norder)
    cont1 = np.poly1d(coef1)(x)
    
    # Now remove large negative outliers and refit
    gdmask = np.zeros(nx,bool)
    gdmask[(spec/cont1)>0.8] = True
    xbin = np.zeros(nbins,float)
    ybin = np.zeros(nbins,float)
    for i in range(nbins):
        xbin[i] = np.nanmean(x[i*bin:i*bin+bin][gdmask[i*bin:i*bin+bin]])
        ybin[i] = np.nanpercentile(spec[i*bin:i*bin+bin][gdmask[i*bin:i*bin+bin]],perc)
    # Fit polynomial to the binned values
    coef = robust.polyfit(xbin,ybin,norder)
    cont = np.poly1d(coef)(x)
    
    return cont,coef
