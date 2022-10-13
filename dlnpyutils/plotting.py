#!/usr/bin/env python
#
# PLOTTING.PY - plotting functions
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200209'  # yyyymmdd

import time
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyWarning
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backend_bases import MouseButton
from scipy import stats
import copy
from . import utils as dln
from . import ladfit

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def zscaling(im,contrast=0.25,nsample=50000):
    """
    This finds the IRAF display z1,z2 scalings
    for an image
    
    Parameters:
    im         2D image
    =contrast  Contrast to use (contrast=0.25 by default)
    =nsample   The number of points of the image to use for the sample
                (nsample=5e4 by default).

    Returns:
     [z1,z1]         The minimum and maximum value to use for display scaling.

    Examples:
      zcals = zscale(im)


    From IRAF display help

    If  the  contrast  is  not  zero  the  sample  pixels  are ranked in
    brightness to form the function I(i) where i  is  the  rank  of  the
    pixel  and  I is its value.  Generally the midpoint of this function
    (the median) is very near the peak of the image histogram and  there
    is  a  well defined slope about the midpoint which is related to the
    width of the histogram.  At the ends of the I(i) function there  are
    a  few very bright and dark pixels due to objects and defects in the
    field.  To determine  the  slope  a  linear  function  is  fit  with
    iterative rejection;
    
            I(i) = intercept + slope * (i - midpoint)
    
    If  more  than half of the points are rejected then there is no well
    defined slope and the full range of the sample defines  z1  and  z2.
    Otherwise  the  endpoints  of the linear function are used (provided
    they are within the original range of the sample):
    
            z1 = I(midpoint) + (slope / contrast) * (1 - midpoint)
            z2 = I(midpoint) + (slope / contrast) * (npoints - midpoint)

    The actual IRAF program is called zsc_zlimits and is in the file
    /net/astro/iraf2.12/iraf/pkg/images/tv/display/zscale.x


    By D.Nidever  Oct 2007  (using IRAF display zscale algorithm) 
    """

    if im.ndim != 2:
        raise ValueError('The input must be 2 dimensiona')
    nx,ny = im.shape
    n = nx*ny

    nsample = np.minimum(nsample, n)

    xind = np.round(np.random.random(nsample)*(nx-1)).astype(int)
    yind = np.round(np.random.random(nsample)*(ny-1)).astype(int)    

    f = im[xind,yind]
    si = np.argsort(f)
    f2 = f[si]
    x = np.arange(nsample)
    midpoint = np.round(nsample*0.5)
    zmin = np.min(f)
    zmax = np.max(f)
    zmed = np.median(f)

    #zmin = np.min(im)
    #zmax = np.max(im)
    #zmed = np.median(im)

    # Robust fitting program
    coef,dum = ladfit.ladfit(x,f2)
    coef = coef[::-1]
    #coef = dln.poly_fit(x,f2,1,robust=True)
    
    # y = m*x + b
    # I = intercept + slope * (i-midpoint)
    # I = intercept + slope * i - slope*midpoint
    # I = slope * i + (intercept - slope*midpoint)
    # b = intercept - slope*midpoint
    # intercept = b + slope*midpoint
    
    slope = coef[0]
    intercept = coef[1] + slope*midpoint

    #    z1 = I(midpoint) + (slope / contrast) * (1 - midpoint)
    #    z2 = I(midpoint) + (slope / contrast) * (npoints - midpoint)
    #z1 = f2[midpoint] + (slope/contrast) * (1L - midpoint)
    #z2 = f2[midpoint] + (slope/contrast) * (nsample - midpoint)
    z1 = zmed + (slope/contrast) * (1 - midpoint)
    z2 = zmed + (slope/contrast) * (nsample - midpoint)

    z1 = np.maximum(z1, zmin)
    z2 = np.minimum(z2, zmax)
    
    return [z1,z2]

    

def hist2d(x,y,z=None,statistic=None,xr=None,yr=None,dx=None,dy=None,nx=200,ny=200,
           zscale=None,log=None,norm=None,noerase=False,vmin=None,vmax=None,center=True,xflip=False,
           yflip=False,force=True,cmap=None,figure=None,figsize=(8,8),xtitle=None,ytitle=None,title=None,
           colorlabel=None,charsize=12,origin='lower',aspect='auto',interpolation='none',bright=None,
           minhue=0.0,maxhue=0.7,minbright=0.1,maxbright=0.7,saturation=0.9,noplot=False,save=None):
    """
    Make a 2D histogram of points.
    
    Parameters
    ----------
    x : numpy array
       Array of x-data points to plot.
    y : numpy array
       Array of y-data points to plot.
    z : numpy array, optional
       Array of z-data points to plot if we using a statistic.
    statistic : str, optional
       The statistic to use.  Default is 'count' if z is not input otherwise 'sum'.
         The options are: 'sum', 'mean', 'median', 'std', 'mad'.
    xr : list, optional
       The range of x-values to plot.  The default is [min(x),max(x)]
    yr : list, optional
       The range of y-values to plot.  The default is [min(x),max(x)]
    dx : float, optional
       The step size in the x-dimension.  Either dx+dy or nx+ny must be specified.
    dy : float, optional
       The step size in the y-dimension.
    nx : int, optional
       Number of bins in the x-dimension. Either nx+ny or dx+dy must be specified.
          Default is 200.
    ny : int, optional
       Number of bins in the z-dimension.  Default is 200.
    zscale : bool, optional
       Automatic min and max interval of statistic based on IRAF's zscale.
    log : bool, optional
       Use logarithmic scaling of the statistic.  Default is linear.
    norm : normalize object, optional
        The `.Normalize` instance used to scale scalar data to the [0, 1]
        range before mapping to colors using *cmap*. By default, a linear
        scaling mapping the lowest value to 0 and the highest to 1 is used.
        This parameter is ignored for RGB(A) data.
    noerase : bool, optional
       Do not erase or clear the figure window.  Default is to clear the figure.
    vmin : float, optional
       Minimum value to plot on colorbar.  Default is the minimum of the statistic.
    vmax : float, optional
       Maximum value to plot on colorbar.  Default is the maximum of the statistic.    
    force : bool, optional
       Force the xrange and yrange to be exactly as the way they are set (default).
    center : bool, optional
       The x/y values of a bin should correspond to the center.  By default they
         correspond to the bottom-left corner.
    xflip : bool, optional
       Flip the X-coordinate axis.  Default is False.
    yflip : bool, optional
       Flip the Y-coordinate axis.  Default is False.
    cmap : str, optional
       The matplotlib color map to use.  Default is 'viridis'.
    xtitle : str, optional
       The label for the x-axis.  Default is 'X'.
    ytitle : str, optional
       The label for the y-axis.  Default is 'Y'.
    title : str, optional
       The plot title.  Default is statistic+'(Z)'.
    figure : int, optional
      The figure window number.  Default is to use the current window.
    figsize : list, optional
       Two-element list giving figure size in X and Y.  Default is (8,8).
    colorlabel : str, optional
       The colorbar label.  Default is statistic+'(Z)'.
    charsize : int, optional
       Character size.  Default is 12.
    origin : str, optional
       The origin of the plot.  Default is 'lower'.
    aspect : str, optional
       The aspect ratio of the plot.  Default is 'auto'.
    interpolation : str, optional
       Type of interpolation in the image.  Default is 'none'.
    noplot : boolean, optional
       Do not plot anything.  Just return the data.
    save : str, optional
       Save the figure to this file.

    Returns
    -------

    Figure is plotted to the screen.

    Example
    -------

    hist2d(x,y,z,'mean')

    """

    # Getting the current figure, creating a new one if necessary
    if figure is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(figure,figsize=figsize)

    # Character size
    font = {'size': charsize}
    matplotlib.rc('font', **font)

    if noerase is False:
        plt.clf()   # clear the plot    
    
    # Statistic default
    if statistic is None:
        if z is None:
            statistic = 'count'
        else:
            statistic = 'sum'
        
    # Input dx/dy
    if dx is None:
        dx0 = None
    else:
        dx0 = copy.copy(dx)
    if dy is None:
        dy0 = None
    else:
        dy0 = copy.copy(dy)
        
    # Min and Max's
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)    

    # Temporary DX and DY if not input
    # X-axis
    if (dx is None):
        # XRANGE set
        if xr is not None:
            dx = (xr[1]-x[0])/(nx-1)
        # XRANGE not set
        else:
            dx = (xmax-xmin)/(nx-1)
    # Y-axis
    if (dy is None):
        # YRANGE set
        if yr is not None:
            dy = (yr[1]-yr[0])/(ny-1)
        # YRANGE not set
        else:
            dy = (ymax-ymin)/(ny-1)

    # Axes reversed
    if xr is not None:
        # X axis reversed
        if xr[1]<xr[0]:
            xr = np.flip(xr)
            xflip = 1
    if yr is not None:
        # Y axis reversed
        if yr[1]<yr[0]:
            yr = np.flip(yr)
            yflip = 1

    # Cutting based on the xrange/yrange if set
    #   Otherwise the image size might get too large if there are
    #   points far outside the xrange/yrange
    # Xrange set
    if xr is not None:
        mask1 = ((x>=(xr[0]-dx)) & (x<=(xr[1]+dx)))
        if np.sum(mask1)>0:
            x = x[mask1]
            y = y[mask1]
            if z is not None:
                z = z[mask1]
        else:
            raise ValueError('No points left')

        # Min and Max's
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        # Setting DX/DY if not input
        if dx0 is None: dx=(xmax-xmin)/(nx-1)
        if dy0 is None: dy=(ymax-ymin)/(ny-1)
    # yrange set
    if yr is not None:
        mask2 = ((y>=(yr[0]-dy)) & (y<=(yr[1]+dy)))
        if np.sum(mask2)>0:
            x = x[mask2]
            y = y[mask2]
            if z is not None:
                z = z[mask2]
        else:
            raise ValueError('No points left')

        # Min and Max's
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        # Setting DX/DY if not input
        if dx0 is None: dx=(xmax-xmin)/(nx-1)
        if dy0 is None: dy=(ymax-ymin)/(ny-1)


    # Step sizes must be positive
    dx = np.abs(dx)
    dy = np.abs(dy)


    # Setting the range
    if xr is not None:
        # Default is that the xrange/yrange are for the centers
        # Move xmin/xmax accordingly
        if center is True:
            off = 0.5*dx
        else:
            off = 0.0

        # xmin and xmax must be xrange+/-integer*dx
        if force is True:
            # If xmin < xrange[0] then move it down an integer number of dx's
            if (xmin < (xr[0]-off)):
                diff = (xr[0]-off)-xmin
                xmin = (xr[0]-off)-np.ceil(diff/dx)*dx
            else:
                xmin = xr[0]-off
            # If xmax > xrange[1] then move it up an integer number of dx's
            if (xmax > (xr[1]+off)):
                diff = xmax-(xr[1]+off)
                xmax = (xr[1]+off)+np.ceil(diff/dx)*dx
            else:
                xmax = xr[1]+off

        # Don't force the xrange
        else:
            if ((x[0]-off) < xmin) | ((xr[1]+off) > xmax):
                xmin = np.minimum((xr[0]-off), xmin)
                xmax = np.maximum((xr[1]+off), xmax)

    # Setting the range
    if yr is not None:
        # Default is that the xrange/yrange are for the centers
        # Move ymin/ymax accordingly
        if center is True:
            off = 0.5*dx
        else:
            off = 0.0

        # xmin and xmax must be xrange+/-integer*dx
        if force is True:
            # If ymin < yrange[0] then move it down an integer number of dy's
            if (ymin < (yr[0]-off)):
                diff = (yr[0]-off)-ymin
                ymin = (yr[0]-off)-np.ceil(diff/dy)*dy
            else:
                ymin = yr[0]-off
            # If ymax > yrange[1] then move it up an integer number of dy's
            if (ymax > (yr[1]+off)):
                diff = ymax-(yr[1]+off)
                ymax = (yr[1]+off)+np.ceil(diff/dy)*dy
            else:
                ymax = yr[1]+off

        # Don't force the xrange
        else:
            if ((yr[0]-off) < ymin) | ((yr[1]+off) > ymax):
                ymin = np.minimum((yr[0]-off), ymin)
                ymax = np.maximum((yr[1]+off), ymax)


    # Setting final DX/DY and NX/NY
    # X-axis
    if dx0 is None:
        dx = (xmax-xmin)/(nx-1.)
    else:
        nx = int(np.floor((xmax-xmin)/dx)+1)  # only want bins fully within the range
    # Y-axis
    if dy0 is None:
        dy = (ymax-ymin)/(ny-1)
    else:
        ny = int(np.floor((ymax-ymin)/dy)+1)  # only want bins fully within the range


    # Final xrange/yrange, if not already set
    if xr is None:
        xr = [xmin,xmax]
    if yr is None:
        yr = [ymin,ymax]
        
    # No z input
    if z is None or statistic=='count':
        im, xedges, yedges = np.histogram2d(x,y,range=[xr,yr],bins=[nx,ny])
        im = im.T  # np.histogram2d() returns [x,y], while python/matplotlib convention is [y,x]
    # Statistic using z-values
    else:
        if statistic=='avg':
            cmap = 'jet'
            ima, xedges, yedges, binnumber = stats.binned_statistic_2d(x,y,z,statistic='mean',range=[xr,yr],bins=[nx,ny])
            ima = ima.T
            if bright is None:
                imt, xedges, yedges, binnumber = stats.binned_statistic_2d(x,y,z,statistic='count',range=[xr,yr],bins=[nx,ny])
                imt = imt.T
                if log:
                    pos = (imt > 0)
                    imt[pos] = np.log10(imt[pos])
                    imt[~pos] = -10
            else:
                imt, xedges, yedges, binnumber = stats.binned_statistic_2d(x,y,bright,statistic='sum',range=[xr,yr],bins=[nx,ny])
                imt = imt.T
                if log:
                    pos = (imt > 0)
                    imt[pos] = np.log10(imt[pos])
                    imt[~pos] = -10
                    
            # convert image to RGB, using HLS
            # hue is Average (IMA)   (0-360)
            # 0-red, 120-green, 240-blue
            # brightness is Total (im) (0-1)
            #ima2 = -ima    ; (blue-green-red)
            # color_convert, hue, bright, im*0.+saturation, r, g, b, /HLS_RGB
            gooda = np.isfinite(ima)
            if vmin is None: vmin = np.min(ima[gooda])
            if vmax is None: vmax = np.max(ima[gooda])            
            #ima = -ima

            # hue goes: red, orange, yellow, green, light blue, dark blue, purple, red
            # want to stop after dark blue, around hue=0.7
            
            # hsv is hue, saturation, value or hue, saturation, brightness
            hueim = np.zeros(ima.shape,float)
            #hueim[gooda] = dln.limit(dln.scale(ima[gooda],[-zmax,-zmin],[minhue,maxhue]),minhue,maxhue)
            hueim[gooda] = dln.limit(dln.scale(ima[gooda],[vmin,vmax],[minhue,maxhue]),minhue,maxhue)
            # flip hue to get blue to red
            hueim = 1-hueim
            goodt = np.isfinite(imt)
            tmin = np.min(imt[goodt])
            tmax = np.max(imt[goodt])            
            brightim = np.zeros(imt.shape,float)
            brightim[goodt] = dln.limit(dln.scale(imt[goodt],[tmin,tmax],[minbright,maxbright]),minbright,maxbright)
            hsvim = np.zeros((*ima.shape,3),float)
            hsvim[:,:,0] = hueim
            hsvim[:,:,1] = saturation
            hsvim[:,:,2] = brightim
            rgbim = colors.hsv_to_rgb(hsvim)
            im = rgbim

            if title is None:
                title = 'mean(Z)'
            if colorlabel is None:
                colorlabel = 'mean(Z)'
            
        # Normal statistic
        else:
            im, xedges, yedges, binnumber = stats.binned_statistic_2d(x,y,z,statistic=statistic,range=[xr,yr],bins=[nx,ny])
            im = im.T

    # Plot the image
    if noplot==False:
        if log is True and statistic != 'avg' and norm is None:
            norm = colors.LogNorm(vmin=vmin,vmax=vmax)
        if zscale is True:
            vmin,vmax = zscaling(im)
        if vmin is None:
            vmin = np.min(im[np.isfinite(im)])
        if vmax is None:
            vmax = np.max(im[np.isfinite(im)])        
        extent = [xr[0], xr[1], yr[0], yr[1]]

        if norm is not None:
            plt.imshow(im,cmap=cmap,norm=norm,aspect=aspect,origin=origin,
                       extent=extent,interpolation=interpolation)
        else:
            plt.imshow(im,cmap=cmap,norm=norm,aspect=aspect,vmin=vmin,vmax=vmax,origin=origin,
                       extent=extent,interpolation=interpolation)
        if xflip: plt.xlim(xr[1],xr[0])
        if yflip: plt.ylim(yr[1],yr[0])
                       
        # Axis titles
        if xtitle is None:
            xtitle = 'X'
        plt.xlabel(xtitle)
        if ytitle is None:
            ytitle = 'Y'
        plt.ylabel(ytitle)        
        if title is None:
            if statistic is 'count':
                title = statistic
            else:
                title = statistic+'(Z)'
        plt.title(title)
        
        # Add the colorbar
        if colorlabel is None:
            if statistic is 'count':
                colorlabel = statistic
            else:
                colorlabel = statistic+'(Z)'
        plt.colorbar(label=colorlabel)

        # Save the figure
        if save is not None:
            plt.savefig(save,bbox_inches='tight')
    
    return xedges, yedges, im


def display(im,x=None,y=None,log=False,xr=None,yr=None,noerase=False,zscale=False,norm=None,
            vmin=None,vmax=None,xtitle=None,ytitle=None,title=None,origin='lower',aspect='auto',
            xflip=False,yflip=False,cmap=None,figure=None,figsize=(8,8),save=None,colorlabel=None,
            charsize=12,interpolation=None):
    """
    Display an image.  The usual python convention is used where the image is [NY,NX]

    Parameters
    ----------
    im : 2D numpy array
      2D numpy array image to display.
    x : 1D numpy array,  optional
      1D array of x-values.  Default is to use np.arange(nx).
    y : 1D numpy array, optional
      1D array of y-values.  Default is to use np.arange(ny).
    log : boolean, optional
      Use logarithmic scaling.  Default is to use linear scaling.
    xr : list, optional
      X-axis range to use.  Default is to show the full image.
    yr : list, optional
      Y-axis range to use.  Default is to show the full image.
    noerase : boolean, optional
      Do not erase the plot window.  Default is to erase.
    zscale : boolean, optional
      Set image min/max values using the zscale algorithm.
    norm : normalize object, optional
        The `.Normalize` instance used to scale scalar data to the [0, 1]
        range before mapping to colors using *cmap*. By default, a linear
        scaling mapping the lowest value to 0 and the highest to 1 is used.
        This parameter is ignored for RGB(A) data.
    vmin : float, optional
      Minimum image value to plot.  Default is minimum of the entire image.
    vmax : float, optional
      Maximum image value to plot.  Default is maximum of the entire image.
    xtitle : str, optional
      X-axis title  Default is no title.
    ytitle : str, optional
      Y-axis title  Default is no title.
    title : str, optional
      The plot title.  Default is no title.
    origin : str, optional
      The origin.  Default is "lower".
    aspect : str, optional
      The aspect of the plot window.  Default is "auto".
    xflip : bool, optional
       Flip the X-coordinate axis.  Default is False.
    yflip : bool, optional
       Flip the Y-coordinate axis.  Default is False.    
    cmap : str, optional
      The color map.  Default is "viridis".
    figure : int, optional
      The figure window number.  Default is to use the current window.
    figsize : list, optional
       Two-element list giving figure size in X and Y.  Default is (8,8).
    save : str, optional
       Save the figure to this file.
    colorlabel : str, optional
       The colorbar label.  Default is statistic+'(Z)'.
    charsize : int, optional
       Character size.  Default is 12.
    interpolation : str, optional
       The type of interpolation.

    Returns
    -------
    A plot is made in the figure window.

    Example
    -------
    
    display(im,vmin=100,vmax=5000,xr=[100,300])


    """

    # Getting the current figure, creating a new one if necessary
    if figure is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(figure,figsize=figsize)

    # Character size
    font = {'size': charsize}
    matplotlib.rc('font', **font)
        
    if noerase is False:
        plt.clf()   # clear the plot
        
    ny,nx = im.shape        
        
    # No X/Y inputs
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)

    # Min's and Max's
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # Trim based on xrange/yrange
    if xr is not None:
        val0,xlo = dln.closest(x,xr[0])
        val1,xhi = dln.closest(x,xr[1])        
        im = im[:,xlo:xhi+1]
        x = x[xlo:xhi+1]
        nx,ny = im.shape
        xmin = np.min(x)
        xmax = np.max(x)
    if yr is not None:
        val0,ylo = dln.closest(y,yr[0])
        val1,yhi = dln.closest(y,yr[1])        
        im = im[ylo:yhi+1,:]
        y = y[ylo:yhi+1]
        nx,ny = im.shape        
        ymin = np.min(y)
        ymax = np.max(y)
        
    # Plot the image
    #fig, ax = plt.subplots()
    if log is True and norm is None:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    if zscale is True:
        vmin,vmax = zscaling(im)

    # (left, right, bottom, top)
    extent = [xmin, xmax, ymin, ymax]
    if norm is not None:
        plt.imshow(im,cmap=cmap,norm=norm,aspect=aspect,origin=origin,
                   extent=extent,interpolation=interpolation)
    else:
        plt.imshow(im,cmap=cmap,aspect=aspect,vmin=vmin,vmax=vmax,origin=origin,
                   extent=extent,interpolation=interpolation)        
    if xflip:
        if xr is not None:
            plt.xlim(xr[1],xr[0])
        else:
            plt.xlim(xmax,xmin)            
    if yflip:
        if yr is not None:
            plt.ylim(yr[1],yr[0])
        else:
            plt.ylim(ymax,ymin)
            
    # Axis titles
    if xtitle is not None:
        plt.xlabel(xtitle)
    if ytitle is not None:
        plt.ylabel(ytitle)        
    if title is not None:
        plt.title(title)
        
    # Add the colorbar
    if colorlabel is None:
        colorlabel = ''
    plt.colorbar(label=colorlabel)

    # Save the figure
    if save is not None:
        plt.savefig(save,bbox_inches='tight')
    
    return


def plot(x,y=None,c=None,marker=None,fill=True,size=None,log=False,noerase=False,
         vmin=None,vmax=None,linewidth=None,xtitle=None,ytitle=None,title=None,
         xr=None,yr=None,cmap=None,alpha=None,figure=None,figsize=(8,8),xflip=False,
         yflip=False,save=None,colorlabel=None,charsize=12):
    """
    Create a line or scatter plot.  like plotc.pro

    Parameters
    ----------
    x : numpy array
       Array of x-values to plot.
    y : numpy array
       Array of x-values to plot.
    c : numpy array, optional
       Array of values to color-code the points by.
    marker : float/int, optional
       Marker type.
    fill : boolean, optional
       Use filled markers.  Default is fill=True.
    size : float/int, optional
       Marker size.
    log : boolean, optional
       If c input, use a logarithmic scale.
    noerase : boolean, optional
       Do not erase the current plotting figure.  Default is to erase the figure.
    vmin : float, optional
       Minimum color value to plot.
    vmax : float, optional
       Maximum color value to plot.
    linewidth : float, optional
       Linewidth to use.
    xtitle : str, optional
       X-axis title.
    ytitle : str, optional
       Y-axis title.
    title : str, optional
       Main figure title.
    xr : list, optional
       X-axis minimum and maximum range.
    yr : list, optional
       Y-axis minimum and maximum range.
    cmap : str, optional
       Color map to use.
    alpha : float, optional
       Alpha value to use (between 0 and 1).
    figure : int, optional
       Figure number.  The default is to use the current figure.
    figsize : list, optional
       Two-element list giving figure size in X and Y.  Default is (8,8).
    xflip : boolean, optional
       Flip the x-axis.
    yflip : boolean, optional
       Flip the y-axis.
    save : str, optional
       Save the figure to this file.
    colorlabel : str, optional
       The colorbar label.  Default is statistic+'(Z)'.
    charsize : int, optional
       Character size.  Default is 12.

    Returns
    -------
    Figure is plotted to the screen.

    Example
    -------

    plotting(x,y)

    """

    # xerr, yerr, symbol size

    # No Y-input
    if y is None:
        y = np.array(x).copy()
        x = np.arange(len(y))
    
    # Getting the current figure, creating a new one if necessary
    if figure is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(figure,figsize=figsize)

    # Character size
    font = {'size': charsize}
    matplotlib.rc('font', **font)
        
    if noerase is False:
        plt.clf()   # clear the plot

    if fill==False:
        if c is not None:
            edgecolor = None
        else:
            edgecolor = 'b'
        facecolor = 'none'
    else:
        edgecolor = None
        facecolor = None
        
    # Make the plot
    norm = None
    if log is True:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
        plt.scatter(x,y,c=c,marker=marker,s=size,cmap=cmap,norm=norm,
                    alpha=alpha,edgecolor=edgecolor,facecolor=facecolor,
                    linewidth=linewidth)        
    else:
        plt.scatter(x,y,c=c,marker=marker,s=size,cmap=cmap,norm=norm,
                    edgecolor=edgecolor,facecolor=facecolor,vmin=vmin,
                    vmax=vmax,alpha=alpha,linewidth=linewidth)

    
    # Axis titles
    if xtitle is not None:
        plt.xlabel(xtitle)
    if ytitle is not None:
        plt.ylabel(ytitle)        
    if title is not None:
        plt.title(title)

    # Axis ranges
    if xr is not None:
        plt.xlim(xr)
    if yr is not None:
        plt.ylim(yr)

    # Flip axes
    if xflip:
        if xr is None:
            xr = dln.minmax(x)
            xr = [xr[0]-0.05*dln.valrange(x),xr[1]+0.05*dln.valrange(x)]            
        plt.xlim(np.flip(xr))
    if yflip:
        if yr is None:
            yr = dln.minmax(y)
            yr = [yr[0]-0.05*dln.valrange(y),yr[1]+0.05*dln.valrange(y)]  
        plt.ylim(np.flip(yr))
                         
    # Add the colorbar
    if c is not None and len(c)==len(x):
        if colorlabel is None:
            colorlabel = ''
        plt.colorbar(label=colorlabel)

    # Save the figure
    if save is not None:
        plt.savefig(save,bbox_inches='tight')
        
    return

def oplot(*args,**kwargs):
    plot(*args,**kwargs,noerase=True)

def scatter(*args,**kwargs):
    plot(*args,**kwargs)


class Cursor():

    def __init__(self):
        self.name = 'test'
        self.coords = None
        self.binding_id = None
     
def cursor():
    """
    This returns the position of a cursor click on a matplotlib figure window.

    Parameters
    ----------
    None

    Returns
    -------
    coords : list
      List of data coordinates.

    Example
    -------

    coords = cursor()

    Click in the figure window.
    data coords: 2.981026 5.637763

    print(coords)
    [2.981026, 5.637763]

    Note, the coords variable will be a blank list until the click in the figure window.

    """

    fig = plt.gcf()

    curs.coords = []
    curs.binding_id = None

    
    def on_click(event):
        if event.button is MouseButton.LEFT:
            x, y = event.x, event.y
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                print('data coords: %f %f' % (event.xdata, event.ydata))
                curs.coords += [event.xdata,event.ydata]
            plt.disconnect(curs.binding_id)
            
    binding_id = plt.connect('button_press_event', on_click)
    curs.binding_id = binding_id

    return curs.coords


def curpdiff(spherical=False,arcsec=False):
    """
    This returns the difference of two clicked positions on a figure window.

    Parameters
    ----------
    spherical : boolean, optional
       Use spherical coordinates.  Default is False.
    arcsec : boolean, optional
       User arcseconds for spherical units.  Default is degrees.

    Returns
    -------
    data : dict
      Dictionary of data values

    Example
    -------

    data = curpdiff()

    First Click

    data coords: 2.001687 1.907833
    Second Click
    data coords: 7.058273 7.033132
    Distance = 7.1998 pixels
    Delta X = 5.0566 pixels
    Delta Y = 5.1253 pixels
    Angle = 45.3867 (CCW from Right)

    Note, the data variable will be a blank dict until the clicks in the figure window.

    """

    fig = plt.gcf()

    curs.coords = {}
    curs.binding_id = None
    print('First Click')
    
    def on_click(event):
        if event.button is MouseButton.LEFT:
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                if len(curs.coords)==0:
                    print('data coords: %f %f' % (event.xdata, event.ydata))
                    curs.coords['x1'] = event.xdata
                    curs.coords['y1'] = event.ydata                    
                    curs.time = time.time()
                    print('Second Click')
                elif len(curs.coords)==2:
                    if time.time()-curs.time > 0.1:
                        print('data coords: %f %f' % (event.xdata, event.ydata))
                        curs.coords['x2'] = event.xdata
                        curs.coords['y2'] = event.ydata                                            
                        plt.disconnect(curs.binding_id)
                        
            if len(curs.coords)==4:
                x1,y1 = curs.coords['x1'],curs.coords['y1']
                x2,y2 = curs.coords['x2'],curs.coords['y2']                
                # Spherical ra/dec coordinates (or similar)
                #  use cos(dec) correction 
                if spherical:
                    mndec = np.mean([y1,y2])
                    cosdec = np.cos(np.deg2rad(mndec))
                    # Convert to arc seconds
                    if arsec:
                        mfac = 3600
                        units = 'arcsec'
                    else:
                        units = 'deg'
                else:
                    cosdec = 1.0
                    mfac = 1.0
                    units = 'pixels'
                dist = np.sqrt(((x1-x2)*cosdec)**2 + (y1-y2)**2)*mfac
                deltax = (x2-x1)*mfac*cosdec
                deltay = (y2-y1)*mfac
                angle = np.rad2deg(np.arctan2(y2-y1,x2-x1))
                print('Distance = %.4f %s' % (dist,units))
                print('Delta X = %.4f %s' % (deltax,units))
                print('Delta Y = %.4f %s' % (deltay,units))
                print('Angle = %.4f %s' % (angle,'(CCW from Right)'))
                slp,yoff = linear_coefficients([x1,x2],[y1,y2])

                curs.coords['dist'] = dist
                curs.coords['units'] = units
                curs.coords['deltax'] = deltax
                curs.coords['deltay'] = deltay
                curs.coords['angle'] = angle
                curs.coords['slp'] = slp
                curs.coords['yoff'] = yoff

                
    binding_id = plt.connect('button_press_event', on_click)
    curs.binding_id = binding_id

    return curs.coords

    
def clicker(over=False,connect=False):
    """
    This returns the position of mulitple cursor click on a matplotlib figure window.
    Click outside the data axes or right-click to stop.

    Parameters
    ----------
    over : boolean, optional
       Overplot the clicked position on the figure.  Default is False.
    connect : boolean, optional
       Connect the points with lines.  Default is False.

    Returns
    -------
    coords : list
      List of data coordinates.

    Example
    -------

    coords = clicker()

    <multiple clicks in the figure window.>

    data coords: 1.042335 7.838153
    data coords: 3.220864 6.093941
    data coords: 3.800472 8.213830
    data coords: 7.198178 3.759381
    data coords: 4.579946 0.807638
    data coords: 0.962389 3.866717
    data coords: 1.522011 6.845294

    <click outside data axes or right-click to stop>

    coords
    [[1.0423346449011444, 7.83815321583179],
    [3.2208636836628513, 6.093941326530615],
    [3.8004723270031215, 8.213829622758198],
    [7.198178167273673, 3.7593807977736566],
    [4.5799460197710715, 0.8076376004947439],
    [0.9623886251300728, 3.8667169140383444],
    [1.5220107635275753, 6.845294140383428]]

    Note, the coords variable will be a blank list until the click in the figure window.

    """

    fig = plt.gcf()

    curs.coords = []
    curs.binding_id = None
    
    def on_click(event):
        if event.button is MouseButton.LEFT:
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                print('data coords: %f %f' % (event.xdata, event.ydata))
                curs.coords += [[event.xdata,event.ydata]]
                if over:
                    plt.scatter([event.xdata],[event.ydata],marker='+',c='r')
                if connect and len(curs.coords)>1:
                    x1,y1 = curs.coords[-2]
                    x2,y2 = curs.coords[-1]
                    plt.plot([x1,x2],[y1,y2],c='lightsalmon')
                if connect or over:
                    fig.canvas.draw()
            else:
                plt.disconnect(curs.binding_id)

        if event.button is MouseButton.RIGHT:
            plt.disconnect(curs.binding_id)
            
    binding_id = plt.connect('button_press_event', on_click)
    curs.binding_id = binding_id

    return curs.coords


def selector(xdata,ydata,over=False,verbose=True,color='r',backcolor='white'):
    """
    Allows the user to select data points by clicking on or near then in a figure window.
    Use left-button to select and right-button to de-select points.  
    Click outside the data axes to stop.

    Parameters
    ----------
    xdata : array or list
      List or array of x-data.
    ydata : array or list
      List or array of y-data.
    over : boolean, optional
      Overplot the points selected.  Default is False.
    verbose : boolean, optional
      Verbose output to the screen.  Default is True.
    color : str, optional
      The color for overplotting.  Default is 'r'.
    backcolor : str, optional
      The background for overplotting if a point is de-selected.
         Default is 'white'.

    Returns
    -------
    index : list
      Index of selected points.

    Example
    -------

    index = selector(x,y,over=True)                                                                                                        
    ------------------------------------------------
        NUM       X           Y        IND        
    ------------------------------------------------

           1       7.000       7.000       7
           2       6.000       6.000       6
           3       5.000       5.000       5
           4       8.000       8.000       8
           5       2.000       2.000       2
           6       5.000       5.000       5         DE-SELECTED
           7       6.000       6.000       6         DE-SELECTED
           8       7.000       7.000       7         DE-SELECTED
           9       1.000       1.000       1
    ------------------------------------------------

    index                                                                                                                                   
    [8, 2, 1]

    Note, the index variable will be a blank list until the clicks in the figure window.

    """

    fig = plt.gcf()

    curs.coords = []
    curs.binding_id = None
    curs.xdata = xdata
    curs.ydata = ydata
    curs.index = []
    curs.time = time.time()
    curs.count = 0

    if verbose:
        print('------------------------------------------------')
        print('      NUM       X           Y        IND        ')
        print('------------------------------------------------')

    
    def on_click(event):
        if event.button is MouseButton.LEFT:
            # Add point
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                xcoord,ycoord = event.xdata,event.ydata
                if time.time()-curs.time > 0.1:
                    curs.time = time.time()
                    curs.coords += [[xcoord,ycoord]]
                    # Get the closest point
                    idx = ((curs.xdata-xcoord)**2+(curs.ydata-ycoord)**2).argmin()
                    # New point
                    if idx not in curs.index:
                        curs.count += 1                        
                        curs.index += [idx]
                        if verbose:
                            print('%8d%12.3f%12.3f%8d' % (curs.count,curs.xdata[idx],curs.ydata[idx],idx))
                        if over:
                            plt.scatter([curs.xdata[idx]],[curs.ydata[idx]],marker='D',facecolor='none',edgecolor=color)
                            fig.canvas.draw()
            else:
                plt.disconnect(curs.binding_id)
                if verbose:
                    print('------------------------------------------------')
                
        if event.button is MouseButton.RIGHT:
            # Remove point
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                xcoord,ycoord = event.xdata,event.ydata
                if time.time()-curs.time > 0.1 and len(curs.index)>0:
                    curs.time = time.time()                    
                    curs.count += 1
                    # Get the closest point that was previously selected
                    idx = ((curs.xdata[curs.index]-xcoord)**2+(curs.ydata[curs.index]-ycoord)**2).argmin()
                    index,xcoo,ycoo = curs.index[idx],curs.xdata[curs.index][idx],curs.ydata[curs.index][idx]
                    # Point to remove
                    curs.index.pop(idx)
                    if verbose:
                        print('%8d%12.3f%12.3f%8d%20s' % (curs.count,xcoo,ycoo,index,'DE-SELECTED'))
                    if over:
                        plt.scatter([xcoo],[ycoo],marker='D',facecolor='none',edgecolor=backcolor)
                        fig.canvas.draw()                   

                
    binding_id = plt.connect('button_press_event', on_click)
    curs.binding_id = binding_id

    return curs.index


curs = Cursor()

