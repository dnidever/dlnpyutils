#!/usr/bin/env python
#
# ASTRO.PY - astronomy utility functions
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20191226'  # yyyymmdd

#import re
#import logging
#import os
#import sys
import numpy as np
import warnings
#from astropy.io import fits
from astropy.table import Table, Column
#from astropy import modeling
#from astropy.convolution import Gaussian1DKernel, convolve
#from glob import glob
#from scipy.signal import medfilt
#from scipy.ndimage.filters import median_filter,gaussian_filter1d
#from scipy.optimize import curve_fit, least_squares
#from scipy.special import erf
#from scipy.interpolate import interp1d
#from scipy.linalg import svd
from astropy.utils.exceptions import AstropyWarning
#import socket
#from scipy.signal import convolve2d
#from scipy.ndimage.filters import convolve
#import astropy.stats
from . import utils
from .coords import xyz2lbd,rotate_lb


# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Convert wavelengths in air to vacuum
def airtovac(wave):
    """
    Convert air wavelengths to vacuum wavelengths 

    Wavelengths are corrected for the index of refraction of air under 
    standard conditions.  Wavelength values below 2000 A will not be 
    altered.  Uses relation of Ciddor (1996).

    INPUT/OUTPUT:
      WAVE_AIR - Wavelength in Angstroms, scalar or vector
              If this is the only parameter supplied, it will be updated on
              output to contain double precision vacuum wavelength(s). 

    EXAMPLE:
      If the air wavelength is  W = 6056.125 (a Krypton line), then 
      AIRTOVAC, W yields an vacuum wavelength of W = 6057.8019

    METHOD:
	Formula from Ciddor 1996, Applied Optics 62, 958

    NOTES: 
      Take care within 1 A of 2000 A.   Wavelengths below 2000 A *in air* are
      not altered.       
    REVISION HISTORY
      Written W. Landsman                November 1991
      Use Ciddor (1996) formula for better accuracy in the infrared 
          Added optional output vector, W Landsman Mar 2011
      Iterate for better precision W.L./D. Schlegel  Mar 2011
    """

    wave_air = np.atleast_1d(wave).copy()  # makes sure it's an array
    wave_vac = np.atleast_1d(wave).copy()  # initialize
    
    g,ng = utils.where(wave_vac >= 2000)     #Only modify above 2000 A
    
    if ng>0:
        for iter in range(2):
            sigma2 = (1e4/wave_vac[g] )**2     # Convert to wavenumber squared
            
            # Compute conversion factor
            fact = 1.0 +  5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/( 57.362e0 - sigma2)
            
            wave_vac[g] = wave_air[g]*fact              # Convert Wavelength

    return wave_vac


def vactoair(wave_vac):
    """
    Convert vacuum wavelengths to air wavelengths

    Corrects for the index of refraction of air under standard conditions.  
    Wavelength values below 2000 A will not be altered.  Accurate to 
    about 10 m/s.


    INPUT/OUTPUT:
	WAVE_VAC - Vacuum Wavelength in Angstroms, scalar or vector
		If the second parameter is not supplied, then this will be
               updated on output to contain double precision air wavelengths.

    EXAMPLE:
	If the vacuum wavelength is  W = 2000, then 

	IDL> VACTOAIR, W 

	yields an air wavelength of W = 1999.353 Angstroms

    METHOD:
	Formula from Ciddor 1996  Applied Optics , 35, 1566

    REVISION HISTORY
      Written, D. Lindler 1982 
      Documentation W. Landsman  Feb. 1989
      Use Ciddor (1996) formula for better accuracy in the infrared 
           Added optional output vector, W Landsman Mar 2011
    """

  
    wave_vac = np.atleast_1d(wave_vac).copy()  # makes sure it's an array
    wave_air = np.atleast_1d(wave_vac).copy()  # initialize
    g,ng = utils.where(wave_air >= 2000)     # Only modify above 2000 A
    
    if ng>0:
        sigma2 = (1e4/wave_vac[g] )**2   # Convert to wavenumber squared

        # Compute conversion factor
        fact = 1.0 +  5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/( 57.362e0 - sigma2)
    
        # Convert wavelengths
        wave_air[g] = wave_vac[g]/fact

    return wave_air


def vgsr2vhelio(vgsr,lon,lat,vcirc=240.0):
    """
    Given an input array called vhel with values of heliocentric
    radial velocities and input arrays of the same length with
    the Galactic coordinates of the object (gl,gb),
    this code calculates Vlsr and Vgsr.

    INPUTS:
    vgsr     Galactocentric Standard of Rest Radial Velocity
    lon      Galactic longitude (in degrees)
    lat      Galactic latitude (in degrees)
    =vcirc   MW circular velocity.  220 km/s by default.
   
    OUTPUTS:
    Vhelio  The heliocentric velocity in km/s.

    USAGE:
    IDL>vhelio = vgsr2vhelio(vgsr,lon,lat)

    By D.Nidever 2005
    """

    # code assumes gl & gb given in degrees.

    gl = np.deg2rad(lon)
    gb = np.deg2rad(lat)

    cgl=np.cos(gl)
    sgl=np.sin(gl)
    cgb=np.cos(gb)
    sgb=np.sin(gb)

    #  This equation takes the solar motion w.r.t. the LSR as
    #  (9,11,6) km/sec (Mihalas & Binney)
    #vlsr=vhel+((9.0d0*cgb*cgl)+(11.0d0*cgb*sgl)+(6.0d0*sgb))

    #  This equation takes the rotation velocity of the LSR to
    #  be 220 km/sec

    vlsr = vgsr-(vcirc*cgb*sgl)

    #   Using updated values from Dehnen & Binney 1998, MNRAS, 298, 387
    #   U = 10.00 +/- 0.36 km/s
    #   V = 5.25 +/- 0.62 km/s
    #   W = 7.17 +/- 0.38 km/s
    #   This is in a right-handed system
    #vlsr=vhel+((9.0d0*cgb*cgl)+(11.0d0*cgb*sgl)+(6.0d0*sgb))
    #vhel=vlsr-( (9.0d0*cgb*cgl)+(11.0d0*cgb*sgl)+(6.0d0*sgb) )
    vhel = vlsr-( (10.0*cgb*cgl)+(5.25*cgb*sgl)+(7.17*sgb) )
    
    return vhel


def vgsr2vlsr(v,l,b,dir,vcirc=240.0):
    """

    PURPOSE:
    Given an input array called vhel with values of heliocentric
    radial velocities and input arrays of the same length with
    the Galactic coordinates of the object (gl,gb),
    this code calculates Vlsr and Vgsr.

    INPUTS:
    vel     Vgsr/Vlsr velocity to convert
    l       Galacitc longitude in degrees
    b       Galactic latitutde in degrees
    dir     Direction of conversion
             dir=1    Vgsr->Vlsr
             dir=-1   Vlsr->Vgsr
    =vcirc  The rotation speed of the MW disk at the sun's
                 radius.  Vcirc=240 km/s by default.

    OUTPUTS:
    Vlsr/Vgsr velocity array

    USAGE:
    IDL>vlsr = vgsr2vlsr(vgsr,lon,lat)

    By D.Nidever Feb 2006
    """

    gl = l
    gb = b
    
    gl = np.deg2rad(gl)
    gb = np.deg2rad(gb)

    cgl = np.cos(gl)
    sgl = np.sin(gl)
    cgb = np.cos(gb)
    sgb = np.sin(gb)

    #  This equation takes the solar motion w.r.t. the LSR as
    #  (9,11,6) km/sec (Mihalas & Binney)
    #
    #vlsr=vhel+((9.0d0*cgb*cgl)+(11.0d0*cgb*sgl)+(6.0d0*sgb))


    #  This equation takes the rotation velocity of the LSR to
    #  be 220 km/sec
    #
    #vgsr=vlsr+(220.0d0*cgb*sgl)

    #vgsr=vlsr+(220.0d0*cgb*sgl)
    #vlsr=vgsr-(220.0d0*cgb*sgl)

    # input vgsr, output vlsr
    if dir == 1:
        v2 = v-(vcirc*cgb*sgl)

    # input vlsr, output vgsr
    if dir == -1:
        v2 = v+(vcirc*cgb*sgl)

    return v2

def lmcvlos(ra,dec,halo=False,hicenter=False,alpha=81.9000,
            delta=-69.8666666,dist=50.1,inc=34.75,lineofnodes=None,
            vsys=262.0,muw=-1.858,mun=0.385,didt=-0.37):
    """
    This computes van der Marel's (AJ 2002), line-of-sight
    velocity for the LMC, RA and DEC should be in degrees!!
    
    Parameters
    ----------
    ra : numpy array
       Array of Right Ascension values (in degrees!) for
         which the radial velocity is desired.
    dec : numpy array
       Array of Declination values for which the radial
         velocity is desired.
    halo : boolean, optional
       Halo model with no rotation or nutation (didt).
         Default is False.
    hicenter : boolean, optional
       Use the HI parameters (stellar by default).
         alpha = 79.40 deg
         delta = -69.03333 deg
         lineofnodes = 162. deg
    alpha : float, optional
       RA at the center of the LMC in degrees.
         Default is alpha=81.9000.
    delta : float, optional
       DEC at the center of the LMC in degrees.
         Default is delta=-69.8666666.
    dist : float, optional
       Distance to the center of the LMC in kpc.
         Default is dist=50.1 kpc.
    inc : float, optional
       Inclination of the LMC disk in degrees.
         Default is inc=34.75 deg.
    lineofnodes : float, optional
       The orientation angle of the line-of-nodes CCW from North (in degrees).
         This is "bigtheta" in vdM2002.
         Default is lineofnodes=129.9 deg unless hicenter=True, then 162 deg is used.
    vsys : float, optional
       Heliocentric radial velocity of the center of the LMC in km/s.
         Default is vsys=262.0 km/s.
    muw : float, optional
       Proper motion in RA towards the West (so negative pmra) in mas/yr.
         Default is muw=-1.858 mas/yr.
    mun : float, optional
       Proper motion in DEC towards the North (pmdec) in mas/yr.
         Default is mun=0.385 mas/yr.
    didt : float, optional
       The nutation or change in inclination over time.
         Default is didt=-0.37 mas/yr.

    Returns
    -------
    res : table
       Results with vlos, rho, bigphi.

    Examples
    --------

    res = lmcvlos(ra,dec)

    By D.Nidever  August 2005
    Translated to Python by D.Nidever  July 2024
    """

    #---- All of the parameters -----

    # Stellar LMC CM position in ra,dec
    # alpha = 81.9000      # deg
    # delta = -69.8666666  # deg
    # these are the default values
    
    # Inclination & Line of Nodes
    #inc = 34.75
    if halo:
        inc = 0.0
    #bigtheta = 129.9         # bigtheta, counter-clockwise from North
    if lineofnodes is None and hicenter==False:
        bigtheta = 129.9         # default value if hicenter not set
    
    # HI LMC center from Kim et al.(1998)
    if hicenter:
        alpha = 79.4        # deg
        delta = -69.03333   # deg
        # HI line-of-nodes
        bigtheta = 162.

    # di/dt
    ##didt = -103.0              # deg/Gyr = -0.37 mas/yr  # 278.378 mas/yr -> deg/Gyr  # Eq. 35
    ##didt = -6.0                # mas/yr, from Kallivayalil et al.2006
    #didt = -0.37                # default now
    if halo:
        didt = 0.0

    # Distance (in kpc)
    # d0 = 50.1
    # default
    
    # CM Radial velocity (Heliocentric, in km/s)
    # vsys = 262.2          # Eq.37
    # default
    
    # Proper motion
    #muw = -1.68           # proper motion towards WEST ( -d alpha/dt * cos(dec) ),in mas/year
    #mun = 0.34            # proper motion towards NORTH ( d dec/dt) in mas/year
    # from Kallivaylil et al. (2006)
    #muw = -1.94
    #mun = 0.43
    # from Kallivaylil et al. (2013)
    #muw = -1.910  # -1.910+/-0.02
    #mun =  0.229  #  0.229+/0.047
    # from Luri+2021
    #muw = -1.858
    #mun = 0.385
    # default now
    
    # Vt = mu(arcsec/year) * 4.74 * dist(pc)
    # bigthetat = atan(-muw,mun)*!radeg
    # vtc = vt * cos(bigthetat-bigtheta)  Eq.26
    # vts = vt * sin(bigthetat-bigtheta)
    mutot = np.sqrt(muw**2 + mun**2)
    vt = mutot*4.74*dist
    bigthetaT = np.rad2deg(np.arctan2(-muw,mun))   # angle of transverse velocity vector
    #vtc = vt * cos((bigthetat-bigtheta)/!radeg)
    #vts = vt * sin((bigthetat-bigtheta)/!radeg)
    #vtc = 253.            # vt = sqrt(vtc^2.+vts^2.), Vt along the line of nodes, from Eq.43
    #wts = -402.9          # vts + dist*(di/dt), Eq.29,37
    #wts = vts + dist*didt

    # I'm not getting the same wts

    #---- Create the projected rotation curve ----

    n = len(np.atleast_1d(ra))

    # CONVERTING coordinates to LMC-centric coordinate system (rho,phi)
    #  rho is radial distance from LMC CM
    #  phi is the position angle, counter-clockwise from WEST
    #  position angles with "big" are measured from NORTH

    # Copied from sphtrigdist.pro
    # Can also use sphdist.pro
    cosa = np.sin(np.deg2rad(delta))*np.sin(np.deg2rad(dec))
    cosa += np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(dec))*np.cos(np.deg2rad(alpha-ra))
    rho = np.rad2deg(np.arccos(cosa))

    npole = [alpha,delta]                 # North pole at CM
    equator = [alpha-1,delta]             # longitude starts from west
    phi,_,_,_ = rotate_lb(ra,dec,npole,equator)
    phi = -phi                                    # want counter-clockwise
    phi[phi<0] += 360                             # all positive

    # Getting van der Marel's model
    bigphi = phi-90.    # position angle from NORTH
    bigthetat = np.rad2deg(np.arctan2(-muw,mun))     # bigthetat = thetat-90
    # thetat is the direction of the transverse motion
    # on the sky counter-clockwise from WEST
    # bigthetat is measured from NORTH
    # Calculate directly from proper motion values
    # f from Eq.25
    f = (np.cos(np.deg2rad(inc))*np.cos(np.deg2rad(rho)) -
         np.sin(np.deg2rad(inc))*np.sin(np.deg2rad(rho))*np.sin(np.deg2rad(bigphi-bigtheta)))
    f /= np.sqrt( (np.cos(np.deg2rad(inc))*np.cos(np.deg2rad(bigphi-bigtheta)))**2 +
                  np.sin(np.deg2rad(bigphi-bigtheta))**2 )


    # ROTATION component
    # The best fit is for vtc=599.  I found eta=2.6, r0=2.5 seems to look the most
    # like his fig.6 using Eq.36.
    # For other vtc use Eq.34.
    #rho = scale_vector(findgen(1000),0,0.3*50.)
    #f = fltarr(1000)+1.

    rprime = dist*np.sin(np.deg2rad(rho))/f    # distance from LMC CM in kpc, in plane, Eq.19
    # Olsen & Massey 2006 use eta=3.0, v0=61.1, r0/d0=0.041 (do=50.1 kpc) -> r0=2.054
    v0 = 61.6
    eta = 3.0
    r0 = 2.054
    #v0 = 130.0     #  pg. 2647 (pg.9), col.2, end of first paragraph.  for vtc=600 #49.8
    #eta = 2.5 #2.6  #0.5
    #r0 = 2.5   #2.5 #1.4
    # van der Marel 2002, section 10
    # This v0 might be a little low
    v0 = 49.7
    eta = 2.68
    r0 = 2.7555       # r0/d0=0.055

    VR = (v0*rprime**eta)/((rprime**eta)+(r0**eta))            # Eq. 36
    ##vtc = 300.
    #VR = VR + (vtc-599.)*tan(rho/!radeg)/sin(inc/!radeg)   # Eq. 34, also see 3rd paragraph on pg.2647 (pg.9)

    #r = [0.009, 0.028, 0.044, 0.06, 0.081, 0.096, 0.113, 0.130, 0.148, 0.178]*50.0
    #v = [-27.9, 14.2, 25.3, 35.7, 57.3, 50.0, 39.4, 46.6, 55.8, 32.1]
    #
    #plot,rprime/50.,vr,/nodata,xr=[0.,0.23],yr=[-75,200],xs=1,ys=1
    #oplot,rprime/50.,vr,thick=1.3
    #oplot,[0,1],[0,0]
    #oplot,[0,1],[100,100],linestyle=2
    #oplot,r/50.,v,ps=8

    # Correction for asymmetric drift, Section 8.1 (pg.14)
    #sigv = 21.                        # velocity dispersion, sect. 7.3, fig.6, or table 2.
    #VR = sqrt(VR^2. + 6.*sigv^2.)

    # CALCULATING the Line-of-Sight velocity (Vlos), Eq.30+31
    #vlos1 = vsys*cos(rho/!radeg)
    #vlos2 = wts*sin(rho/!radeg)*sin((bigphi-bigtheta)/!radeg)
    #vlos3 = vtc*sin(rho/!radeg)*cos((bigphi-bigtheta)/!radeg)
    #vlos4 = -f*VR*sin(inc/!radeg)*cos((bigphi-bigtheta)/!radeg)
    # Using Eq.24 instead
    vlos1 = vsys*np.cos(np.deg2rad(rho))
    vlos2 = vt*np.sin(np.deg2rad(rho))*np.cos(np.deg2rad(bigphi-bigthetaT))
    # didt is in mas/yr
    # didt(mas/yr) * d0(kpc) * 4.74 = velocity(km/s)
    vlos3 = dist*didt*4.74*np.sin(np.deg2rad(rho))*np.sin(np.deg2rad(bigphi-bigtheta))
    vlos4 = -f*VR*np.sin(np.deg2rad(inc))*np.cos(np.deg2rad(bigphi-bigtheta))
    #vlos = vsys*cos(rho/!radeg) + wts*sin(rho/!radeg)*sin((bigphi-bigtheta)/!radeg)
    #vlos = vlos + (vtc*sin(rho/!radeg)-f*VR*sin(inc/!radeg))*cos((bigphi-bigtheta)/!radeg)
    #vgsr = vlos1 + vlos2 + vlos3 + vlos4
    vlos = vlos1 + vlos2 + vlos3 + vlos4

    # Put the results in a table
    dt = [('ra',float),('dec',float),('rho',float),('vlos',float),('bigphi',float)]
    res = Table(np.zeros(n,dtype=np.dtype(dt)))
    res['ra'] = ra
    res['dec'] = dec
    res['rho'] = rho
    res['vlos'] = vlos
    res['bigphi'] = bigphi

    return res
