#!/usr/bin/env python

# Make thumbnail PNG of an APOGEE 2D exposure using CDR

# D. Nidever 10/25/21

import os
import numpy as np
from glob import glob
from argparse import ArgumentParser
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def thumbnail(files,nbin=4,outfile=None,ext='jpg',nsig=3,cmap='Greys'):
    """ Make a thumbnail for an exposure."""

    if type(files) is str:
        files = [files]

    for filename in files:
    
        # Load the file
        im = fits.getdata(filename)

        # Rebin
        newsh = (im.shape[0]//nbin,im.shape[1]//nbin)
        shape = (newsh[0], im.shape[0] // newsh[0],
                 newsh[1], im.shape[1] // newsh[1])
        smim = im.reshape(shape).mean(-1).mean(1)

        med = np.median(im)
        sig = np.median(np.abs(im-med))*1.4826

        # Make image
        base = os.path.basename(filename)
        base,_ = os.path.splitext(base)
        dirname = os.path.abspath(os.path.dirname(filename))
        outfile = dirname+'/'+base+'_thumb.'+ext
        figsize = 8 
        fig, ax = plt.subplots(1,1,figsize=(figsize,figsize))
        vmin = med-nsig*sig
        vmax = med+nsig*sig
        ax1 = ax.imshow(smim,origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        title = str(filename)
        plt.title(title)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().y1-ax.get_position().y0])
        fig.colorbar(ax1, cax=cax)
        plt.savefig(outfile,bbox_inches='tight')
        plt.close(fig)
        print('Saving figure to ',outfile)
