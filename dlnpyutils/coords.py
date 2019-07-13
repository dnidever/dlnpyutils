#!/usr/bin/env python
#
# COORDS.PY - coordinate utility functions.
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20190723'  # yyyymmdd

import numpy as np


def rotsph(lon,lat,clon,clat,anode=None,reverse=False,original=False):
    '''
    This rotates a spherical coordinate system to a new pole

    I got the equations for this from the paper
    Calabretta et al. 2002, A&A, 395, 1077
    Equation 5.

    Also, see rotate_lb.pro that uses a matrix method
    and for which you can specify the equator you'd like.
    rotsph.pro is faster than rotate_lb.pro
    By default, the origin is the point where the two equators
    cross (unless =ANODE is set).
    This should give you the same result (within ~1E-10")
    rotate_lb,lon,lat,[clon,clat],[clon+90,0.],nlon,nlat

    Parameters
    ----------
    lon       Array of longitudes to be rotated
    lat       Array of latitudes to be rotated
    clon      Longitude of the new NORTH POLE in the old coordinate system
    clat      Latitude of the new NORTH POLE in the old coordinate system
    =anode    The "Ascending Node" which is the longitude (in the new
             system) of the first point where the old equator cross
             the new one.  This sets the zero-point of the new
             longitude.  By default the zero-point of the new
             coordinate system is the point where the two equators
             cross.
    /original Set the new longitude zero-point to be clon (if clat>0)
             and clon+180 (if clat<0).  This is the way it was
             originally done.  DON'T USE WITH "ANODE"
    /stp      Stop at the end of the program
    /reverse  The reverse operation.  In that case (nlon,nlat) should be input
           as (lon,lat). E.g.

           rotsph,ra,dec,cra,cdec,nlon,nlat
           rotsph,nlon,nlat,cra,cdec,nra,ndec,/reverse
           
           (ra,dec) and (nra,ndec) should be identical to 1E-10.

    Returns
    -------
    nlon  Array of rotated longitudes
    nlat  Array of rotated latitudes

    '''

    radeg = 180.0/np.pi

    alphap = np.array(clon/radeg)
    deltap = np.array(clat/radeg)
    phip = np.array(90.0/radeg)
    if original: phip = np.array(180.0/radeg)   # original way
    thetap = np.array(90.0/radeg)

    # By default the origin of the new coordinate system is the point
    # where the two equators cross for the first time
    #  Unless /original is set.  Then the zero-point is at clon
    #   (if clat>0) and clon+180 (if clat<0)

    # NORMAL
    if reverse is False:
        alpha = np.array(lon/radeg)
        delta = np.array(lat/radeg)

        # arg(x,y) but atan(y,x)
        phi = phip + np.arctan2( -np.cos(delta)*np.sin(alpha-alphap), np.sin(delta)*np.cos(deltap)- \
                                 np.cos(delta)*np.sin(deltap)*np.cos(alpha-alphap) )

        theta = np.arcsin( limit((np.sin(delta)*np.sin(deltap)+np.cos(delta)*np.cos(deltap)*np.cos(alpha-alphap)),-1,1) )

        # Preparing the output
        nlon = phi*radeg
        nlat = theta*radeg

        # Ascending Node
        #  By default the origin of nlon is the point where the two equators
        #  cross the first time
        if anode is not None: nlon += anode

    # REVERSE
    else:
        phi = np.array(lon/radeg)
        theta = np.array(lat/radeg)

        # Ascending Node
        if anode is not None: phi = (lon-anode)/radeg

        # arg(x,y) but atan(y,x)
        alpha = alphap + np.arctan2( -np.cos(theta)*np.sin(phi-phip), np.sin(theta)*np.cos(deltap) - \
                                     np.cos(theta)*np.sin(deltap)*np.cos(phi-phip))
        delta = np.arcsin( np.sin(theta)*np.sin(deltap) + np.cos(theta)*np.cos(deltap)*np.cos(phi-phip) )

        # Preparing the output
        nlon = alpha*radeg
        nlat = delta*radeg

    # Want everything less than 360.0
    nlon = nlon % 360.0

    # Make negative points positive
    bd, = np.where(nlon < 0.0)
    if len(bd)>0:
        nlon[bd] = nlon[bd]+360.0

    return nlon, nlat


def rotsphcen(lon,lat,clon,clat,polar=False,gnomic=False,reverse=False):
    '''
    This is very similar to rotsph.pro except that the coordinates
    input are not for the north pole but for the new equator.
    Everything is in DEGREES.
    
    Parameters
    ----------
    lon       Array of longitudes to be rotated
    lat       Array of latitudes to be rotated
    clon      Longitude of the new EQUATOR in the old coordinate system
    clat      Latitude of the new EQUATOR in the old coordinate system
    /polar    Return polar coordinates (rad,phi) instead of LON/LAT.
    phi starts at North.
    /gnomic   Also do a gnomic (tangent plane) projection.
    /reverse  The reverse operation.  In that case (nlon,nlat) should be input
    as (lon,lat). E.g.

           rotsphcen,ra,dec,cra,cdec,nlon,nlat
           rotsphcen,nlon,nlat,cra,cdec,nra,ndec,/reverse
           
           (ra,dec) and (nra,ndec) should be identical to 1E-10.

    Returns
    -------
    nlon  Array of rotated longitudes.  If /polar then this is PHI
       the polar angle (measured from N toward E).
       
    nlat  Array of rotated latitudes.  If /polar then this is RAD
       the polar radial distance.
    '''

    radeg = 180.0/np.pi

    # NOT polar coordinates
    if (polar is False) and (gnomic is False):

        # Get coordinates for the north pole
        np_lon = np.array(clon)
        np_lat = np.array(clat+90.0)
        if (np_lat > 90.0):
            np_lon = np.array(clon+180.0)
            np_lon = np_lon % 360.0
            np_lat = 90.0-clat

        # Run rotsph.pro
        # NORMAL
        if reverse is False:
            nlon, nlat = rotsph(lon,lat,np_lon,np_lat,original=True)
        # REVERSE
        else:
            nlon, nlat = rotsph(lon,lat,np_lon,np_lat,reverse=True,original=True)

            # need to flip them around by 180 deg b/c the zero-point
            #  is set by the NP lon
            if ((clat+90.0) > 90.0):
                nlon = (nlon+180.0) % 360.0
                nlat = -nlat

        # Make the longitudes continuous
        nlon = (nlon+180.0) % 360.0
        nlon = nlon-180.0


    # POLAR or GNOMIC
    else:

        # Making polar coordinates

        #-----------------
        # NORMAL
        #------------------
        if reverse is False:
            # Run rotsph.pro and specify the center of the field (the origin) as the
            #  the Npole
            phi, theta = rotsph(lon,lat,clon[0],clat[0],original=True)
            # phi is now going clockwise and from South
            orig_phi = phi
            phi = -phi+180.0      # counterclockwise
            phi = phi % 360.0
            rad = 90.0-theta

            # Making gnomic projection
            if gnomic:
                # Scale the radius
                rad = radeg * np.cos(theta/radeg)/np.sin(theta/radeg)
                # Now convert from gnomic polar to X/Y
                # phi is from N toward E
                # x = R*sin(phi)
                # y = R*cos(phi)
                nlon = rad*np.sin(phi/radeg)
                nlat = rad*np.cos(phi/radeg)

            # Output polar coordinates
            if polar:
                nlon = phi
                nlat = rad

        #-----------------
        # REVERSE
        #-----------------
        else:

            # Polar
            if polar:
                phi = lon
                rad = lat
                theta = 90.0-rad

            # Removing gnomic projection
            if gnomic:
                # Now convert from X/Y to gnomic polar
                # phi is from N toward E
                # x = R*sin(phi)
                # y = R*cos(phi)
                #nlon = rad*sin(phi/radeg)
                #nlat = rad*cos(phi/radeg)
                rad = sqrt(lon**2.0+lat**2.0)
                phi = radeg*np.arctan2(lon,lat)      # in degrees
                # Scale the radius
                #rad = radeg * cos(theta/radeg)/sin(theta/radeg)
                theta = radeg*np.arctan(radeg/rad)   # in degrees

            # phi is now going clockwise and from South
            phi = -phi+180.0       # reverse phi

            #Run rotsph.pro and specify the center of the field (the origin) as the
            #  the Npole
            nlon, nlat = rotsph(phi,theta,clon[0],clat[0],reverse=True,original=True)

    return nlon, nlat


def doPolygonsOverlap(xPolygon1, yPolygon1, xPolygon2, yPolygon2):
    """Returns True if two polygons are overlapping."""

    # How to determine if two polygons overlap.
    # If a vertex of one of the polygons is inside the other polygon
    # then they overlap.
    
    n1 = len(xPolygon1)
    n2 = len(xPolygon2)
    isin = False

    # Loop through all vertices of second polygon
    for i in range(n2):
        # perform iterative boolean OR
        # if any point is inside the polygon then they overlap   
        isin = isin or isPointInPolygon(xPolygon1, yPolygon1, xPolygon2[i], yPolygon2[i])

    # Need to do the reverse as well, not the same
    for i in range(n1):
        isin = isin or isPointInPolygon(xPolygon2, yPolygon2, xPolygon1[i], yPolygon1[i])

    return isin

def isPointInPolygon(xPolygon, yPolygon, xPt, yPt):
    """Returns boolean if a point is inside a polygon of vertices."""
    
    # How to tell if a point is inside a polygon:
    # Determine the change in angle made by the point and the vertices
    # of the polygon.  Add up the delta(angle)'s from the first (include
    # the first point again at the end).  If the point is inside the
    # polygon, then the total angle will be +/-360 deg.  If the point is
    # outside, then the total angle will be 0 deg.  Points on the edge will
    # outside.
    # This is called the Winding Algorithm
    # http://geomalgorithms.com/a03-_inclusion.html

    n = len(xPolygon)
    # Array for the angles
    angle = np.zeros(n)

    # add first vertex to the end
    xPolygon1 = np.append( xPolygon, xPolygon[0] )
    yPolygon1 = np.append( yPolygon, yPolygon[0] )

    wn = 0   # winding number counter

    # Loop through the edges of the polygon
    for i in range(n):
        # if edge crosses upward (includes its starting endpoint, and excludes its final endpoint)
        if yPolygon1[i] <= yPt and yPolygon1[i+1] > yPt:
            # if (P is  strictly left of E[i])    // Rule #4
            if isLeft(xPolygon1[i], yPolygon1[i], xPolygon1[i+1], yPolygon1[i+1], xPt, yPt) > 0: 
                 wn += 1   # a valid up intersect right of P.x

        # if edge crosses downward (excludes its starting endpoint, and includes its final endpoint)
        if yPolygon1[i] > yPt and yPolygon1[i+1] <= yPt:
            # if (P is  strictly right of E[i])    // Rule #4
            if isLeft(xPolygon1[i], yPolygon1[i], xPolygon1[i+1], yPolygon1[i+1], xPt, yPt) < 0: 
                 wn -= 1   # a valid up intersect right of P.x

    # wn = 0 only when P is outside the polygon
    if wn == 0:
        return False
    else:
        return True

def isLeft(x1, y1, x2, y2, x3, y3):
    # isLeft(): test if a point is Left|On|Right of an infinite 2D line.
    #   From http://geomalgorithms.com/a01-_area.html
    # Input:  three points P1, P2, and P3
    # Return: >0 for P3 left of the line through P1 to P2
    # =0 for P3 on the line
    # <0 for P3 right of the line
    return ( (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) )


