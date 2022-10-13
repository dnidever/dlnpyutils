#!/usr/bin/env python
#
# COORDS.PY - coordinate utility functions.
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20190723'  # yyyymmdd

import numpy as np
import copy
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from . import utils,ladfit
from astropy.coordinates import frame_transform_graph,SkyCoord
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
import astropy.coordinates as coord
import astropy.units as u

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

        theta = np.arcsin( utils.limit((np.sin(delta)*np.sin(deltap)+np.cos(delta)*np.cos(deltap)*np.cos(alpha-alphap)),-1,1) )

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
    bd = (nlon < 0.0)
    if np.sum(bd)>0:
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
            phi, theta = rotsph(lon,lat,clon,clat,original=True)
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
                rad = np.sqrt(lon**2.0+lat**2.0)
                phi = radeg*np.arctan2(lon,lat)      # in degrees
                # Scale the radius
                #rad = radeg * cos(theta/radeg)/sin(theta/radeg)
                theta = radeg*np.arctan(radeg/rad)   # in degrees

            # phi is now going clockwise and from South
            phi = -phi+180.0       # reverse phi

            #Run rotsph.pro and specify the center of the field (the origin) as the
            #  the Npole
            nlon, nlat = rotsph(phi,theta,clon,clat,reverse=True,original=True)

    return nlon, nlat


def doPolygonsOverlap(xPolygon1, yPolygon1, xPolygon2, yPolygon2):
    """Returns True if two polygons are overlapping."""

    # How to determine if two polygons overlap.
    # If a vertex of one of the polygons is inside the other polygon
    # then they overlap.
    
    n1 = len(xPolygon1)
    n2 = len(xPolygon2)
    isin = False

    # If ranges don't overlap, then polygons don't overlap
    if rangeoverlap(xPolygon1,xPolygon2)==False or rangeoverlap(yPolygon1,yPolygon2)==False:
        return False
    
    # Loop through all vertices of second polygon
    for i in range(n2):
        # perform iterative boolean OR
        # if any point is inside the polygon then they overlap   
        isin = isin or isPointInPolygon(xPolygon1, yPolygon1, xPolygon2[i], yPolygon2[i])

    # Need to do the reverse as well, not the same
    for i in range(n1):
        isin = isin or isPointInPolygon(xPolygon2, yPolygon2, xPolygon1[i], yPolygon1[i])

    # Two polygons can overlap even if there are no vertices inside each other.
    # Need to check if the line segments overlap
    if isin==False:
        intersect = False
        # Add first vertex to the end
        xp1 = np.append( xPolygon1, xPolygon1[0] )
        yp1 = np.append( yPolygon1, yPolygon1[0] )
        xp2 = np.append( xPolygon2, xPolygon2[0] )
        yp2 = np.append( yPolygon2, yPolygon2[0] )
        for i in range(4):
            for j in range(4):
                intersect = intersect or doLineSegmentsIntersect(xp1[i:i+2],yp1[i:i+2],xp2[j:j+2],yp2[j:j+2])
                if intersect==True:
                    return True
        isin = isin or intersect
        
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

def rangeoverlap(a,b):
    """does the range (start1, end1) overlap with (start2, end2)"""
    return max(a) >= min(b) and min(a) <= max(b)
    
def doLineSegmentsIntersect(x1, y1, x2, y2):
    """ Do two line segments intersect."""

    # Check vertical lines

    # Vertical lines, but NOT same X-values
    if x1[0]==x1[1] and x2[0]==x2[1] and x1[0]!=x2[0]:
        return False  # No overlap
    
    # Vertical lines with same X values
    if x1[0]==x1[1] and x2[0]==x2[1] and x1[0]==x2[0]:
        # Check intersection of Y ranges
        I1 = [np.min(y1), np.max(y1)]
        I2 = [np.min(y2), np.max(y2)]
    
        # And we could say that Xa is included into :
        Ia = [max( np.min(y1), np.min(y2) ),
              min( np.max(y1), np.max(y2) )]
    
        # Now, we need to check that this interval Ia exists :
        if rangeoverlap(y1,y2)==False:
            return False  # There is no mutual abcisses        
        else:
            return True   # There is overlap

    # The equation of a line is:
    #
    # f(x) = A*x + b = y
    # For a segment, it is exactly the same, except that x is included on an interval I.
    # 
    # If you have two segments, defined as follow:
    #
    # Segment1 = {(X1, Y1), (X2, Y2)}
    # Segment2 = {(X3, Y3), (X4, Y4)}
    # The abcisse Xa of the potential point of intersection (Xa,Ya) must be contained in both interval I1 and I2, defined as follow:
    I1 = [np.min(x1), np.max(x1)]
    I2 = [np.min(x2), np.max(x2)]
    
    # And we could say that Xa is included into :
    Ia = [max( np.min(x1), np.min(x2) ),
          min( np.max(x1), np.max(x2) )]
    
    # Now, we need to check that this interval Ia exists :
    if rangeoverlap(x1,x2)==False:        
        return False  # There is no mutual abcisses

    # Check that the Y-ranges overlap as well
    if rangeoverlap(y1,y2)==False:        
        return False  # There is no mutual y-value overlap
    
    # So, we have two line formula, and a mutual interval. Your line formulas are:
    # f1(x) = m1*x + b1 = y
    # f2(x) = m2*x + b2 = y
    
    # As we got two points by segment, we are able to determine A1, A2, b1 and b2:
    dx1 = x1[1]-x1[0]
    if dx1==0:
        m1 = np.inf
        b1 = 0
    else:
        m1 = (y1[1]-y1[0])/(x1[1]-x1[0])  # Pay attention to not dividing by zero
        b1 = y1[0]-m1*x1[0]        
    dx2 = x2[1]-x2[0]
    if dx2==0:
        m2 = np.inf
        b2 = 0
    else:
        m2 = (y2[1]-y2[0])/(x2[1]-x2[0])  # Pay attention to not dividing by zero
        b2 = y2[0]-m2*x2[0]
    
    # If the segments are parallel, then m1 == m2:
    if (m1 == m2) and (b1 != b2):
        return False  # Parallel segments
    
    # If the segments are parallel and on top of each other, the m1==m2 and b1==b2
    # we've already required that the x-ranges (abcissas) overlap
    if (m1 == m2) and (b1 == b2):
        return True   # parallel segments on top of each other
    
    # A point (Xa,Ya) standing on both lines must satisfy both formulas f1 and f2:
    # Ya = m1 * Xa + b1
    # Ya = m2 * Xa + b2
    # A1 * Xa + b1 = m2 * Xa + b2

    # Line segment 1 is vertical line 
    if x1[0]==x1[1]:
        Xa = x1[0]
        Ya = m2*Xa+b2
        if rangeoverlap(x1,[Xa]) and rangeoverlap(x2,[Xa]) and \
           rangeoverlap(y1,[Ya]) and rangeoverlap(y2,[Ya]):
            return True
        else:
            return False
    # Line semgent 2 is vertical line
    elif x2[0]==x2[1]:
        Xa = x2[0]
        Ya = m1*Xa+b1
        if rangeoverlap(x1,[Xa]) and rangeoverlap(x2,[Xa]) and \
           rangeoverlap(y1,[Ya]) and rangeoverlap(y2,[Ya]):        
            return True
        else:
            return False
    # Neither are vertical lines
    else:
        Xa = (b2 - b1) / (m1 - m2)
        
    # The last thing to do is check that Xa is included into Ia:
    if ( (Xa < max( np.min(x1), np.min(x2) )) or
         (Xa > min( np.max(x1), np.max(x2) )) ):
        return False  # intersection is out of bound
    else:
        return True

    
# from astroML
def crossmatch(X1, X2, max_distance=np.inf,k=1):
    """Cross-match the values between X1 and X2

    By default, this uses a KD Tree for speed.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)
    max_distance : float (optional)
        maximum radius of search.  If no point is within the given radius,
        then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The distance and index of the closest point in X2 to each point in X1
        Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    N1, D = X1.shape
    N2, D2 = X2.shape

    if D != D2:
        raise ValueError('Arrays must have the same second dimension')

    kdt = cKDTree(X2)

    dist, ind = kdt.query(X1, k=k, distance_upper_bound=max_distance)

    return dist, ind

# from astroML, modified by D. Nidever
def xmatch(ra1, dec1, ra2, dec2, dcr=2.0,unique=False,sphere=True):
    """Cross-match angular values between RA1/DEC1 and RA2/DEC2

    Find the closest match in the second list for each element
    in the first list and within the maximum distance.

    By default, this uses a KD Tree for speed.  Because the
    KD Tree only handles cartesian distances, the angles
    are projected onto a 3D sphere.

    This can return duplicate matches if there is an element
    in the second list that is the closest match to two elements
    of the first list.

    Parameters
    ----------
    ra1/dec1 : array_like
        first dataset, arrays of RA and DEC
        both measured in degrees
    ra2/dec2 : array_like
        second dataset, arrays of RA and DEC
        both measured in degrees
    dcr : float (optional)
        maximum radius of search, measured in arcsec.
        This can be an array of the same size as ra1/dec1.
    unique : boolean, optional
        Return unique one-to-one matches.  Default is False and
           allows duplicates.
    sphere : boolean, optional
        The coordinates are spherical in degrees.  Otherwise, the dcr
          is assumed to be in the same units as the input values.
          Default is True.


    Returns
    -------
    ind1, ind2, dist: ndarrays
        The indices for RA1/DEC1 (ind1) and for RA2/DEC2 (ind2) of the
        matches, and the distances (in arcsec).
    """
    X1 = np.vstack((ra1,dec1)).T
    X2 = np.vstack((ra2,dec2)).T

    # Spherical coordinates in degrees
    if sphere:
        X1 = X1 * (np.pi / 180.)
        X2 = X2 * (np.pi / 180.)
        if utils.size(dcr)>1:
            max_distance = (np.max(dcr) / 3600) * (np.pi / 180.)
        else:
            max_distance = (dcr / 3600) * (np.pi / 180.)

        # Convert 2D RA/DEC to 3D cartesian coordinates
        Y1 = np.transpose(np.vstack([np.cos(X1[:, 0]) * np.cos(X1[:, 1]),
                                     np.sin(X1[:, 0]) * np.cos(X1[:, 1]),
                                     np.sin(X1[:, 1])]))
        Y2 = np.transpose(np.vstack([np.cos(X2[:, 0]) * np.cos(X2[:, 1]),
                                     np.sin(X2[:, 0]) * np.cos(X2[:, 1]),
                                     np.sin(X2[:, 1])]))

        # law of cosines to compute 3D distance
        max_y = np.sqrt(2 - 2 * np.cos(max_distance))
        k = 1 if unique is False else 10
        dist, ind = crossmatch(Y1, Y2, max_y, k=k)
    
        # convert distances back to angles using the law of tangents
        not_inf = ~np.isinf(dist)
        x = 0.5 * dist[not_inf]
        dist[not_inf] = (180. / np.pi * 2 * np.arctan2(x,
                                np.sqrt(np.maximum(0, 1 - x ** 2))))
        dist[not_inf] *= 3600.0      # in arcsec
    # Regular coordinates
    else:
        k = 1 if unique is False else 10
        dist, ind = crossmatch(X1, X2, np.max(dcr), k=k)
        not_inf = ~np.isinf(dist)
            
    # Allow duplicates
    if unique is False:

        # no matches
        if np.sum(not_inf)==0:
            return [], [], [np.inf]
        
        # If DCR is an array then impose the max limits for each element
        if utils.size(dcr)>1:
            bd,nbd = utils.where(dist > dcr)
            if nbd>0:
                dist[bd] = np.inf
                not_inf = ~np.isinf(dist)
    
        # Change to the output that I want
        ind1 = np.arange(len(ra1))[not_inf]
        ind2 = ind[not_inf]
        mindist = dist[not_inf]

    # Return unique one-to-one matches
    else:

        # no matches
        if np.sum(~np.isinf(dist[:,0]))==0:
            return [], [], [np.inf]
        
        done = 0
        niter = 1
        # Loop until we converge
        while (done==0):

            # If DCR is an array then impose the max limits for each element
            if utils.size(dcr)>1:
                bd,nbd = utils.where(dist[:,0] > dcr)
                if nbd>0:
                    for i in range(nbd):
                        dist[bd[i],:] = np.inf

            # no matches
            if np.sum(~np.isinf(dist[:,0]))==0:
                return [], [], [np.inf]

            # closest matches
            not_inf1 = ~np.isinf(dist[:,0])
            ind1 = np.arange(len(ra1))[not_inf1]
            ind2 = ind[:,0][not_inf1]
            mindist = dist[:,0][not_inf1]
            if len(ind2)==0:
                return [], [], [np.inf]
            index = utils.create_index(ind2)
            # some duplicates to deal with
            bd,nbd = utils.where(index['num']>1)
            if nbd>0:
                torem = []            
                for i in range(nbd):
                    indx = index['index'][index['lo'][bd[i]]:index['hi'][bd[i]]+1]
                    # keep the one with the smallest minimum distance
                    si = np.argsort(mindist[indx])
                    if index['num'][bd[i]]>2:
                        torem += list(indx[si[1:]])    # add list
                    else:
                        torem.append(indx[si[1:]][0])  # add single element
                ntorem = utils.size(torem)
                # For each object that was "removed" and is now unmatched, check the next possible
                # match and move it up in the dist/ind list if it isn't INF
                for i in range(ntorem):
                    # There is a next possible match 
                    if ~np.isinf(dist[torem[i],niter-1]):
                        ind[torem[i],:] = np.hstack( (ind[torem[i],niter:].squeeze(), np.repeat(-1,niter)) )
                        dist[torem[i],:] = np.hstack( (dist[torem[i],niter:].squeeze(), np.repeat(np.inf,niter)) )
                    # All INFs
                    else:
                        ind[torem[i],:] = -1
                        dist[torem[i],:] = np.inf
            else:
                ntorem = 0

            niter += 1
            # Are we done, no duplicates or hit the maximum 10
            if (ntorem==0) | (niter>=10): done=1
                                
    return ind1, ind2, mindist

def dist(x1, y1, x2, y2):
    """ Calculate Euclidian distance between two sets of points."""

    if (utils.size(x1) != utils.size(y1)):
        raise ValueError('x1/y1 must have same number of elements')
    if (utils.size(x2) != utils.size(y2)):
        raise ValueError('x2/y2 must have same number of elements')    

    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
    

def sphdist(lon1, lat1, lon2, lat2):
    """Calculate the angular distance between two sets of points.

    Parameters
    ----------
    lon1/lat1 : scalar or array_like
        first dataset, arrays of LON/LAT or RA/DEC
        both measured in degrees
    lon2/lat2 : scalar array_like
        second dataset, arrays of LON/LAT or RA/DEC
        both measured in degrees

    Returns
    -------
    dist: ndarrays
        The angular distance in degrees.
    """

    if (utils.size(lon1) != utils.size(lat1)):
        raise ValueError('lon1/lat1 must have same number of elements')
    if (utils.size(lon2) != utils.size(lat2)):
        raise ValueError('lon2/lat2 must have same number of elements')    
    
    # From this website;
    # http://www2.sjsu.edu/faculty/watkins/sphere.htm

    cosa = np.sin(np.deg2rad(lat1))*np.sin(np.deg2rad(lat2))+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.cos(np.deg2rad(lon1-lon2))
    dist = np.rad2deg(np.arccos(cosa))

    return dist

def lbd2xyz(l,b,d,R0=8.5):
    """
    Convert from LON, LAT and DISTANCE to galactocentric
    cartesian coordinates.

    The L,B and D (distance) are all with respect to the sun
    L and B need to be in degrees and D in kpc.
    The X, Y and Z coordinates are with respect to the Galactic Center
    Z is going up, towards b=90
    X is towards the galactic center from our direction, L=0
    Y is towards the L=90 direction

    Parameters:
    -----------
    l    Galactic longitude in degrees
    b    Galactic latitude in degrees
    d    Distance from sun in kpc
    =R0  Distance of the sun to the galactic center. 8.5 kpc is the default

    Returns:
    -------
    X  The galactocentric cartesian X coordinate in kpc.  Positive X is
          towards the Galactic center or L=0.
    Y  The galactocentric cartesian Y coordinate in kpc.  Positive Y is
          towards L=90.
    Z  The galactocentric cartesian Z coordinate in kpc.  Positive Z is
          towards the North Galactic pole.
    """

    brad = np.deg2rad(np.atleast_1d(b).copy().astype(np.float64))
    lrad = np.deg2rad(np.atleast_1d(l).copy().astype(np.float64))
    dd = np.atleast_1d(d).copy().astype(np.float64)

    x = dd*np.sin(0.5*np.pi-brad)*np.cos(lrad)-R0
    y = dd*np.sin(0.5*np.pi-brad)*np.sin(lrad)
    z = dd*np.cos(0.5*np.pi-brad)

    return x,y,z


def xyz2lbd(x,y,z,R0=8.5):
    """ Convert galactocentric X/Y/Z coordinates to l,b,dist."""

    rho = np.sqrt( (x+R0)**2 + y**2)
    lrad = np.arctan2(y,x+R0)
    brad = 0.5*np.pi - np.arctan2(rho,z)      # this is more straighforward

    brad[brad > 0.5*np.pi] -= np.pi
    brad[brad < -0.5*np.pi] += np.pi

    # This doesn't work if z=0
    #if cos(0.5*!dpi-brad) ne 0.0 then d = zz/cos(0.5*!dpi-brad)
    #if cos(0.5*!dpi-brad) eq 0.0 then d = abs(zz)
    d = np.sqrt( (x+R0)**2 + y**2 + z**2 )
    b = np.rad2deg(brad)
    l = np.rad2deg(lrad)
    l = l % 360

    return l,b,d


class MagellanicStream(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the Magellanic Stream
    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    MSLongitude : `Angle`, optional, must be keyword
        The longitude-like angle corresponding to the Magellanic Stream.
    MSLatitude : `Angle`, optional, must be keyword
        The latitude-like angle corresponding to the Magellanic Stream.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the Stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The radial velocity of this object.

    Developed by Y. Choi

    """
    default_representation = coord.SphericalRepresentation

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'MSLongitude'),
            coord.RepresentationMapping('lat', 'MSLatitude'),
            coord.RepresentationMapping('distance', 'distance')]#,
    }
    frame_specific_representation_info[coord.UnitSphericalRepresentation] = frame_specific_representation_info[coord.SphericalRepresentation]

MS_PHI = (180 + 8.5 + 90) * u.degree # Euler angles (from Nidever 2010)
MS_THETA = (90 + 7.5) * u.degree
MS_PSI = -32.724214217871349 * u.degree  # anode parameter from gal2mag.pro

D = rotation_matrix(MS_PHI, "z")
C = rotation_matrix(MS_THETA, "x")
B = rotation_matrix(MS_PSI, "z")
A = np.diag([1., 1., 1.])
MS_MATRIX = matrix_product(A, B, C, D)

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, MagellanicStream)
def galactic_to_MS():
    """ Compute the transformation matrix from Galactic spherical to
        Magellanic Stream coordinates.
    """
    return MS_MATRIX

@frame_transform_graph.transform(coord.StaticMatrixTransform, MagellanicStream, coord.Galactic)
def MS_to_galactic():
    """ Compute the transformation matrix from Magellanic Stream coordinates to
        spherical Galactic.
    """
    return matrix_transpose(MS_MATRIX)

#c_icrs = SkyCoord(ra=tmp.ra*u.degree, dec=tmp.dec*u.degree)
#c_ms = c_icrs.transform_to(MagellanicStream)
#ms_l,ms_b = c_ms.MSLongitude.degree, c_ms.MSLatitude.degree #subtract off 360 from ms_l

def gal2mag(glon,glat):
    """ Convert Galactic longitude/latitude to Magellanic Stream longitude/latitude."""
    c_glon = SkyCoord(glon,glat,frame='galactic',unit='deg')
    c_ms = c_glon.transform_to(MagellanicStream)
    mlon,mlat = c_ms.MSLongitude.degree, c_ms.MSLatitude.degree #subtract off 360 from ms_l
    return mlon,mlat

def mag2gal(mlon,mlat):
    """ Convert Magellanic Stream longitude/latitude to Galactic longitude/latitude."""
    c_ms = SkyCoord(mlon,mlat,frame=MagellanicStream,unit='deg')
    c_glon = c_ms.transform_to('galactic')
    glon,glat = c_glon.l.degree, c_glon.b.degree #subtract off 360 from ms_l
    return glon,glat

def wcsfit(wcs,tab,verbose=False):
    """
    Fit the WCS using a catalog of stars with known X/Y and RA/DEC coordinates.

    Parameters
    ----------
    wcs : WCS object
      The WCS object with an estimate of the WCS.
    tab : table
      Catalog with x, y, ra, and dec of matched sources.
    verbose : boolean, optional
      Print informaton to the screen.  Default is False.

    Returns
    -------
    fwcs : WCS object
      WCS with improved paramters.
    
    Example
    -------

    fwcs = wcsfit(wcs,tab)L

    """

    if len(tab)==0:
        raise ValueError('Input catalog has no stars')
    
    if verbose:
        print('Fitting WCS with '+str(len(tab))+' stars')
    
    coo = SkyCoord(ra=tab['ra'],dec=tab['dec'],unit='deg')

    def newwcs(pars):
        # pars = [CRVAL1,CRVAL2,CDELT1,CDELT2,PC1_1,PC1_2,PC2_1,PC2_2]
        # pars = [delta_ra (arcsec), delta_dec (arcsec), cdelt1_scale_change (multiplicative)
        #  cdelt2_scale_change (multiplicate), rotation (deg)]
        twcs = copy.deepcopy(wcs)
        twcs.wcs.crval[0] += pars[0]/3600.0
        twcs.wcs.crval[1] += pars[1]/3600.0  
        twcs.wcs.cdelt[0] *= pars[2]
        twcs.wcs.cdelt[1] *= pars[3]
        # Rotation matrix
        # R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        rot = np.array([[np.cos(np.deg2rad(pars[4])),-np.sin(np.deg2rad(pars[4]))],
                        [np.sin(np.deg2rad(pars[4])),np.cos(np.deg2rad(pars[4]))] ])
        newpc = np.dot(twcs.wcs.pc,rot)
        twcs.wcs.pc = newpc
        return twcs
    def diffcoords(pars,good=None):
        twcs = newwcs(pars)
        try:
            vcoo = twcs.pixel_to_world(tab['x'],tab['y'])
            diff = coo.separation(vcoo)
            if good is not None:
                meddiff = np.nanmean(diff[good].arcsec)
            else:
                meddiff = np.nanmean(diff.arcsec)
        except:
            meddiff = 999999.
        return meddiff

    def diffcoords_jac(pars):
        delta = [0.2,0.2,0.1,0.1,0.5]
        jac = np.zeros(5,float)
        y0 = diffcoords(pars)
        for i in range(len(pars)):
            tpars = np.array(pars).copy()
            tpars[i] += delta[i]
            y1 = diffcoords(tpars)
            jac[i] = (y1-y0)/delta[i]
        return y0,jac
            
    
    # fit delta_ra (arcsec), delta_dec (arcsec), cdelt1_scale_change (multiplicative)
    #  cdelt2_scale_change (multiplicate), rotation (deg)]


    # Get initial guess of delta_ra and delta_dec using the coordinate ra/dec values
    # and initial WCS coordinates
    vcoo = wcs.pixel_to_world(tab['x'],tab['y'])
    vra = vcoo.ra.deg
    vdec = vcoo.dec.deg
    dra = np.median(tab['ra']-vra)*3600
    ddec = np.median(tab['dec']-vdec)*3600

    # Estimate the rotation
    vx,vy = wcs.world_to_pixel(coo)
    coefx,absdevx = ladfit.ladfit(tab['y'],tab['x']-vx)
    coefy,absdevy = ladfit.ladfit(tab['x'],tab['y']-vy)    
    rot = np.mean([coefx[1],-coefy[1]])

    # Do the fit
    estimates = [dra,ddec,1.0,1.0,rot]    
    bounds = len(estimates)*[[-np.inf,np.inf]]
    res1 = minimize(diffcoords,estimates,bounds=bounds)
    pars1 = res1.x
    
    # Remove outliers and refit
    twcs = newwcs(pars1)
    vcoo = twcs.pixel_to_world(tab['x'],tab['y'])
    diff = coo.separation(vcoo)
    meddiff = np.nanmedian(diff.arcsec)
    sigdiff = utils.mad(diff.arcsec)
    good, = np.where(diff.arcsec < (meddiff+3.5*sigdiff))
    noutlier = len(tab)-len(good)
    if noutlier>0:
        if verbose:
            print('Rejecting '+str(noutlier)+' outlier(s)')
        res = minimize(diffcoords,pars1,args=good,bounds=bounds)
        pars = res.x
    else:
        res = res1
        pars = pars1
                  
    fwcs = newwcs(pars)
    
    if verbose:
        print('--- Original WCS ---')
        print(wcs)
        print(' ')
        print('--- Final WCS ---')
        print(fwcs)
        print(' ')
        resid0 = diffcoords([0.0,0.0,1.0,1.0,0.0])
        resid = diffcoords(pars,good)
        print('Original mean residuals: {:.3f} arcsec'.format(resid0))
        print('Final mean residuals   : {:.3f} arcsec'.format(resid))
        
    return fwcs
