import numpy as np

import healpy as hp

from astropy.coordinates.angles import Angle
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic
from astropy import units as u

from helperfunctions import *
from dipolefunctions_CatWISE import *

def make_galmask(nside=256, planecut=30) :
    
    """
    Computes a Galactic plane mask
    """
    
    mask = np.ones(hp.nside2npix(nside))
    vector = hp.ang2vec(0,90,lonlat=1)
    indices = hp.query_disc(nside,vector,np.deg2rad(90+planecut))
    mask[indices] = 0
    indices = hp.query_disc(nside,vector,np.deg2rad(90-planecut))
    mask[indices] = 1
    
    return mask

def make_eclmask(nside=256, planecut=30) :
    
    """
    Computes a Galactic plane mask in Ecliptic coordinates
    """
    
    mask = np.ones(hp.nside2npix(nside))
    lon,lat = 0,90
    ra,dec = GalactictoEquatorial(lon,lat)
    C = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    E = C.transform_to('barycentricmeanecliptic')
    lonecl,latecl = E.lon.value,E.lat.value
    vector = hp.ang2vec(lonecl,latecl,lonlat=True)
    indices = hp.query_disc(nside,vector,np.deg2rad(90+planecut))
    mask[indices] = 0
    indices = hp.query_disc(nside,vector,np.deg2rad(90-planecut))
    mask[indices] = 1
    
    return mask

def make_supergalmask(nside=256, planecut=10) :
    
    """
    Computes a Supergalactic plane mask
    """
    
    mask = np.ones(hp.nside2npix(nside))
    slon,slat = 0,90
    S = SkyCoord(slon*u.deg, slat*u.deg, frame='supergalactic')
    G = S.transform_to('galactic')
    lon,lat = G.l.value,G.b.value
    vector = hp.ang2vec(lon,lat,lonlat=True)
    indices = hp.query_disc(nside,vector,np.deg2rad(90+planecut))
    mask[indices] = 0
    indices = hp.query_disc(nside,vector,np.deg2rad(90-planecut))
    mask[indices] = 1
    
    return mask

def makeMask(psmasks,nside=256,galcut=0,masking='symmetric',ecliptic=False,nops=False,factor=1.) :
    
    """
    Computes a mask given a file that specifies locations and extent of point sources
    """
    
    mask = np.ones(hp.nside2npix(nside))
    pixels = np.arange(hp.nside2npix(nside))
    mask_lon,mask_lat = hp.pix2ang(nside,pixels,lonlat=True)

    cmasks = psmasks[(psmasks['pa']<=2)*(psmasks['radius']<15)]
    emasks = psmasks[(psmasks['pa']>2)]
    
    if nops :
        cmasks = psmasks[(psmasks['pa']<=2)*(psmasks['radius']>5)]
        emasks = psmasks[(psmasks['pa']>2)*(psmasks['radius']>5)]
        
    cmask_lon,cmask_lat,cmask_rad = *EquatorialtoGalactic(cmasks['ra'],cmasks['dec']),cmasks['radius']
    
    if ecliptic :
        Cmask = SkyCoord(cmasks['ra']*u.deg, cmasks['dec']*u.deg, frame='icrs')
        Emask = Cmask.transform_to('barycentricmeanecliptic')
        cmask_lon,cmask_lat = Emask.lon.value,Emask.lat.value
    
    for lon,lat,radius in zip(cmask_lon,cmask_lat,cmask_rad):
        vector = hp.ang2vec(lon,lat,lonlat=True)
        indices = hp.query_disc(nside,vector,factor*np.deg2rad(radius))
        mask[indices] = 0
        if masking=='symmetric' :
            indices = hp.query_disc(nside,-vector,factor*np.deg2rad(radius))
            mask[indices] = 0
    
    emask_lon,emask_lat,emask_rad,emask_ba,emask_pa = *EquatorialtoGalactic(emasks['ra'],emasks['dec']),emasks['radius'], emasks['ba'], emasks['pa']
    
    if ecliptic :
        Cmask = SkyCoord(emasks['ra']*u.deg, emasks['dec']*u.deg, frame='icrs')
        Emask = Cmask.transform_to('barycentricmeanecliptic')
        emask_lon,emask_lat = Emask.lon.value,Emask.lat.value

    for lon,lat,rad,ba,pa in zip(emask_lon,emask_lat,emask_rad,emask_ba,emask_pa) :
        ell = evaluateEllipse(mask_lon,mask_lat,lon,lat,factor*rad,ba,pa)
        mask[ell] = 0
        if masking=='symmetric' :
            ell = evaluateEllipse(mask_lon,mask_lat,lon+180.,-1.*lat,factor*rad,ba,-1.*pa)
            mask[ell] = 0
    
    if galcut : 
        planemask = make_galmask(nside=nside,planecut=galcut)
        if ecliptic :
            planemask = make_eclmask(nside=nside,planecut=galcut)
    else : planemask = np.ones_like(mask)
    
    return mask*planemask