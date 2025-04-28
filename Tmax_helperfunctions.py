import numpy as np
import healpy as hp

import pandas as pd

# https://en.wikipedia.org/wiki/Haversine_formula
delta=np.pi/2
def smoothing_arg1(b1,l1,b2,l2):
    """Get the distance between two coordiantes on a unit sphere"""
    ###Formula for colatitude
    hav_theta = np.sin((b2-b1)/2)**2 + np.sin(b1)*np.sin(b2)*np.sin((l2-l1)/2)**2
    distance =2*np.arcsin(np.sqrt(hav_theta))
    return distance

def smoothing_arg(theta1,phi1,theta2,phi2):
    print(theta1,phi1,theta2,phi2)
    distance = (np.cos(theta1)*np.cos(phi1)-np.cos(theta2)*np.cos(phi2))**2 + \
               (np.cos(theta1)*np.sin(phi1)-np.cos(theta2)*np.sin(phi2))**2 + \
               (np.sin(theta1)-np.sin(theta2))**2
    distance = 2*np.arcsin(np.abs(np.sqrt(distance))/2)
    return distance

def window_function(b1,l1,b2,l2,delta):
    """Smoothing Window function"""
    # b1 = b1-np.pi/2
    window = 1.0/(np.sqrt(2*np.pi)*delta)*np.exp(-smoothing_arg1(b1,l1,b2,l2)**2/(2*delta**2))
    
    return window

def window_function2(b1,l1,b2,l2,delta,nside=64):
    """Top hat smoothing"""
    
    theta = smoothing_arg1(b1,l1,b2,l2)

    return np.where(theta < delta, 1, 0)

def window_function3(b1,l1,b2,l2,delta,nside=64):
    """Gaussian + tophat"""
    
    theta = smoothing_arg1(b1,l1,b2,l2)
    window = 1.0/(np.sqrt(2*np.pi)*delta)*np.exp(-theta**2/(2*delta**2))
    window = window * np.where(theta < delta, 1, 0)
    return window

def Qvalue(q_error_norm,theta1,phi1,theta2,phi2,delta,smoothing):
    
    if smoothing == "gaussian":
        window_mat = window_function(theta1,phi1,theta2,phi2,delta)
    elif smoothing == "tophat":
        window_mat = window_function2(theta1,phi1,theta2,phi2,delta,nside=64)
    elif smoothing == "gaussiantophat":
        window_mat = window_function3(theta1,phi1,theta2,phi2,delta,nside=64)

    qvalue = np.dot(q_error_norm,window_mat)
    return qvalue   

def Q_measure(q_error_norm,theta1,phi1,delta,smoothing="gaussian"):
    """Calculation of the measure Q"""

    phi2,theta2 =phi1,theta1
    vec = hp.ang2vec(theta = theta2,phi = phi2)
    
    theta2_antipodal,phi2_antipodal = hp.vec2ang(-vec) 
    
    Qmeasure = pd.DataFrame({'theta2':theta2.flatten(),'phi2':phi2.flatten(),
                             'theta2_antipodal':theta2_antipodal.flatten(),'phi2_antipodal':phi2_antipodal.flatten()})
    Qmeasure['Q'] = Qmeasure.apply(lambda x : Qvalue(q_error_norm,theta1,phi1,x['theta2'],x.phi2,delta,smoothing=smoothing) 
                                   ,axis=1)
    Qmeasure['Q_d']= Qmeasure.apply(lambda x : Qvalue(q_error_norm,theta1,phi1,x['theta2_antipodal'],x.phi2_antipodal,delta,smoothing),axis=1)
                                
    Qmeasure['delta_Qd']= Qmeasure.apply(lambda x : (x.Q-x.Q_d),axis=1)
    
    return Qmeasure

