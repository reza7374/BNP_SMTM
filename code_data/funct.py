import numpy as np
import sys
from scipy.stats import gamma as gam
from scipy.stats import norm as NORM
from scipy.stats import invgamma
import matplotlib.pyplot as plt

from math import gamma
from numpy.core.numeric import zeros_like
from numpy import sqrt, cos, sin, meshgrid, mod, exp,size, zeros, pi,arange, array,arctan2,ceil,zeros_like,shape, reshape, cumsum, log
from numpy.random import multivariate_normal, rand, normal,poisson
from scipy.spatial.distance import cdist
import datetime
from matplotlib import gridspec


def change_Iteration(it, iteration):
    if it==3000:
        iteration = 30
    elif it==30000:
        iteration=35
    elif it==60000:
        iteration=40
    return iteration
def calculate_log_post(Data, chan, psf, Struct, lpr_bg, lpr_i, lpr_d, bnp, LPrior_A, LPrior_Phi, lp):
    gm = bnp['gm']
    loads = chan['loads']
    m = loads.shape[0]
    X = chan['X']; Y = chan['Y']; Z = chan['Z'];
    D = chan['D']; Dt = bnp['Dt'];
    Sig_D = np.sqrt(2*D*Dt); SigX = Struct['NPix']*Struct['PixelSize']/2
    DLPrior1X = np.sum(log(normpdf(X[0,:],0,SigX)))
    DLPrior1Y = np.sum(log(normpdf(Y[0,:],0,SigX)))
    DLPrior1Z = np.sum(log(normpdf(Z[0,:],0,SigX)))
    DLPriorX = np.sum(log(normpdf(X[1:,:],X[:-1,:],Sig_D)))
    DLPriorY = np.sum(log(normpdf(Y[1:,:],Y[:-1,:],Sig_D)))
    DLPriorZ = np.sum(log(normpdf(Z[1:,:],Z[:-1,:],Sig_D)))
    Lpr_traj = DLPrior1X + DLPrior1Y + DLPrior1Z + DLPriorX + DLPriorY + DLPriorZ
    l_like = np.sum(Data*(np.log(psf)) - psf)
    l_post = l_like + lpr_bg + lpr_i + Lpr_traj + lpr_d + LPrior_A + LPrior_Phi

    return l_post
def tempreture(it, tp):
    if it//3000==it/3000:
        if tp>20000:
            tp /= 2
        elif tp>10000:
            tp-=4000
        elif tp>2000:
            tp-=2000
        elif tp>1000:
            tp-=500
        else:
            tp=1
    if tp==0:
        t=1

    return tp
def normpdf(x, mu, sigma):
    rv = NORM(mu, sigma)
    pdf = rv.pdf(x)
    return pdf
def benolipdf(X, M, G):
    param = 1/(1+((M-1)/G))
    pdf = np.zeros_like(X)
    for xi in range(len(X)):
        if X[xi]==0:
            pdf[xi] = 1-param
        elif X[xi]==1:
            pdf[xi] = param
    return pdf
def gampdf(x, k, theta):
    rv=gam(a=k,scale=theta)
    pdf = rv.pdf(x)
    return pdf
def Invgampdf(x, a, b):
    y = b**a/gamma(a)*x**(-a-1)*exp(-b/x)
    return y
def calCOV(Xg,Yg, T, L, Kernel= 'Exponential'):
      Xg = np.array(Xg).flatten()
      Yg = np.array(Yg).flatten()
      X1 = np.array([Xg,Yg]).transpose()
      Dist = cdist(X1,X1)
      if Kernel=='Exponential':
        K =(T**2)*exp(-Dist**2/L**2/2);
      elif Kernel=='Quadratic':
        K =(T**2)*(1+Dist**2/L**2/2)
      return K

def sampleDiff(Chain, DelX, Dt):
    """
    Calculates the diffusion coefficient and its log-prior probability given a Chain, DelX, and Dt.

    Args:
        Chain (dict): Dictionary containing information about the chain, including 'loads', 'X', 'Y', and 'Z'.
        DelX (numpy.ndarray): Numpy array of shape (n,3) representing the displacement of the particle in 3D space over time.
        Dt (float): Time interval.

    Returns:
        D (float): Diffusion coefficient calculated based on the provided information.
        LPrior_D (float): Log-prior probability of D calculated based on the provided information.
    """
    
    # Get loads, AlphaPrior, and BetaPrior from Chain
    loads = Chain['loads']
    AlphaPrior = 13
    BetaPrior = 1200
    
    # Check if any particle is active
    if np.sum(loads)>0:
        
        # Get X, Y, and Z from Chain
        X = Chain['X']; Y = Chain['Y']; Z = Chain['Z']
        
        # Calculate Alpha using R, NN, and size of X
        R = np.size(DelX,axis =0)
        NN = np.sum(loads)
        Alpha = (3/2)*R*NN*(np.size(X,axis=0)-1) + AlphaPrior
        
        # Calculate Beta using R, X, Y, Z, and Dt
        Beta = BetaPrior
        for nn in [nn for nn in range(size(Z,axis=1)) if loads[nn]!=0]:
            Beta += (R*np.sum((X[1:,nn]-X[:-1,nn])**2 +(Y[1:,nn]-Y[:-1,nn])**2 + (Z[1:,nn]-Z[:-1,nn])**2))/(4*Dt)
        
        # Generate D using a gamma distribution with Alpha and Beta values
        D = 1/np.random.gamma(shape=Alpha, scale =1/Beta)
        
        # Calculate log-prior probability of D
        LPrior_D = np.log(Invgampdf(D,  AlphaPrior, BetaPrior))
    
    # If there is not active particle, use AlphaPrior and BetaPrior to calculate D
    else:
        D = D = 1/np.random.gamma(shape=AlphaPrior, scale =1/BetaPrior)
        LPrior_D = np.log(Invgampdf(D, AlphaPrior, BetaPrior))
    
    # Return D and its log-prior probability LPrior_D
    return D, LPrior_D
def sampleIntensity(Data, Chain, PSFstack, Norm1_PSFstack, DelX, AcceptI, tp):
    """
    Samples the intensity of the PSF and updates the PSF stack accordingly.

    Parameters:
        Data (numpy.ndarray): The data array.
        Chain (dict): The dictionary containing the current values of the parameters.
        PSFstack (numpy.ndarray): The array of PSFs.
        Norm1_PSFstack (numpy.ndarray): The array of normalized PSFs.
        DelX (numpy.ndarray): The array of displacements.
        AcceptI (int): The acceptance rate for the intensity.
        tp (float): The tempreture.

    Returns:
        float: The updated value of the intensity.
        numpy.ndarray: The updated PSF stack.
        int: The updated acceptance rate.
        float: The logarithm of the prior probability distribution.
    """
    # Extract current parameter values
    Bg = Chain['Bg']
    I = Chain['I']

    # Set proposal distribution parameters
    Alpha_Prop = 5000
    Alpha = 8000
    Beta = 0.5

    # Sample from the proposal distribution
    tI = gam.rvs(a=Alpha_Prop, scale=I/Alpha_Prop)

    # Update the PSF stack using the new intensity
    tPSF = []
    for ii in range(np.size(DelX, axis=0)):
        tPSF.append(tI*Norm1_PSFstack[ii,:,:,:] + Bg[ii])
    tPSF = np.array(tPSF)

    # Compute the log-likelihood, log-prior, and log-proposal probabilities
    DLogL = np.sum(Data*(np.log(tPSF)-np.log(PSFstack))-(tPSF-PSFstack))/tp
    DLogPrior = log(gam.pdf(tI, a=Alpha, scale=Beta)) - log(gam.pdf(I, a=Alpha, scale=Beta))
    DLogProp = log(gampdf(I, Alpha_Prop, tI/Alpha_Prop)) - log(gampdf(tI, Alpha_Prop, I/Alpha_Prop))
    DLogPost = DLogL + DLogPrior + DLogProp

    # Accept or reject the new proposal
    if DLogPost > log(np.random.rand()):
        I = tI
        PSFstack = tPSF
        AcceptI = AcceptI + 1

    # Compute and return the log-prior probability
    LogPr = np.log(gampdf(I, Alpha, Beta))
    return I, PSFstack, AcceptI, LogPr
def sampleBg(Data, Chain, PSFstack, Norm1_PSFstack, DelX, AcceptBg, tp):
    """
    Samples the background intensity from a gamma distribution using the Metropolis-Hastings algorithm.

    Args:
    - Data (np.array): a 3D array of data with dimensions (foclaplane, x, y, t)
    - Chain (dict): a dictionary containing the current background intensity and object intensity chains
    - PSFstack (np.array): a 4D array of PSFs with dimensions (n, x, y, t)
    - Norm1_PSFstack (np.array): a 4D array of normalized PSFs with dimensions (n, x, y, t)
    - DelX (np.array): a 2D array of spatial offsets between the PSFs and the data with dimensions (n, 3)
    - AcceptBg (int): the number of accepted proposals for the background intensity
    - tp (float): the tempreture

    Returns:
    - Bg (np.array): the updated background intensity chain
    - PSFstack (np.array): the updated PSFstack based on the new background intensity
    - AcceptBg (int): the updated number of accepted proposals for the background intensity
    - LogPrior (float): the log prior probability of the new background intensity
    """

    Bg = Chain['Bg']
    I = Chain['I']
    Alpha_Prop = 10000
    Alpha = 1.1
    Beta = 120
    tBg = gam.rvs(a=Alpha_Prop, scale=Bg/Alpha_Prop)
    tPSF = []

    # Calculate the new PSFstack based on the new background intensity
    for ii in range(np.size(DelX, axis=0)):
        tPSF.append(I*Norm1_PSFstack[ii,:,:,:] + tBg[ii])

    tPSF = np.array(tPSF)

    # Calculate the likelihood, prior, proposal, and posterior probabilities
    DLogL = np.sum(Data * (np.log(tPSF) - np.log(PSFstack)) - (tPSF - PSFstack)) / tp
    DLogPrior = np.sum(np.log(gampdf(tBg, Alpha, Beta)) - np.log(gampdf(Bg, Alpha, Beta)))
    DLogProp = np.sum(np.log(gampdf(Bg, Alpha_Prop, tBg/Alpha_Prop)) - np.log(gampdf(tBg, Alpha_Prop, Bg/Alpha_Prop)))
    DLogPost = DLogL + DLogPrior + DLogProp

    # If the posterior probability is greater than a random number drawn from a uniform distribution between 0 and 1,
    # accept the new background intensity
    if DLogPost > np.log(np.random.rand()):
        Bg = tBg.copy()
        PSFstack = tPSF
        AcceptBg = AcceptBg + 1

    LogPrior = np.sum(np.log(gampdf(tBg, Alpha, Beta)))

    return Bg, PSFstack, AcceptBg, LogPrior
def Switch_label(Chain, Struct, BNP, Acceptc):
    """
    Switches the labels of two randomly chosen particles within a randomly chosen frame of a given particle trajectory.
    
    Args:
    Chain (dict): dictionary containing the particle trajectories
    Struct (dict): dictionary containing structural information
    BNP (dict): dictionary containing Bayesian Network parameters
    Acceptc (int): count of accepted proposals

    Returns:
    tuple: A tuple containing the updated particle trajectories X, Y, Z, the updated Acceptc count, and the DLogP value

    """
    # Extracting the necessary variables from the Chain dictionary
    X = Chain['X']; Y = Chain['Y']; Z = Chain['Z']
    loads = Chain['loads']

    # Initialize DLogP and only continue if more than one particle is active
    DLogP = 0
    if np.sum(loads) > 1:

        D = Chain['D']
        Dt = BNP['Dt']
        Sig_D = np.sqrt(2*D*Dt)
        prob = loads/np.sum(loads)
        SigX = Struct['NPix']*Struct['PixelSize']/2
        tcx = np.copy(X)
        tcy = np.copy(Y)
        tcz = np.copy(Z)
        sz = Z.shape

        #Choose a frame in random
        while True:
            frame = np.sort(np.random.choice(sz[0], size=2, replace=False))
            if frame[1]-frame[0] < sz[0]:
                break

        #Select 2 particle from the list of active particle (random)
        n_particle = np.random.choice(len(loads), size=2, p=prob, replace=False)

        #Switch label
        tcx[frame[0]:frame[1], n_particle[0]] = X[frame[0]:frame[1], n_particle[1]]
        tcy[frame[0]:frame[1], n_particle[0]] = Y[frame[0]:frame[1], n_particle[1]]
        tcz[frame[0]:frame[1], n_particle[0]] = Z[frame[0]:frame[1], n_particle[1]]
        tcx[frame[0]:frame[1], n_particle[1]] = X[frame[0]:frame[1], n_particle[0]]
        tcy[frame[0]:frame[1], n_particle[1]] = Y[frame[0]:frame[1], n_particle[0]]
        tcz[frame[0]:frame[1], n_particle[1]] = Z[frame[0]:frame[1], n_particle[0]]

        #calculate priors
        DLogPrior = 0
        DLogP = 0
        for ni in n_particle:
            DLPrior1X = np.sum(log(normpdf(tcx[0,ni], 0, SigX))-log(normpdf(X[0,ni], 0, SigX)))
            DLPrior1Y = np.sum(log(normpdf(tcy[0,ni], 0, SigX))-log(normpdf(Y[0,ni], 0, SigX)))
            DLPrior1Z = np.sum(log(normpdf(tcz[0,ni], 0, SigX))-log(normpdf(Z[0,ni], 0, SigX)))
            DLPriorX = np.sum(log(normpdf(tcx[1:,ni],tcx[:-1,ni],Sig_D))-log(normpdf(X[1:,ni],X[:-1,ni],Sig_D)))
            DLPriorY = np.sum(log(normpdf(tcy[1:,ni],tcy[:-1,ni],Sig_D))-log(normpdf(Y[1:,ni],Y[:-1,ni],Sig_D)))
            DLPriorZ = np.sum(log(normpdf(tcz[1:,ni],tcz[:-1,ni],Sig_D))-log(normpdf(Z[1:,ni],Z[:-1,ni],Sig_D)))
            kl =  DLPrior1X + DLPrior1Y + DLPrior1Z + DLPriorX + DLPriorY + DLPriorZ
            DLogPrior +=kl
        #accept or reject
        if DLogPrior > log(np.random.rand()):
            X = tcx
            Y = tcy
            Z = tcz
            Acceptc = Acceptc + 1
            DLogP = DLogPrior
    return X, Y, Z, Acceptc, DLogP
def Switch_one_frame(Chain, Struct, BNP, Acceptc):
    """
    Switch the labels of two particles in a single frame of a particle trajectory.

    Parameters:
    -----------
    Chain : dict
        Dictionary containing the particle trajectory coordinates and load indicators.
    Struct : dict
        Dictionary containing structural information about the particle trajectory.
    BNP : dict
        Dictionary containing Bayesian network parameters.
    Acceptc : int
        Number of accepted transitions.

    Returns:
    --------
    X : ndarray
        Updated X coordinate of the particle trajectory.
    Y : ndarray
        Updated Y coordinate of the particle trajectory.
    Z : ndarray
        Updated Z coordinate of the particle trajectory.
    Acceptc : int
        Updated number of accepted transitions.
    DLogP : float
        Difference in log probability between new and old configurations.

    """
    # Extract data from input dictionaries
    X = Chain['X']
    Y = Chain['Y']
    Z = Chain['Z']
    loads = Chain['loads']
    DLogP = 0
    
    # Check if there are at least two active particles
    if np.sum(loads) > 1:
        D = Chain['D']
        Dt = BNP['Dt']
        Sig_D = np.sqrt(2 * D * Dt)
        prob = loads / np.sum(loads)
        SigX = Struct['NPix'] * Struct['PixelSize'] / 2
        tcx = np.copy(X)
        tcy = np.copy(Y)
        tcz = np.copy(Z)
        sz = Z.shape
        
        # Choose a frame in random
        frame = np.random.randint(sz[0])
        
        # Select two particles from the list of active particles (randomly)
        while True:
            n_particle = np.random.choice(len(loads), size=2, p=prob)
            if n_particle[0] != n_particle[1]:
                break
                
        # Switch the labels of the selected particles
        tcx[frame, n_particle[0]] = X[frame, n_particle[1]]
        tcy[frame, n_particle[0]] = Y[frame, n_particle[1]]
        tcz[frame, n_particle[0]] = Z[frame, n_particle[1]]
        tcx[frame, n_particle[1]] = X[frame, n_particle[0]]
        tcy[frame, n_particle[1]] = Y[frame, n_particle[0]]
        tcz[frame, n_particle[1]] = Z[frame, n_particle[0]]
        
        # Calculate priors
        DLogPrior = 0
        DLogP = 0
        for ni in n_particle:
            DLPrior1X = np.sum(log(normpdf(tcx[0, ni], 0, SigX)) - log(normpdf(X[0, ni], 0, SigX)))
            DLPrior1Y = np.sum(log(normpdf(tcy[0, ni], 0, SigX)) - log(normpdf(Y[0, ni], 0, SigX)))
            DLPrior1Z = np.sum(log(normpdf(tcz[0, ni], 0, SigX)) - log(normpdf(Z[0, ni], 0, SigX)))
            DLPriorX = np.sum(log(normpdf(tcx[1:, ni], tcx[:-1, ni], Sig_D)) - log(normpdf(X[1:, ni], X[:-1, ni], Sig_D)))
            DLPriorY = np.sum(log(normpdf(tcy[1:,ni],tcy[:-1,ni],Sig_D))-log(normpdf(Y[1:,ni],Y[:-1,ni],Sig_D)))
            DLPriorZ = np.sum(log(normpdf(tcz[1:,ni],tcz[:-1,ni],Sig_D))-log(normpdf(Z[1:,ni],Z[:-1,ni],Sig_D)))
            kl =  DLPrior1X + DLPrior1Y + DLPrior1Z + DLPriorX + DLPriorY + DLPriorZ
            DLogPrior +=kl
        #accept or reject
        if DLogPrior > log(np.random.rand()):
            X = tcx
            Y = tcy
            Z = tcz
            Acceptc = Acceptc + 1
            DLogP = DLogPrior
    return X, Y, Z, Acceptc, DLogP
def findPSF(Pupil_Mag, Pupil_Phase, Bg, I, loads, DefocusK, Z, Mask, SubPixelZeros, StartInd, EndInd, SubPixel, X, Y, XOffsetPhase='Not Given', YOffsetPhase='Not Given'):
    """
    Computes the Point Spread Function (PSF) for a given set of parameters.

    Args:
        Pupil_Mag (ndarray): Magnitude of the pupil function.
        Pupil_Phase (ndarray): Phase of the pupil function.
        Bg (float): Background level.
        I (float): Intensity scaling factor.
        loads (ndarray): Array indicating active particle.
        DefocusK (float): Defocus scaling factor.
        Z (ndarray): Z-coordinates.
        Mask (ndarray): Binary mask of the pupil.
        SubPixelZeros (ndarray): Array for padding zeros for subpixel resolution.
        StartInd (int): Starting index for subpixel padding.
        EndInd (int): Ending index for subpixel padding.
        SubPixel (int): Number of subpixels for subpixel resolution.
        X (ndarray): X-coordinates.
        Y (ndarray): Y-coordinates.
        XOffsetPhase (float or str, optional): X offset phase. Defaults to 'Not Given'.
        YOffsetPhase (float or str, optional): Y offset phase. Defaults to 'Not Given'.

    Returns:
        tuple: A tuple containing the background PSF and the PSF.
    """
    NPix = np.size(Mask, axis=1)
    PSF = np.zeros((NPix, NPix, size(Z, axis=0), size(Z, axis=1)))
    BgPSF = np.zeros((NPix, NPix, size(Z, axis=0)))
    
    # if XOffsetPhase == 'Not Given' and YOffsetPhase == 'Not Given':
    #     # Compute PSF for the case where no offset phase is given
    #     for nn in [nn for nn in range(size(Z, axis=1)) if loads[nn] == 1]:
    #         for zz in range(size(Z, axis=0)):
    #             DefocusPhase = DefocusK * Z[zz, nn]
    #             Phase = Mask * (DefocusPhase + Pupil_Phase)
    #             # Compute Optical Transfer Function (propagator)
    #             OTF = Mask * Pupil_Mag * np.exp(1j * Phase)
    #             # Parseval normalization
    #             Norm = np.sqrt(np.sum(np.abs(OTF))) * NPix
    #             # Padding zeros for subpixel resolution
    #             SubPixelZeros[StartInd:EndInd, StartInd:EndInd] = OTF
    #             tmpPSF = (np.abs(np.fft.fftshift(np.fft.fft2(SubPixelZeros / Norm)))) ** 2
    #             PSF[:, :, zz, nn] = tmpPSF
    #             PSF[:, :, zz, nn] = PSF[:, :, zz, nn] / np.sum(PSF[:, :, zz, nn])

    # else:
    # Compute PSF for the case where offset phase is given
    for nn in [nn for nn in range(size(Z, axis=1)) if loads[nn] == 1]:
        for zz in range(size(Z, axis=0)):
            DefocusPhase = DefocusK * Z[zz, nn]
            Phase = Mask* (DefocusPhase+ Pupil_Phase+ XOffsetPhase* X[zz,nn]+ YOffsetPhase* Y[zz,nn])
            #Optical transfer function (propagator)
            OTF = Mask*Pupil_Mag*np.exp(1j*Phase)
            #Parseval Normalizaton
            Norm = np.sqrt(np.sum(np.abs(OTF)))*NPix
            #Padding zeros for subpixel resolution
            SubPixelZeros[StartInd:EndInd,StartInd:EndInd] = OTF
            tmpPSF = (np.abs(np.fft.fftshift(np.fft.fft2(SubPixelZeros/Norm))))**2
            PSF[:,:,zz,nn] = tmpPSF
            PSF[:,:,zz,nn] = PSF[:,:,zz,nn]/np.sum(PSF[:,:,zz,nn])

    PSF = np.sum(PSF, axis=3)
    BgPSF = I*PSF+ Bg
    return BgPSF,PSF
def samplePupil(ITr, Data, Chain, PSFstack, Chol_A, Chol_Phi, DefocusK, Mask, DelX, AcceptJ, XOffsetPhase, YOffsetPhase, tmpPhase, ADisk, Z0, Zx, Zy, Zz, SubPixelZeros, StartInd, EndInd, SubPixel, tp):
    """Perform one step of a Metropolis-Hastings algorithm to sample from the Phase and the Magnitude of a pupil function model.

    Args:
        ITr (ndarray): An array of integers representing the itration number of MCMC.
        Data (ndarray): An array of floats representing the observed image.
        Chain (dict): A dictionary containing the current state of the Markov chain. Should contain keys 'Phase', 'Mag', 'Bg', 'I', 'X', 'Y', 'Z', 'loads'.
        PSFstack (ndarray): An array of floats representing the stack of simulated PSFs.
        Chol_A (ndarray): The Cholesky decomposition of the covariance matrix for the log magnitude proposal distribution.
        Chol_Phi (ndarray): The Cholesky decomposition of the covariance matrix for the phase proposal distribution.
        DefocusK (float): A float representing the defocus parameter.
        Mask (ndarray): A boolean array representing the pupil mask.
        DelX (ndarray): An array of floats representing the offsets in x, y, and z coordinates for each PSF.
        AcceptJ (int): An integer representing the number of accepted jumps in the Markov chain.
        XOffsetPhase (ndarray): An array of floats representing the phase offset in the x direction.
        YOffsetPhase (ndarray): An array of floats representing the phase offset in the y direction.
        tmpPhase (ndarray): An array of floats representing the current estimated phase without zernich parts.
        ADisk (float): A float representing the area of the pupil mask.
        Z0 (numpy.ndarray): Zernike polynomial coefficients for piston.
        Zx (numpy.ndarray): Zernike polynomial coefficients for x-tilt.
        Zy (numpy.ndarray): Zernike polynomial coefficients for y-tilt.
        Zz (numpy.ndarray): Zernike polynomial coefficients for defocus.
        SubPixelZeros (ndarray): An array of floats representing the sub-pixel positions of the PSFs.
        StartInd (int): An integer representing the starting index in the PSFstack for each PSF.
        EndInd (int): An integer representing the ending index in the PSFstack for each PSF.
        SubPixel (ndarray): An array of floats representing the sub-pixel positions of the image.
        tp (float): A float representing the temperature parameter for the Metropolis-Hastings algorithm.

    Returns:
        tuple: A tuple containing the updated magnitude, phase, estimated phase, PSF stack, acceptance count for magnitude, and prior probability for magnitude and phase.
    """

    # Extract current state from the Markov chain
    Phase = Chain['Phase']
    Mag = Chain['Mag']
    Bg = Chain['Bg']
    I = Chain['I']
    Sig = 0.005
    X = Chain['X']
    Y = Chain['Y']
    Z = Chain['Z']
    loads = Chain['loads']

    # Propose a new phase value
    tPhase = Phase.flatten() + Sig*np.matmul(np.random.normal(size=(1,np.size(Phase))), Chol_Phi).flatten()
    tPhase = tPhase.reshape(*Mask.shape)
    SubPhase = tmpphsaec(ITr, tPhase.copy(), Z0, Zx, Zy, Zz, Mask, ADisk)
   
    # Sample new PSF using new Phase
    tPSF = []
    for ii in range(np.size(DelX,axis=0)):
        tPSF.append(findPSF(Mag, SubPhase,Bg[ii],I, loads,DefocusK,Z+DelX[ii,2],Mask,SubPixelZeros,StartInd,EndInd,SubPixel,X+DelX[ii,0],Y+DelX[ii,1],XOffsetPhase,YOffsetPhase)[0])
    tPSF = np.array(tPSF)

    # Accept or reject new Phase
    DLogL = np.sum(Data*(np.log(tPSF)-np.log(PSFstack))-(tPSF-PSFstack))/tp
    if DLogL > np.log(np.random.rand()):
        Phase = tPhase.copy()
        tmpPhase = SubPhase.copy()
        PSFstack = tPSF.copy()
        AcceptJ = AcceptJ + 1

    # Sample new Mag
    tMag = np.exp(np.log(Mag.flatten()) + 0.2*Sig*np.matmul(np.random.normal(size=(1,np.size(Phase))), Chol_A).flatten())
    tMag = tMag.reshape(*Mask.shape)

    # Sample new PSF using new Mag
    tPSF = []
    for ii in range(np.size(DelX,axis=0)):
        tPSF.append(findPSF(tMag,tmpPhase,Bg[ii],I, loads,DefocusK,Z+DelX[ii,2],Mask,SubPixelZeros,StartInd,EndInd,SubPixel,X+DelX[ii,0],Y+DelX[ii,1],XOffsetPhase,YOffsetPhase)[0])
    tPSF = np.array(tPSF)

    # Accept or reject new Mag
    DLogL = np.sum(Data*(np.log(tPSF)-np.log(PSFstack))-(tPSF-PSFstack))/tp
    LogPriorProp = 0
    LogPriorOld = 0
    LogPriorRatio = LogPriorProp - LogPriorOld
    LogPostRatio = DLogL + LogPriorRatio
    if LogPostRatio > np.log(np.random.rand()):
        Mag = tMag.copy()
        PSFstack = tPSF.copy()
        AcceptJ +=  1

    # Compute log prior probabilities for A and Phi
    LPrior_A = 0
    LPrior_Phi = 0

    return Mag, Phase, tmpPhase, PSFstack, AcceptJ, LPrior_A, LPrior_Phi
def tmpphsaec(Iteration, Phase, Z0, Zx, Zy, Zz, Mask, ADisk):
    """
    Applies phase retrieval algorithm to obtain a phase estimate.

    Args:
        Iteration (int): Number of iterations to perform.
        Phase (numpy.ndarray): Phase estimate.
        Z0 (numpy.ndarray): Zernike polynomial coefficients for piston.
        Zx (numpy.ndarray): Zernike polynomial coefficients for x-tilt.
        Zy (numpy.ndarray): Zernike polynomial coefficients for y-tilt.
        Zz (numpy.ndarray): Zernike polynomial coefficients for defocus.
        Mask (numpy.ndarray): Binary mask.
        ADisk (float): Area of the disk used for averaging.

    Returns:
        numpy.ndarray: Updated phase estimate.
    """
    # create a copy of the original phase estimate
    SubPhase = np.copy(Phase)
    
    # iterate the phase retrieval algorithm
    for _ in range(Iteration):
        # compute the shift in x, y, z, and offset
        XShift = np.sum(SubPhase*Zx*Mask)/ADisk
        YShift = np.sum(SubPhase*Zy*Mask)/ADisk
        ZShift = np.sum(SubPhase*Zz*Mask)/ADisk
        Offset = np.sum(SubPhase*Z0*Mask)/ADisk
        
        # subtract the shifts and offset from the phase estimate, and apply the mask
        SubPhase = (SubPhase - XShift*Zx - YShift*Zy - ZShift*Zz - Offset*Z0)*Mask

    return SubPhase
def sample_load(Data, Chain, PSFstack, tmpPhase, DelX, DefocusK, Mask, bnp, SubPixelZeros, StartInd, EndInd, SubPixel, XOffsetPhase, YOffsetPhase, tp):
    """
    This function performs the Metropolis-Hastings step in the MCMC algorithm for Loads.
    
    Parameters:
    -----------
    Data : ndarray
        Observed data.
    Chain : dict
        Dictionary containing current state of the MCMC chain.
    PSFstack : ndarray
        Stack of PSFs for all observed images.
    tmpPhase : ndarray
        Current estimate of the object phase.
    DelX : ndarray
        Array containing sub-pixel shifts for each observed image.
    DefocusK : ndarray
        Defocus aberration coefficient.
    Mask : ndarray
        Binary array defining the support region of the object.
    bnp : dict
        Dictionary containing information.
    SubPixelZeros : ndarray
        Array containing sub-pixel coordinates of zeros of the PSF.
    StartInd : ndarray
        Starting index of sub-pixel PSF for each observed image.
    EndInd : ndarray
        Ending index of sub-pixel PSF for each observed image.
    SubPixel : int
        Sub-pixel sampling rate for generating PSFs.
    XOffsetPhase : ndarray
        Array containing x-coordinates of phase centers for each observed image.
    YOffsetPhase : ndarray
        Array containing y-coordinates of phase centers for each observed image.
    tp : float
        tempreture.
    
    Returns:
    --------
    load : ndarray
        Updated load array.
    PSFstack : ndarray
        Updated PSF stack.
    """
    
    # Extract current state of the MCMC chain
    Phase = Chain['Phase']
    Mag = Chain['Mag']
    Bg = Chain['Bg']
    I = Chain['I']
    X = Chain['X']
    Y = Chain['Y']
    Z = Chain['Z']
    loads = Chain['loads']
    
    # Copy loads chain and initialize weak parameters in bernoli pdf
    m = np.size(loads)
    tloads = np.copy(loads)
    gm = bnp['gm']
    
    # Choose two random indices in the loads 
    r1, r2 = np.random.choice(m, 2, replace=False)
    
    # Iterate through all possible loads 
    pair = [(x, y) for x in [0,1] for y in [0,1]]
    for it in range(4):
        x, y = pair[it]
        
        # Update the loads 
        tloads[r1] = x
        tloads[r2] = y
        
        # Compute the PSF stack for the updated loads
        tPSF = []
        for ii in range(np.size(DelX,axis=0)):
            tPSF.append(findPSF(Mag, tmpPhase, Bg[ii], I, tloads, DefocusK, Z+DelX[ii,2], Mask, SubPixelZeros, StartInd, EndInd, SubPixel, X+DelX[ii,0], Y+DelX[ii,1], XOffsetPhase, YOffsetPhase)[0])
        tPSF = np.array(tPSF)
        
        # Compute the log-likelihood and log-prior probabilities for the updated chain
        DLogL = np.sum(Data*(log(tPSF)-log(PSFstack))-(tPSF-PSFstack))
        DLogPrior = np.sum(log(benolipdf(tloads,m,gm)) - log(benolipdf(loads,m,gm)))
        Dpost = DLogL+ DLogPrior
        if Dpost> log(np.random.rand()):
            loads[r1] = x
            loads[r2] = y
            PSFstack = tPSF.copy()

    return loads, PSFstack
def plot_trj(jkl, M, PSFstruct, XX, YY, ZZ):
    """
    Plot the trajectories of particles.

    Args:
    - jkl (int): An integer indicating the iteration number.
    - M (int): An integer indicating the number of particles.
    - PSFstruct (dict): A dictionary containing the ground truth trajectories of particles.
                        The keys are 'X', 'Y', and 'Z', and the values are arrays of shape (3, N),
                        where N is the number of frames.
    - XX (array): An array of shape (N, M) containing the learned trajectories of particles in the X dimension.
    - YY (array): An array of shape (N, M) containing the learned trajectories of particles in the Y dimension.
    - ZZ (array): An array of shape (N, M) containing the learned trajectories of particles in the Z dimension.

    Returns:
    - None.
    """
    rows = M
    cols = 1

    # create the figure
    fig = plt.figure(figsize=(21, M*7))

    # create a grid for pairs of subplots
    grid = plt.GridSpec(rows, cols)

    for i in range(rows * cols):
        # create fake subplot just to set the title for the set of subplots
        fake = fig.add_subplot(grid[i])
        fake.set_title(f'Particle #{i+1}\n', fontweight='semibold', size=14)
        fake.set_axis_off()

        # create subgrid for three subplots without space between them
        gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[i])

        # real subplot #1
        ax = fig.add_subplot(gs[0])
        ax.plot(PSFstruct['X'][0], color="black", label='Ground truth_1')
        ax.plot(PSFstruct['X'][1], color="darkgreen", label='Ground truth_2')
        ax.plot(PSFstruct['X'][2], color="darkblue", label='Ground truth_3')
        ax.plot(XX[:, i], '-.', color="coral", label='Learned')
        ax.set_title('X')
        ax.set_xlabel("frame")
        ax.set_ylabel("X (nm)")
        ax.legend()

        # real subplot #2
        ax = fig.add_subplot(gs[1], sharey=ax)
        ax.plot(PSFstruct['Y'][0], color="black", label='Ground truth_1')
        ax.plot(PSFstruct['Y'][1], color="darkgreen", label='Ground truth_2')
        ax.plot(PSFstruct['Y'][2], color="darkblue", label='Ground truth_3')
        ax.plot(YY[:, i], '-.', color="coral", label='Learned')
        ax.set_title('Y')
        ax.set_xlabel("frame")
        ax.set_ylabel("Y (nm)")
        ax.legend()
    
        # real subplot #3
        ax = fig.add_subplot(gs[2], sharey=ax)
        ax.plot(PSFstruct['Z'][0], color = "black", label='Ground truth_1')
        ax.plot(PSFstruct['Z'][1], color = "darkgreen", label='Ground truth_2')
        ax.plot(PSFstruct['Z'][2], color = "darkblue", label='Ground truth_3')
        ax.plot(ZZ[:,i], '-.', color = "coral", label='Learned')
        ax.set_title('Z')
        ax.set_xlabel("frame")
        ax.set_ylabel("Z (nm)")
        ax.legend()

    fig.patch.set_facecolor('white')
    fig.suptitle('Trajectories', fontweight='bold', size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"traj_{jkl}")
    plt.show()
def Save_info(itr, x, y, z, phase, mag, I, bg, D, loads, post, psf, path=""):
    """
    Save intermediate results of the MCMC algorithm for debugging and analysis.

    Args:
        itr (int): The current iteration number.
        x (list): List of estimated X coordinates of the particles.
        y (list): List of estimated Y coordinates of the particles.
        z (list): List of estimated Z coordinates of the particles.
        phase (list): List of estimated phases of the particles.
        mag (list): List of estimated magnitudes of the particles.
        I (list): List of estimated particle intensities.
        bg (numpy.ndarray): Background noise array.
        D (numpy.ndarray): Estimated diffusion coefficients of the particles.
        loads (numpy.ndarray): Array each particle is activeor not.
        post (list): List of log posterior probabilities.
        psf (numpy.ndarray): Point spread function array.
        path (str): Path to save the intermediate results.

    Returns:
        None
    """
    # Save results every 10,000 iterations
    each = 10000
    kj = 1000
    if itr % each == 0:
        r = int(itr / each)
        # Save diffusion coefficients
        name = path + f"D_{r}"
        np.save(name, np.array(D[:itr]))
        # Save log posterior probabilities
        name = path + f"Log_post_{r}"
        np.save(name, np.array(post[:itr]))
        # Save particle intensities
        name = path + f"I_{r}"
        np.save(name, np.array(I[:itr]))
        # Save background noise
        name = path + f"Bg_{r}"
        np.save(name, np.array(bg[:itr, :]))
        # Save X coordinates
        name = path + f"X_{r}"
        np.save(name, np.array(x)[itr - kj : itr, :, :])
        # Save Y coordinates
        name = path + f"Y_{r}"
        np.save(name, np.array(y)[itr - kj : itr, :, :])
        # Save Z coordinates
        name = path + f"Z_{r}"
        np.save(name, np.array(z)[itr - kj : itr, :, :])
        # Save phases
        name = path + f"Phase_{r}"
        np.save(name, np.array(phase)[itr - kj : itr, :, :])
        # Save magnitudes
        name = path + f"Mag_{r}"
        np.save(name, np.array(mag)[itr - kj : itr, :, :])
        # Save loads array
        name = path + f"loads{r}"
        np.save(name, loads[:itr, :])
def sample_traj(it,Data, 
                Chain, 
                PSFstack,
                Norm1_PSFstack, 
                BNP,
                Struct,
                DefocusK,
                Mask,
                DelX,
                AcceptX,
                XOffsetPhase,
                YOffsetPhase,
                tmpPhase,
                SubPixelZeros,
                StartInd,
                EndInd,
                SubPixel,
                tp
                ):
    """
    Sample a new particle trajectory given the current state of the chain.

    Parameters:
    -----------
    it : int
        Current iteration index in the MCMC chain.
    Data : ndarray
        Input data.
    Chain : dict
        Dictionary containing information about the current state of the MCMC chain.
    PSFstack : ndarray
        Stack of PSFs used for generating the simulated data.
    Norm1_PSFstack : ndarray
        Stack of normalized PSFs used for generating the simulated data.
    BNP : dict
        Dictionary containing information about Brownian noise process used in simulating the data.
    Struct : dict
        Dictionary containing structural information.
    DefocusK : float
        Defocus constant used in finding the PSFs.
    Mask : ndarray
        Binary mask for the region of interest.
    DelX : ndarray
        Array of delX values used in calculating the acceptance probability.
    AcceptX : ndarray
        Array of acceptX values used in calculating the acceptance probability.
    XOffsetPhase : ndarray
        Array of X offset phase values.
    YOffsetPhase : ndarray
        Array of Y offset phase values.
    tmpPhase : ndarray
        Temporary phase variable.
    SubPixelZeros : ndarray
        Array of sub-pixel zeros.
    StartInd : int
        Starting index for the particle.
    EndInd : int
        Ending index for the particle.
    SubPixel : bool
        Flag indicating whether sub-pixel localization is being used.
    tp : int
        tempreture.

    Returns:
    --------
    Chain : dict
        Updated dictionary containing information about the new state of the MCMC chain.
    lp : float
        Log probability of the new"""
    # Get some parameters from the input Chain dictionary
    Mag = Chain['Mag']; Bg = Chain['Bg']; I = Chain['I']
    
    # Define possible Sig values and their corresponding probabilities for random selection
    Sig_values = np.array([1, 2, 10, 20])
    Sig_probs = np.array([0.83, 0.15, 0.01, 0.01])
    
    # Choose a random value of Sig according to the defined probabilities
    Sig = np.random.choice(Sig_values, p=Sig_probs)
    
    # Get some more parameters from the input Chain and BNP dictionaries
    X = Chain['X']; Y = Chain['Y']; Z = Chain['Z'];
    D = Chain['D']; Dt = BNP['Dt']; loads = Chain['loads']
    Sig_D = np.sqrt(2*D*Dt); SigX = Struct['NPix']*Struct['PixelSize']/2
    
    lp = 0
    
    # Select two random indices for the next steps
    shp = Z.shape
    rnd = np.random.choice(shp[1], size=2, replace=False)
    
    # Copy the current X, Y, Z values to new temporary variables
    tcx, tcy, tcz = np.copy(X), np.copy(Y), np.copy(Z)
    
    # Loop over the selected indices
    for nn in rnd:
        # If loads is 1 for this index, add random noise to the temporary variables
        if loads[nn]==1:
            tcx[:,nn] = X[:,nn] + Sig*np.random.randn(shp[0])
            tcy[:,nn] = Y[:,nn] + Sig*np.random.randn(shp[0])
            tcz[:,nn] = Z[:,nn] + Sig*np.random.randn(shp[0])
        # If loads is 0 for this index, generate new random X, Y, Z values until they satisfy some constraints
        else:
            while True:
                X[0,nn]= np.random.normal(0,(1/3)*SigX)
                if -1500 < X[0,nn] < 1500:
                    break
            while True:
                Y[0,nn]= np.random.normal(0,(1/3)*SigX)
                if -1500 < Y[0,nn] < 1500:
                    break
            while True:
                Z[0,nn]= np.random.normal(0,(1/5)*SigX)
                if -500 < Z[0,nn] < 500:
                    break
            for it in range(1,shp[0]):
                X[it,nn] = np.random.normal(X[it-1,nn],Sig_D)
                Y[it,nn] = np.random.normal(Y[it-1,nn],Sig_D)
                Z[it,nn] = np.random.normal(Z[it-1,nn],Sig_D)

    # If loads is 1 for at least one of the selected indices, calculate acceptance probability and update variables
    if loads[rnd[0]]==1 or loads[rnd[1]]==1:
        AlphaPrior = 12
        BetaPrior = 900
        # calculate alpha parameter for old trajectories using BetaPrior and current number of loads switches
        R = np.size(DelX,axis =0)
        NN = np.sum(loads)
        Alpha = (3/2)*R*NN*(np.size(X,axis=0)-1) + AlphaPrior
        
        # calculate beta parameter for old trajectories  using BetaPrior and current particle positions
        Beta = BetaPrior
        for nn in [nn for nn in range(size(Z,axis=1)) if loads[nn]==1]:
            Beta += (R*np.sum((X[1:,nn]-X[:-1,nn])**2 +(Y[1:,nn]-Y[:-1,nn])**2 + (Z[1:,nn]-Z[:-1,nn])**2))/(4*Dt)
        
        # calculate beta parameter for propsed trajectories  using BetaPrior and current particle positions
        Beta_prop = BetaPrior
        for nn in [nn for nn in range(size(Z,axis=1)) if loads[nn]==1]:
            Beta_prop += (R*np.sum((tcx[1:,nn]-tcx[:-1,nn])**2 +(tcy[1:,nn]-tcy[:-1,nn])**2 + (tcz[1:,nn]-tcz[:-1,nn])**2))/(4*Dt)

        # calculate Log prior using Beta_prop
        alpha_beta = Alpha* (np.log((Beta)) -np.log((Beta_prop)))
        DLogPost = alpha_beta

        # calculate PSF 
        tPSF =[]
        Norm1_PSFstack =[]
        for ii in range(np.size(DelX,axis=0)):
            BgPSF, tNorm1_PSF = findPSF(Mag, tmpPhase, Bg[ii], I, loads, 
                                    DefocusK, tcz+DelX[ii,2], Mask, 
                                    SubPixelZeros, StartInd, EndInd,
                                    SubPixel, tcx+DelX[ii,0], tcy+DelX[ii,1],
                                    XOffsetPhase, YOffsetPhase)
        
            tPSF.append(BgPSF)
            Norm1_PSFstack.append(tNorm1_PSF)

        PSFstack, Norm1_PSFstack = np.array(tPSF), np.array(Norm1_PSFstack)

        # calculate log likelihood
        DLogL = np.sum(Data*(np.log(tPSF)-np.log(PSFstack))-(tPSF-PSFstack))/tp
        DLogPost += DLogL
        if DLogPost > np.log(1-np.random.rand()):
            X[:,:] = tcx[:,:]
            Y[:,:] = tcy[:,:]
            Z[:,:] = tcz[:,:]
            PSFstack = tPSF
            AcceptX = AcceptX + 1
    if np.sum(loads)==0:
        Norm1_PSFstack = np.zeros_like(PSFstack)
        PSFstack = np.zeros_like(PSFstack)
        PSFstack[0,:,:,:] = Bg[0]
        PSFstack[1,:,:,:] = Bg[1]

    return X, Y, Z, PSFstack, AcceptX, Norm1_PSFstack, lp
def RunRit(Data, Struct, BNP, Bg, D, M, Xstart="None", Ystart="None"):
    """Runs a Bayesian inference algorithm for 3D localization microscopy using a Markov Chain Monte
    Carlo (MCMC) method
    
    
    """


    # Check if sub-pixel flag is present in Struct, and set it to 1 if so
    if 'SubPixel' in Struct:
        Struct['SubPixel'] = 1
    
    # Calculate size of sub-pixel array and initialize an array of zeros with this size
    SZ = Struct['SubPixel']*Struct['NPix']
    SubPixelZeros = np.zeros((SZ,SZ),dtype=np.complex)
    
    # Calculate start and end indices for sub-pixel array
    StartInd = int(SZ/2-Struct['NPix']/2)
    EndInd = int(SZ/2+Struct['NPix']/2)
    
    # Calculate pixel size and pupil radius
    KPixelSize = 1/(Struct['PixelSize']*Struct['NPix'])
    PupilRadius = Struct['Na']/(Struct['Lambda']*KPixelSize)
    
    # Create meshgrid for kx and ky
    Kx,Ky= np.meshgrid(np.arange(-Struct['NPix']/2,Struct['NPix']/2), np.arange(-Struct['NPix']/2,Struct['NPix']/2))
    
    # Calculate autocorrelation matrices for amplitude and phase
    K_A = calCOV(KPixelSize*Kx, KPixelSize*Ky, BNP['T_A'], BNP['L_A'])
    K_Phi = calCOV(KPixelSize*Kx, KPixelSize*Ky, BNP['T_Phi'], BNP['L_Phi'])
    
    # Cholesky decomposition of the autocorrelation matrices
    Chol_A = np.linalg.cholesky(K_A+1000*np.spacing(1)*np.eye(*K_Phi.shape))
    Chol_Phi = np.linalg.cholesky(K_Phi+1000*np.spacing(1)*np.eye(*K_A.shape))
    
    # Create radial mask to filter out high spatial frequencies
    Rho = np.sqrt(Kx**2+Ky**2)
    Mask = np.array(Rho <= PupilRadius,dtype=np.int8)
    
    # Calculate k-space coordinate grid
    Kr_Image = Rho*Mask*KPixelSize
    
    # Calculate defocus term in k-space
    DefocusK = 2*np.pi*np.sqrt((Struct['N']/Struct['Lambda'])**2 - Kr_Image**2)
    
    # Calculate offset phases in k-space
    XOffsetPhase = 2*np.pi*Kx*KPixelSize
    YOffsetPhase = 2*np.pi*Ky*KPixelSize
    
    # Create array to hold stack of particle coordinates and initialize to zeros
    DelX = np.stack((np.array([0, 0, 0]),np.array(Struct['DelX'])[0],np.array(Struct['DelX'])[1]))
    
    # Initialize chain with default values
    NStack = 100
    x0 = np.zeros((NStack,M))
    y0 = np.zeros((NStack,M))
    z0 = np.zeros((NStack,M))
    tChain = {'Mag':1, 'Phase': 1, 'PSFstack':1, 'Bg':np.array([10,25]) ,'I':2500, 
              'X':x0, 'Y':y0, 'Z':z0, 'D':60, 'LogLike':1, 'LogPost':1, 
              'loads': np.zeros(M)}
    tChain['Mag'] =     np.ones(Mask.shape)
    tChain['Phase'] = np.zeros_like(Mask) #Struct.Pupil_Phase;
    tChain['Bg'] = np.random.gamma(1,100,np.size(DelX,axis=0))
    tChain['I'] = gam.rvs(a=3, loc=3800, scale=100)
    tChain['D'] = 100*np.random.rand()
    if Xstart == "None":
        tChain['X'] = np.zeros((NStack,M)) #(ImSZ/10)*randn(1,NStack)/10+ImSZ/2
    else:
        tChain['X'] = Xstart
    if Ystart == "None":
        tChain['Y'] = np.zeros((NStack,M)) #%(ImSZ/10)*randn(1,NStack)/10+ImSZ/2
    else:
        tChain[0]['Y'] = Ystart
    tChain['Z'] = np.zeros((NStack,M))
    
    # tChain['X'] = np.load("/home/reza/My_Code/SMLn/gr6/X_24.npy")[-1,:,:]#np.transpose(Struct['X'])
    # tChain['Y'] = np.load("/home/reza/My_Code/SMLn/gr6/Y_24.npy")[-1,:,:]#np.transpose(Struct['Y'])
    # tChain['Z'] = np.load("/home/reza/My_Code/SMLn/gr6/Z_24.npy")[-1,:,:]#np.transpose(Struct['Z'])
    # tChain['loads'] = np.load("/home/reza/My_Code/SMLn/gr6/bernoli_24.npy")[-1,:]
    # np.sum(Data*np.log(PSFstack)-(PSFstack))

    # Calculate the PSF 
    PSFstack = []; Norm1_PSFstack = []    
    for it in range(np.size(DelX,axis=0)):
        Pstack,Npstak = (findPSF(tChain['Mag'], tChain['Phase'],
                                 tChain['Bg'][it], tChain['I'], 
                                 tChain['loads'], DefocusK, 
                                 tChain['Z'] + DelX[it,2],
                                 Mask, SubPixelZeros, 
                                 StartInd, EndInd,
                                 Struct['SubPixel'],
                                 tChain['X'] + DelX[it,0],
                                 tChain['Y'] + DelX[it,1],
                                 XOffsetPhase, YOffsetPhase))
        PSFstack.append(Pstack)
        Norm1_PSFstack.append(Npstak)
    PSFstack = np.array(PSFstack); Norm1_PSFstack = np.array(Norm1_PSFstack)
    
    Iter = 30
    # Extract Zernike polynomial coefficients
    Z0 = Struct['ZImages'][:,:,0]
    Zx = Struct['ZImages'][:,:,1]
    Zy = Struct['ZImages'][:,:,2]
    Zz = Struct['ZImages'][:,:,3]
    ADisk = np.pi*PupilRadius**2; tmpPhase = tmpphsaec(Iter, tChain['Phase'], Z0, Zx, Zy, Zz, Mask, ADisk)
    
    # Initialize various arrays to store the results
    AcceptPup = 0; AcceptTraj = 0; AcceptOffset = 0; AcceptI = 0; AcceptBg = 0; Acceptc=0; AcceptZ = 0; Accept_load=0
    postrior = np.zeros((BNP['NJump'])); dD = np.zeros((BNP['NJump']))
    xx, yy, zz = np.zeros((BNP['NJump'], NStack, M)), np.zeros((BNP['NJump'], NStack, M)), np.zeros((BNP['NJump'], NStack, M))
    pphase, mmag = np.zeros((BNP['NJump'], *Mask.shape)), np.zeros((BNP['NJump'], *Mask.shape))
    II = np.zeros((BNP['NJump'])); bbg = np.zeros((BNP['NJump'], DelX.shape[0]))
    dloads = np.zeros((BNP['NJump'], M))
    dtt = datetime.datetime.now()
    temp_i = 90000
    temp = 1
    switch_index = 5
    for jj in range(1, BNP['NJump']):
        
        # Save values for current iteration
        dloads[jj,:] = tChain['loads']
        II[jj] = tChain['I']
        dD[jj] = tChain['D']
        bbg[jj,:] = tChain['Bg']
        mmag[jj,:,:] = tChain['Mag']
        pphase[jj,:,:] = tChain['Phase']
        xx[jj,:,:] = tChain['X']
        yy[jj,:,:] = tChain['Y']
        zz[jj,:,:] = tChain['Z']

        # Sample the pupil function
        tChain['Mag'], tChain['Phase'], tmpPhase, PSFstack, AcceptPup, LPrior_A, LPrior_Phi = samplePupil(Iter, Data, tChain, PSFstack, Chol_A, Chol_Phi, 
                                                                                                          DefocusK, Mask, DelX, AcceptPup, 
                                                                                                          XOffsetPhase, YOffsetPhase, tmpPhase,
                                                                                                          ADisk, Z0, Zx, Zy, Zz, SubPixelZeros, 
                                                                                                          StartInd, EndInd, Struct['SubPixel'], temp)
        
        # Sample the trajectory
        if switch_index%5==0:
            for _ in range(M):
                tChain['X'], tChain['Y'], tChain['Z'], PSFstack, AcceptTraj, Norm1_PSFstack, lp = sample_traj(jj, Data, tChain, PSFstack, Norm1_PSFstack, BNP, Struct,
                                                                                                                    DefocusK, Mask, DelX,
                                                                                                                    AcceptTraj, XOffsetPhase, YOffsetPhase,
                                                                                                                    tmpPhase, SubPixelZeros, StartInd,
                                                                                                                    EndInd, Struct['SubPixel'], temp)              
             
        else:
            for _ in range(M):
                tChain['X'], tChain['Y'], tChain['Z'], PSFstack, AcceptTraj, Norm1_PSFstack, lp = sample_traj_hit_run(Data, tChain, PSFstack, Norm1_PSFstack, BNP, Struct,
                                                                                                                    DefocusK, Mask, DelX,
                                                                                                                    AcceptTraj, XOffsetPhase, YOffsetPhase,
                                                                                                                    tmpPhase, SubPixelZeros, StartInd,
                                                                                                                    EndInd, Struct['SubPixel'], temp)

                        
                tChain['Z'], PSFstack, AcceptTraj, Norm1_PSFstack, lp = sample_traj_hit_runZ(Data, tChain, PSFstack, Norm1_PSFstack, BNP, Struct,
                                                                                                                DefocusK, Mask, DelX,
                                                                                                                AcceptTraj, XOffsetPhase, YOffsetPhase,
                                                                                                                tmpPhase, SubPixelZeros, StartInd,
                                                                                                                EndInd, Struct['SubPixel'], temp)

        
        # Switch labels
        if 35000>jj>19000:
            for _ in range(3*int(np.sum(tChain['loads']))):
                tChain['X'], tChain['Y'], tChain['Z'], Acceptc, DLogP = Switch_label(tChain, Struct, BNP, Acceptc)
                tChain['X'], tChain['Y'], tChain['Z'], Acceptc, DLogP = Switch_one_frame(tChain, Struct, BNP, Acceptc)
        
        if 19000>jj>10000:
            for _ in range(3*int(np.sum(tChain['loads']))):
                tChain['X'], tChain['Y'], tChain['Z'], Acceptc, DLogP = Switch_one_frame(tChain, Struct, BNP, Acceptc)

        # Sample loads
        tChain['loads'], PSFstack = sample_load(Data, tChain, PSFstack, tmpPhase, DelX, DefocusK, Mask, BNP, 
                                                SubPixelZeros, StartInd, EndInd, Struct['SubPixel'], XOffsetPhase, YOffsetPhase, temp)

        
        # Sample the intensity and background parameters
        tChain['I'], PSFstack, AcceptI, LogPrior_I = sampleIntensity(Data, tChain, PSFstack, Norm1_PSFstack, DelX, AcceptI, temp_i)
        tChain['Bg'], PSFstack, AcceptBg, LogPrior_Bg = sampleBg(Data, tChain, PSFstack, Norm1_PSFstack, DelX, AcceptBg, temp)

        if jj>2*M:
            # Sample the diffiusion parameter
            tChain['D'], LPrior_D = sampleDiff(tChain, DelX, BNP['Dt'])
            #calculate log-posterior
            dlog_post = calculate_log_post(Data, tChain, PSFstack, Struct, LogPrior_Bg, LogPrior_I, LPrior_D, BNP, LPrior_A, LPrior_Phi, lp)
            postrior[jj] = dlog_post
            #save info
            Save_info(jj, xx, yy, zz, pphase, mmag, II, bbg, dD, dloads, postrior,PSFstack, path ="gr6/")
            
        temp_i = tempreture(jj, temp_i)
        Iter = change_Iteration(jj, Iter)

        #show result
        if  jj%10000 == 0:
            print(f"\n Itr = {jj}")
            print(f"  loads = {np.mean(dloads[jj-200:jj,:],axis=0)}")
            print(f"D = {np.mean(dD[jj-2000:jj])}")
            print(f"  traj = {AcceptTraj}")
            print(f"  I = {tChain['I']}")
            print(f"bg = {tChain['Bg']}")
            print(f"time interval = {datetime.datetime.now()-dtt}")
            print(f"time = {datetime.datetime.now()}")
            print(f"tempreture : {temp_i}")
            plot_trj(jj, M, Struct, tChain['X'], tChain['Y'], tChain['Z'])
            switch_index +=1
        elif (jj%100 ==0
            and jj<=2000):
            print(f"\n Itr = {jj}:")
            print(f"  loads = {np.mean(dloads[jj-10:jj,:],axis=0)}")
            print(f"D = {np.mean(dD[jj-10:jj])}")
            print(f"  I = {tChain['I']}")

    return tChain

def sample_traj_hit_run(Data, 
                Chain, 
                PSFstack,
                Norm1_PSFstack, 
                BNP,
                Struct,
                DefocusK,
                Mask,
                DelX,
                AcceptX,
                XOffsetPhase,
                YOffsetPhase,
                tmpPhase,
                SubPixelZeros,
                StartInd,
                EndInd,
                SubPixel,
                tp
                ):
    """
    Samples the trajectory using hit-or-run algorthim for 2 random particles.

    Args:
    - Data (ndarray): observed image stack (number of focalplane x number of frames x number of pixels x number of pixels)
    - Chain (dict): dictionary of chain variables
        - Mag (float): magnification
        - Bg (ndarray): background value for each frame
        - I (float): intensity
        - X (float): x-coordinate of particle
        - Y (float): y-coordinate of particle
        - Z (ndarray): z-coordinate of particle (number of particle x number of frames)
        - D (float): diffusion coefficient
        - loads (ndarray): binary variable indicating whether the particles is active or not
    - PSFstack (ndarray): PSF stack (number of focalplane xnumber of frames x number of pixels x number of pixels)
    - Norm1_PSFstack (ndarray): PSF stack normalized to have unit L1 norm (number of focalplane x number of frames x number of pixels x number of pixels)
    - BNP (dict): dictionary of Bayesian network parameters
        - Dt (float): time interval
    - Struct (dict): dictionary of structural parameters
        - NPix (int): number of pixels per side of image stack
        - PixelSize (float): pixel size in nanometers
    - DefocusK (ndarray): defocus kernel (number of pixels x number of pixels)
    - Mask (ndarray): binary mask indicating which pixels to use in the calculation
    - DelX (ndarray): array of pixel shifts to calculate the gradient (number of focalplane x 3)
    - AcceptX (int): number of accepted gradient steps
    - XOffsetPhase (ndarray): x-coordinate of particle offset due to phase shift 
    - YOffsetPhase (ndarray): y-coordinate of particle offset due to phase shift 
    - tmpPhase (ndarray): phase shift (number of pixels x number of pixels)
    - SubPixelZeros (ndarray): zero matrices for sub-pixel calculation
    - StartInd (int): starting frame index for likelihood calculation
    - EndInd (int): ending frame index for likelihood calculation
    - SubPixel (int): sub-pixel factor for likelihood calculation
    - tp (int): tempreture

    Returns:
    - X (ndarray): x-coordinate of particle after sampling (number of particle x number of frames)
    - Y (ndarray): y-coordinate of particle after sampling (number of particle x number of frames)
    - Z (ndarray): z-coordinate of particle after sampling (number of particle x number of frames)
    - PSFstack (ndarray): updated PSF stack after sampling (number of particle x number of frames x number of pixels x number of pixels)
    - AcceptX (int): updated number of accepted gradient steps
    - Norm1_PSFstack (ndarray): updated PSF stack normalized to have unit L1 norm (number of particle x number of frames x number of pixels x number of pixels)
    - lp (float): log prior
    """

    # Extract variables
    Mag = Chain['Mag']
    Bg = Chain['Bg']
    I = Chain['I']
    Sig = np.random.choice([1,2], p=[0.9,0.10])
    X = Chain['X']
    Y = Chain['Y']
    Z = Chain['Z']
    D = Chain['D']
    Dt = BNP['Dt']
    loads = Chain['loads']
    Sig_D = np.sqrt(2*D*Dt)
    SigX = Struct['NPix']*Struct['PixelSize']/2
    lp = 0
    shp = Z.shape
    rnd = np.random.choice(shp[1],size=2, replace=False)
    tcx = np.copy(X)
    tcy = np.copy(Y)
    tcz = np.copy(Z)

    # Generate two random numbers and update trajectories accordingly
    for nn in rnd:
        if loads[nn]==1:
            tmp = np.random.rand()
            # With probability 0.2, add a Gaussian perturbation to the current position
            if tmp < 0.2:
                tcx[:,nn] = X[:,nn] + Sig*np.random.randn(shp[0]) 
                tcy[:,nn] = Y[:,nn] + Sig*np.random.randn(shp[0])
                tcz[:,nn] = Z[:,nn] + 2*Sig*np.random.randn(shp[0])
            # Otherwise, draw a random direction and move along it with a Gaussian-distributed distance
            else:
                U = np.random.rand(3,shp[0])
                U = U/np.linalg.norm(U, axis=0)
                Lambda = Sig*np.random.randn()
                tcx[:,nn] = X[:,nn] + Lambda*U[0,:] 
                tcy[:,nn] = Y[:,nn] + Lambda*U[1,:]
                tcz[:,nn] = Z[:,nn] + Lambda*U[2,:]
        # If loads variable for the current trajectory is 0, draw new positions from a normal distribution with certain bounds
        else:
            while True:
                X[0,nn]= np.random.normal(0,(1/3)*SigX)
                if -1500 < X[0,nn] < 1500:
                    break
            while True:
                Y[0,nn]= np.random.normal(0,(1/3)*SigX)
                if -1500 < Y[0,nn] < 1500:
                    break
            while True:
                Z[0,nn]= np.random.normal(0,(1/5)*SigX)
                if -500 < Z[0,nn] < 500:
                    break
            for it in range(1,shp[0]):
                X[it,nn] = np.random.normal(X[it-1,nn],Sig_D)
                Y[it,nn] = np.random.normal(Y[it-1,nn],Sig_D)
                Z[it,nn] = np.random.normal(Z[it-1,nn],Sig_D)

    # Check if any change in active particle
    if loads[rnd[0]]==1 or loads[rnd[1]]==1:
        
        # Calculate the log prior probability of the proposed trajectory
        DLPrior1X = np.sum(log(normpdf(tcx[0,:],0,SigX))-log(normpdf(X[0,:],0,SigX)))
        DLPrior1Y = np.sum(log(normpdf(tcy[0,:],0,SigX))-log(normpdf(Y[0,:],0,SigX)))
        DLPrior1Z = np.sum(log(normpdf(tcz[0,:],0,SigX))-log(normpdf(Z[0,:],0,SigX)))
        DLPriorX = np.sum(log(normpdf(tcx[1:,:],tcx[:-1,:],Sig_D))-log(normpdf(X[1:,:],X[:-1,:],Sig_D)))
        DLPriorY = np.sum(log(normpdf(tcy[1:,:],tcy[:-1,:],Sig_D))-log(normpdf(Y[1:,:],Y[:-1,:],Sig_D)))
        DLPriorZ = np.sum(log(normpdf(tcz[1:,:],tcz[:-1,:],Sig_D))-log(normpdf(Z[1:,:],Z[:-1,:],Sig_D)))
        Lprior = DLPrior1X + DLPrior1Y + DLPrior1Z + DLPriorX + DLPriorY + DLPriorZ
        DLogPost = Lprior

        # Calculate the proposed psf
        tPSF =[]
        Norm1_PSFstack =[]
        for ii in range(np.size(DelX,axis=0)):
            BgPSF, tNorm1_PSF = findPSF(Mag, tmpPhase, Bg[ii], I, loads, 
                                    DefocusK, tcz+DelX[ii,2], Mask, 
                                    SubPixelZeros, StartInd, EndInd,
                                    SubPixel, tcx+DelX[ii,0], tcy+DelX[ii,1],
                                    XOffsetPhase, YOffsetPhase)
        
            tPSF.append(BgPSF)
            Norm1_PSFstack.append(tNorm1_PSF)

        tPSF = np.array(tPSF); Norm1_PSFstack = np.array(Norm1_PSFstack)

        # Calculate the log likelihood difernce
        DLogL = np.sum(Data*(np.log(tPSF)-np.log(PSFstack))-(tPSF-PSFstack))/tp
        DLogPost += DLogL 

        # Decide whether to accept the proposed trajectory based on the log probability ratio
        if DLogPost > np.log(np.random.rand()):
            X[:,:] = tcx[:,:]
            Y[:,:] = tcy[:,:]
            Z[:,:] = tcz[:,:]
            PSFstack = tPSF
            AcceptX = AcceptX + 1
            lp = np.sum(Data*np.log(PSFstack)-(PSFstack))
    
    # If no active particle in the system
    if np.sum(loads)==0:
        Norm1_PSFstack = np.zeros_like(PSFstack)
        PSFstack = np.zeros_like(PSFstack)
        PSFstack[0,:,:,:] = Bg[0]
        PSFstack[1,:,:,:] = Bg[1]

    return X, Y, Z, PSFstack, AcceptX, Norm1_PSFstack, lp
def sample_traj_hit_runZ(Data, 
                Chain, 
                PSFstack,
                Norm1_PSFstack, 
                BNP,
                Struct,
                DefocusK,
                Mask,
                DelX,
                AcceptX,
                XOffsetPhase,
                YOffsetPhase,
                tmpPhase,
                SubPixelZeros,
                StartInd,
                EndInd,
                SubPixel,
                tp
                ):
    """
    Samples the trajectory using hit-or-run algorthim in z-direction for 2 random particle.

    Args:
    - Data (ndarray): observed image stack (number of focalplane x number of frames x number of pixels x number of pixels)
    - Chain (dict): dictionary of chain variables
        - Mag (float): magnification
        - Bg (ndarray): background value for each frame
        - I (float): intensity
        - X (float): x-coordinate of particle
        - Y (float): y-coordinate of particle
        - Z (ndarray): z-coordinate of particle (number of particle x number of frames)
        - D (float): diffusion coefficient
        - loads (ndarray): binary variable indicating whether the particles is acive or not
    - PSFstack (ndarray): PSF stack (number of focalplane xnumber of frames x number of pixels x number of pixels)
    - Norm1_PSFstack (ndarray): PSF stack normalized to have unit L1 norm (number of focalplane x number of frames x number of pixels x number of pixels)
    - BNP (dict): dictionary of Bayesian network parameters
        - Dt (float): time interval
    - Struct (dict): dictionary of structural parameters
        - NPix (int): number of pixels per side of image stack
        - PixelSize (float): pixel size in nanometers
    - DefocusK (ndarray): defocus kernel (number of pixels x number of pixels)
    - Mask (ndarray): binary mask indicating which pixels to use in the calculation
    - DelX (ndarray): array of pixel shifts to calculate the gradient (number of focalplane x 3)
    - AcceptX (int): number of accepted gradient steps
    - XOffsetPhase (ndarray): x-coordinate of particle offset due to phase shift 
    - YOffsetPhase (ndarray): y-coordinate of particle offset due to phase shift 
    - tmpPhase (ndarray): phase shift (number of pixels x number of pixels)
    - SubPixelZeros (ndarray): zero matrices for sub-pixel calculation
    - StartInd (int): starting frame index for likelihood calculation
    - EndInd (int): ending frame index for likelihood calculation
    - SubPixel (int): sub-pixel factor for likelihood calculation
    - tp (int): tempreture

    Returns:
    - Z (ndarray): z-coordinate of particle after sampling (number of particle x number of frames)
    - PSFstack (ndarray): updated PSF stack after sampling (number of particle x number of frames x number of pixels x number of pixels)
    - AcceptX (int): updated number of accepted gradient steps
    - Norm1_PSFstack (ndarray): updated PSF stack normalized to have unit L1 norm (number of particle x number of frames x number of pixels x number of pixels)
    - lp (float): log prior
    """
    
    # Extract variables
    Mag = Chain['Mag']; Bg = Chain['Bg']; I = Chain['I']; Sig = np.random.choice([1,2], p=[0.9,0.1]);
    X = Chain['X']; Y = Chain['Y']; Z = Chain['Z'];
    D = Chain['D']; Dt = BNP['Dt']; loads = Chain['loads']
    Sig_D = np.sqrt(2*D*Dt); SigX = Struct['NPix']*Struct['PixelSize']/2
    lp = 0
    shp = Z.shape

    # Generate two random numbers and update trajectories accordingly
    rnd = np.random.choice(shp[1],size=2, replace=False)
    tcz = np.copy(Z)
    for nn in rnd:
        if loads[nn]==1:
            tcz[:,nn] = Z[:,nn] + 2*Sig*np.random.randn(shp[0])

    # Check if any change in active particle
    if loads[rnd[0]]==1 or loads[rnd[1]]==1:

        # Calculate the log prior probability of the proposed trajectory
        DLPrior1Z = np.sum(log(normpdf(tcz[0,:],0,SigX))-log(normpdf(Z[0,:],0,SigX)))
        DLPriorZ = np.sum(log(normpdf(tcz[1:,:],tcz[:-1,:],Sig_D))-log(normpdf(Z[1:,:],Z[:-1,:],Sig_D)))
        Lprior = DLPrior1Z + DLPriorZ
        DLogPost = Lprior
        
        # Calculate the proposed psf and log-likelihood
        tPSF =[]
        Norm1_PSFstack =[]
        for ii in range(np.size(DelX,axis=0)):
            BgPSF, tNorm1_PSF = findPSF(Mag, tmpPhase, Bg[ii], I, loads, 
                                    DefocusK, tcz+DelX[ii,2], Mask, 
                                    SubPixelZeros, StartInd, EndInd,
                                    SubPixel, X+DelX[ii,0], Y+DelX[ii,1],
                                    XOffsetPhase, YOffsetPhase)
        
            tPSF.append(BgPSF)
            Norm1_PSFstack.append(tNorm1_PSF)

        tPSF = np.array(tPSF); Norm1_PSFstack = np.array(Norm1_PSFstack)
        DLogL = np.sum(Data*(np.log(tPSF)-np.log(PSFstack))-(tPSF-PSFstack))/tp
        DLogPost += DLogL 
        if DLogPost > np.log(np.random.rand()):

            Z[:,:] = tcz[:,:]
            PSFstack = tPSF
            AcceptX = AcceptX + 1
            lp = np.sum(Data*np.log(PSFstack)-(PSFstack))
            # np.save('./logliketraj.npy', np.sum(Data*np.log(PSFstack)-(PSFstack)))
    if np.sum(loads)==0:
        Norm1_PSFstack = np.zeros_like(PSFstack)
        PSFstack = np.zeros_like(PSFstack)
        PSFstack[0,:,:,:] = Bg[0]
        PSFstack[1,:,:,:] = Bg[1]

    return Z, PSFstack, AcceptX, Norm1_PSFstack, lp
