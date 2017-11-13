import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig, inv
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def get_power_spectral_density_matrix(observation, mask=None, frm_expand=1):
    """
    Calculates the weighted power spectral density matrix.

    This does not yet work with more than one target mask.

    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]

    normalization = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-6)
    
    if frm_expand == 1:
        psd = np.einsum('...dt,...et->...de', mask * observation,
                    observation.conj())
    else:
        psd = np.einsum('...dt,...et->...de', np.sqrt(mask) * observation,
                        np.sqrt(mask) * observation.conj())
    psd /= normalization
    return psd


def get_pca_vector(target_psd_matrix):
    """
    Returns the beamforming vector of a PCA beamformer.
    :param target_psd_matrix: Target PSD matrix
        with shape (..., sensors, sensors)
    :return: Set of beamforming vectors with shape (..., sensors)
    """
    # Save the shape of target_psd_matrix
    shape = target_psd_matrix.shape

    # Reduce independent dims to 1 independent dim
    target_psd_matrix = np.reshape(target_psd_matrix, (-1,) + shape[-2:])

    # Calculate eigenvals/vecs
    eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix)
    # Find max eigenvals
    vals = np.argmax(eigenvals, axis=-1)
    # Select eigenvec for max eigenval
    beamforming_vector = np.array(
            [eigenvecs[i, :, vals[i]] for i in range(eigenvals.shape[0])])
    # Reconstruct original shape
    beamforming_vector = np.reshape(beamforming_vector, shape[:-1])

    return beamforming_vector


def get_mvdr_vector(atf_vector, noise_psd_matrix):
    """
    Returns the MVDR beamforming vector: h = (Npsd^-1)*A / (A^H*(Npsd^-1)A)
    :param atf_vector: Acoustic transfer function vector
        with shape (..., bins, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """
    while atf_vector.ndim > noise_psd_matrix.ndim - 1:
        noise_psd_matrix = np.expand_dims(noise_psd_matrix, axis=0)

    # Make sure matrix is hermitian
    noise_psd_matrix = 0.5 * (
        noise_psd_matrix + np.conj(noise_psd_matrix.swapaxes(-1, -2)))

    numerator = solve(noise_psd_matrix, atf_vector)
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

    return beamforming_vector


def apply_sdw_mwf(mix, target_psd_matrix, noise_psd_matrix, mu=1, corr=None):
    """
    Apply speech distortion weighted MWF: h = Tpsd * e1 / (Tpsd + mu*Npsd) 
    :param mix: the signal complex FFT
    :param target_psd_matrix (bins, sensors, sensors) 
    :param noise_psd_matrix
    :param mu: the lagrange factor
    :return 
    """
    bins, sensors, frames = mix.shape
    ref_vector = np.zeros((sensors,1), dtype=np.float)    
    if corr is None:
        ref_ch = 0
    else: # choose the channel with highest correlation with the others
        corr=corr.tolist()        
        while len(corr) > sensors:
            corr.remove(np.min(corr))
        ref_ch=np.argmax(corr)
    ref_vector[ref_ch,0]=1 
    
    mwf_vector = solve(target_psd_matrix + mu*noise_psd_matrix, target_psd_matrix[:,:,ref_ch])
    return np.einsum('...a,...at->...t', mwf_vector.conj(), mix)
     
    
def apply_r1_mwf(mix, target_psd_matrix, noise_psd_matrix, mu=1, corr=None,
                 evd=False, gevd=False):
    """
    Apply rank-1 MWF: h = (Npsd^-1)*Tpsd*e1 / (mu+lamda)
    :param mix: the signal complex FFT
    :param target_psd_matrix (bins, sensors, sensors) 
    :param noise_psd_matrix
    :param mu: tradeoff between speech distortion and noise reduction
    :options: evd or gevd, low-rank approximatino
    :return the processed result
    """
    bins, sensors, frames = mix.shape
    
    # EVD and GSVD reconstructs the target matrix to be rank-1 
    if evd is True:
        evd_vector = np.empty((bins, sensors), dtype=np.complex)   
        for f in range(bins): 
            evd_vector[f,:] = get_pca_vector(target_psd_matrix[f,:,:])                       
            b0 = evd_vector[f,:][:,np.newaxis] 
            b1 = evd_vector[f,:][np.newaxis,:]
            tmp = np.dot(b0, b1.conj())
            tr1 = np.trace(target_psd_matrix[f,:,:])
            tr2 = np.trace(tmp)
            target_psd_matrix[f,:,:] = tmp*tr1/tr2
    elif gevd is True:  
        gevd_vals, gevd_matrix = get_gevd_vals_vecs(target_psd_matrix, noise_psd_matrix)   
        for f in range(bins):
            gsvd_matrix = inv(gevd_matrix[f,:,:]) 
            # attention: gevd_matrix is invertable theoritically, no direct way to compute gsvd in python
            # the reconstruction vector is indeed from the inverse of the GEVD matrix
	    # see ref: low rank approximation based multichannel Wiener filter algorithms for noise reduction with application in cochlear implants
            gsvd_vector = gsvd_matrix.conj().T[:,np.argmax(gevd_vals[f,:])]
            b0 = gsvd_vector[:,np.newaxis] 
            b1 = gsvd_vector[np.newaxis,:]
            tmp = np.dot(b0, b1.conj())
            tr1 = np.trace(target_psd_matrix[f,:,:])
            tr2 = np.trace(tmp)
            target_psd_matrix[f,:,:] = tmp*tr1/tr2
    
    eig_values, _ = get_gevd_vals_vecs(target_psd_matrix, noise_psd_matrix)
    lamda = np.sum(eig_values, axis=1)  # np.max ? np.sum
    numerator = solve(noise_psd_matrix, target_psd_matrix)

    ref_vector = np.zeros((sensors,1), dtype=np.float)    
    if corr is None:
        ref_ch = sensors-1
    else: # choose the channel with highest correlation with the others
        corr=corr.tolist()        
        while len(corr) > sensors:
            corr.remove(np.min(corr))
        ref_ch=np.argmax(corr)
    ref_vector[ref_ch,0]=1  
    
    if mu is 'rnn': # GEV condition      
        rnn=1  # residual noise power, ignore the scale issue because of normaliation  
        mu_plus_lamda=np.sqrt(target_psd_matrix[:,ref_ch,ref_ch]*lamda/rnn)
        mu_data = mu_plus_lamda-lamda
        mwf_vector = np.dot(numerator, ref_vector)[:,:,0]/(mu_plus_lamda)[:,np.newaxis] 
    else:
        mwf_vector = np.dot(numerator, ref_vector)[:,:,0]/(mu+lamda)[:,np.newaxis]
    
    return np.einsum('...a,...at->...t', mwf_vector.conj(), mix)

    
def apply_vs_filter(mix, target_psd_matrix, noise_psd_matrix, mu=1, corr=None, frm_exp=1):
    """
    Apply variable span filter: h = sum_q ( b*b^H / (mu+lamda) )*Xpsd*e1
    :param mix: the signal complex FFT
    :param target_psd_matrix (bins, sensors, sensors) 
    :param noise_psd_matrix
    :return the processed result
    """
    bins, sensors, frames = mix.shape
    
    ref_vector = np.zeros((sensors,1), dtype=np.float)    
    if corr is None:
        ref_ch = sensors-1
    else: # choose the channel with highest correlation with the others
        corr=corr.tolist()        
        while len(corr) > sensors/frm_exp:
            corr.remove(np.min(corr))
        ref_ch=np.argmax(corr)
    ref_vector[ref_ch,0]=1 
    
    Qrank = 1
    #rnn=1
    vs_vector = np.zeros((bins, sensors), dtype=np.complex)  
    eig_values, eig_vectors = get_gevd_vals_vecs(target_psd_matrix, noise_psd_matrix)
    for f in range(bins):
        tmp = np.zeros((sensors, sensors), dtype=np.complex)
        for q in range(Qrank):
            b0 = eig_vectors[f,:,sensors-1-q][:,np.newaxis]
            b1 = eig_vectors[f,:,sensors-1-q][np.newaxis,:]
            tmp += np.dot(b0, b1.conj()) / (mu+eig_values[f,sensors-1-q])
            # rnn condition            
            #mu_plus_lamda=np.sqrt(target_psd_matrix[f,ref_ch,ref_ch]*eig_values[f,sensors-1-q]/rnn)
            #tmp += np.dot(b0, b1.conj()) / mu_plus_lamda
        vs_vector[f,:] = np.dot(tmp, target_psd_matrix[f,:,0])
     
    return np.einsum('...a,...at->...t', vs_vector.conj(), mix)
 
     
def get_gevd_vals_vecs(target_psd_matrix, noise_psd_matrix):
    """
    Returns the eigenvalues and eigenvectors of GEVD
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of eigen values  with shape (bins, sensors)
	         eigenvectors (bins, sensors, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    eigen_values = np.empty((bins, sensors), dtype=np.complex)
    eigen_vectors = np.empty((bins, sensors, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
        # values in increasing order									   
        eigen_values[f,:] = eigenvals
        eigen_vectors[f, :] = eigenvecs
    return eigen_values, eigen_vectors
    
def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]
    return beamforming_vector


def blind_analytic_normalization(vector, noise_psd_matrix):
    bins, sensors = vector.shape
    normalization = np.zeros(bins)
    for f in range(bins):
        normalization[f] = np.abs(np.sqrt(np.dot(
                np.dot(np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                       noise_psd_matrix[f]), vector[f, :])))
        normalization[f] /= np.abs(np.dot(
                np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                vector[f, :]))

    return vector * normalization[:, np.newaxis]


def apply_beamforming_vector(vector, mix):
    return np.einsum('...a,...at->...t', vector.conj(), mix)


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                         setup=None, corr=None):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)
    if noise_mask is None:
        noise_mask = np.clip(1 - target_mask, 1e-6, 1)

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask)
    
    if setup is None:
        setup={}
        setup['output_type'] = 'gev'
        setup['gev_ban'] = True
    else:
        output_type = setup['output_type']
        mu = setup['mwf_mu']
        
    bins, sensors, _ = target_psd_matrix.shape
    if output_type == 'gev':
		# the priciple eigenvector of (noise_psd_matrix)^-1*(target_psd_matrix)
        W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)
        if setup['gev_ban'] is True:
            W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)
        
        output = apply_beamforming_vector(W_gev, mix)
		
    if output_type == 'mvdr':
		# MVDR with options: diagnal-loading
        # steering vector is the principle eigenvector of target_psd_matrix        
        steering_vector = np.empty((bins, sensors), dtype=np.complex)      
        for f in range(bins):
            steering_vector[f,:] = get_pca_vector(target_psd_matrix[f,:,:])
	
        W_mvdr = get_mvdr_vector(steering_vector, noise_psd_matrix)
        output = apply_beamforming_vector(W_mvdr, mix)
		
    if output_type == 'sdw-mwf':
         # speech distortion weighted MWF 
		output = apply_sdw_mwf(mix, target_psd_matrix, noise_psd_matrix, mu, corr)
		
    if output_type == 'r1-mwf':
		# rank-1 MWF with options: EVD and GEVD 
		# ref: on optimal frequency-domain MWF for noise reduction
		#      low-rank approximation based MWF for noise reduction ...
        output = apply_r1_mwf(mix, target_psd_matrix, noise_psd_matrix, mu, corr,
                              evd=setup['r1-mwf_evd'], gevd=setup['r1-mwf_gevd'])
    
    if output_type == 'vs':
		# variable span linear filter
		# ref: noise reduction with optimal variable span linear filters
		output = apply_vs_filter(mix, target_psd_matrix, noise_psd_matrix, mu, corr)
    
    return output.T
                            
                             
