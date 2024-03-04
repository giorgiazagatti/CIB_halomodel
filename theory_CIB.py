#!/usr/bin/env python
# coding: utf-8

import numpy as np
import yaml
#GZ
#import scipy
#import pandas as pd
#from scipy.optimize import minimize
from yaml import SafeLoader
from utils import *
from HODS_mod import *
from spectraldependence import *
from power_spectrum import *
#from matterPS import * 
from cosmology import *

#--------------------------------open the paramfile, different options available--------------------------------
#with open("paramfile_SPIRE.yaml") as f:
#with open("paramfile_Lenz.yaml") as f:
with open("paramfile_Planck.yaml") as f:
    settings = yaml.load(f, Loader=SafeLoader)

#-----------------------------------------------general settings------------------------------------------------
normalization = settings['options']['normalization']
shot          = settings['options']['shot']
redshift_path = settings['options']['redshift']
redshift      = np.loadtxt(redshift_path)

#---------------------------------------------------------------------------------------------------------------
#info about the experiment: units of measure, color correction, shot, emissivities and effective freq for each channel
exp_settings   = settings['frequencies']

experiments = []
for i in exp_settings:
    experiments.append(exp_settings[i])

unit           = []
color_corr     = []
shot_noise     = []
effective_freq = []
emissivities   = []
for exp in experiments:
    if 'units' in exp:
        unit.append(exp['units'])
    if 'cc' in exp:
        color_corr.append(exp['cc'])
    if 'SN' in exp:
        shot_noise.append(exp['SN'])
    if 'eff_freq' in exp:
        effective_freq.append(exp['eff_freq'])
    if 'emissivity' in exp:
        emissivities.append(exp['emissivity'])
color_corr = np.array(color_corr)
shot_noise = np.array(shot_noise)

#---------------------------------------------------------------------------------------------------------------
#info about the parameters: both fixed and variable parameters
param               = settings['parameters']
cosmological_param  = param['cosmology']
clust_param         = param['clustering']
PS_param            = param['power_spectra']

#compute cosmological parameters, matter power spectrum and CMB
cosmo_param = cosmo_param(redshift, cosmological_param, cosmo)

h     = cosmo_param.compute_params()[0]
dV_dz = cosmo_param.compute_params()[1]

k_array  = cosmo_param.read_matter_PS()[0]
Pk_array = cosmo_param.read_matter_PS()[1]

#GZ
#dl_CMB = cosmo_param.read_CMB()

#---------------------------------------------------------------------------------------------------------------
#set mass range with the correct unit Msun*h**-1
logmass = np.arange(2, 15, 0.1)
mh      = 10 ** logmass / (h ** -1)

# set ell range
ell_min = 1
ell_max = 2000
ell     = np.arange(ell_min, ell_max+1)
lenLs   = len(ell)

# define the mass overdensity
delta_200 = 200

# set the normalization
if normalization == 1:
    facto = ell * (ell + 1) / (2 * np.pi)  # Dls
else:
    facto = 1.0  # Cls

# compute frequency dependence
emission = freq_dep(redshift, emissivities, effective_freq, unit)
#--------------------------------------------------compute utils--------------------------------------------------
# set the hydrostatic mass bias REMEMBER the unit for mass is Msun for the tSZ pressure profile
B = 1.0 / (1 - clust_param['b'])
M_tilde = 10 ** logmass / B

instance_200 = u_p_nfw_hmf_btSZ_bCIB(
    ell, k_array, Pk_array, mh, M_tilde, redshift, delta_200
)
def foregrounds_call(shot_noise,clust_param, PS_param, color_corr):
    ##compute HODs and mean_gal
    instance_HOD = hod_ngal(mh, redshift, clust_param, instance_200)
    # ---------------------------------------------------------------------------------------
    # call foreground class and compute all power spectra
    #correlation_array = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    correlation_array = np.array([1., fixed_param['corr12'], fixed_param['corr13'], 1., fixed_param['corr23'], 1.])
    #correlation_array = np.array([1., 0.9493, 0.8113, 1., 0.9281, 1.])
    nnu = 3
    shot_correlations = np.zeros((nnu, nnu))
    index = 0 
    for i in range(nnu):
        for j in range(i, nnu):
            shot_correlations[i,j] = correlation_array[index]
            index = index + 1
                
    calibration_factors = np.array([fit_param['cal0'], fit_param['cal1'], fit_param['cal2']]) 

    spectra = foregrounds(
        shot_correlations,
        color_corr, 
        calibration_factors,
        ell,
        mh,
        redshift,
        instance_HOD,
        instance_200,
        instance_500,
        dV_dz,
        shot_spire,
        #corr_matrix,
        emission
    )
    (   
        Cl_cib_1h_EP,
        Cl_cib_2h_EP,
        Cl_cib_1h_LP,
        Cl_cib_2h_LP,
        Cl_cib_1h_mix,
        Cl_cib_2h_mix
    ) = spectra.halo_terms_CIB_addison()

    if shot == 0:
        Cl_cibp = spectra.CIB_poisson_spt()
    else:
        Cl_cibp = spectra.CIB_poisson_spire()

    cib_1h_EP = Cl_cib_1h_EP * facto 
    cib_1h_LP = Cl_cib_1h_LP * facto
    cib_2h_EP = Cl_cib_2h_EP * facto
    cib_2h_LP = Cl_cib_2h_LP * facto
    cib_1h_mix = Cl_cib_1h_mix * facto
    cib_2h_mix = Cl_cib_2h_mix * facto
    cibc_EP   = (Cl_cib_2h_EP + Cl_cib_1h_EP) * facto 
    cibc_LP   = (Cl_cib_2h_LP + Cl_cib_1h_LP) * facto 
    cibc_mix  = (Cl_cib_1h_mix + Cl_cib_2h_mix) * facto
    cibc      = cibc_EP + cibc_LP + cibc_mix
    cibp      = Cl_cibp * facto
    cib       = cibc + cibp

    return cib

if domcmc==True:

    n_spec = 6

    cls_data_Planck = np.loadtxt('Lenz_cls.txt')[:,3:]
    err_data_Planck = np.loadtxt('Lenz_err.txt')[:,3:]

    obs_ps  = []
    err_data = []

    for i in range(n_spec):
        for j in cls_data_Planck[i,:]:
            obs_ps.append(j)
        for k in err_data_Planck[i,:]:
            err_data.append(k)
    obs_ps = np.array(obs_ps)
    err_data = (1/np.array(err_data))**2

    expected_param = np.array([12.07, 10.8, 1.5, 225, 1454, 5628, 1., 1., 1.])

    n_samples = 600000
    nwalkers  = 32
    nthreads  = 1
    ndim      = len(expected_param)

    def loglike(model, spectrum, error):
        return -0.5*(((model-spectrum)**2)*error).sum()

    def logprior(params):
        if 10.7<=params[0]<=12.8 and 10.5<=params[1]<=12.8 and 0.2<=params[2]<=3.5 and 50<=params[3]<=500 and 400<=params[4]<=4000 and 200<=params[5]<=8000 and 0.9064<=params[6]<=1.0936 and 0.268<=params[7]<=1.732 and 0.232<=params[8]<=1.768:
            cv_353       = 1.0
            cv_545       = 1.0
            cv_857       = 1.0
            sigma_353    = 0.0156
            sigma_545    = 0.122
            sigma_857    = 0.128
            return np.log(1.0/(np.sqrt(2*np.pi)*sigma_353)) + np.log(1.0/(np.sqrt(2*np.pi)*sigma_545)) +np.log(1.0/(np.sqrt(2*np.pi)*sigma_857)) -0.5*(params[6]-cv_353)**2/sigma_353**2 -0.5*(params[7]-cv_545)**2/sigma_545**2 -0.5*(params[8]-cv_857)**2/sigma_857**2 
        return -np.inf

    def logprob(params, var_params, fixed_param, color_corr, spectrum, error, nspec):
        lp = logprior(params)
        if not np.isfinite(lp):
            return -np.inf
        dict_params = {'LogMmin_EP': params[0],
                       'LogMmin_LP': params[1],
                       'alpha_LP': params[2],
                       'shot0': params[3],
                       'shot1': params[4],
                       'shot2': params[5],
                       'cal0': params[6],
                       'cal1': params[7],
                       'cal2': params[8],}
        cl_tot = foregrounds_call(var_param, dict_params, fixed_param, color_corr)
        cl_cib = []
        for i in range(n_spec):
            for j in cl_tot[i,:]:
                cl_cib.append(j)
        cl_cib = np.array(cl_cib)
        return lp + loglike(cl_cib, spectrum, error)

    filename = "chain_Lenz_mix_corr_ones_cutP14.h5"
    backend = emcee.backends.HDFBackend(filename)

    # Check if there are already samples in the backend
    #if backend.iteration > 0:
        # Load the last position of the walkers
        #last_state = backend.get_last_sample()
        #pos = last_state.coords
        #print("Resuming from previous run")
    #else:
        # Start from the initial position
        #pos = [expected_param + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
        #print("Starting from scratch")

    #index = 0
    #autocorr = np.empty(n_samples)

    #old_tau = np.inf
    #print('start mcmc')
    #with Pool() as pool:
        #sampler = emcee.EnsembleSampler(
            #nwalkers,
            #ndim,
            #logprob,
            #args=(var_param, fixed_param, color_corr, obs_ps, err_data, n_spec),
            #threads=nthreads,
            #pool=pool,
            #backend=backend
        #)

        #for sample in sampler.sample(pos, iterations=n_samples, progress=True):
            #if sampler.iteration % 500:
                #continue

            #tau = sampler.get_autocorr_time(tol=0)
            #autocorr[index] = np.mean(tau)
            #index += 1

            #converged = np.all(tau * 100 < sampler.iteration)
            #converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            #if converged:
                #break
            #old_tau = tau

    backend.reset(nwalkers, ndim)
    index = 0
    autocorr = np.empty(n_samples)
#
    old_tau = np.inf
    print('start mcmc')
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
                  nwalkers,
                  ndim,
                  logprob,
                  args=(var_param, fixed_param, color_corr, obs_ps, err_data, n_spec),
                  threads=nthreads,
                  pool=pool,
                  backend=backend
                  )
        pos = [expected_param + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        for sample in sampler.sample(pos, iterations=n_samples, progress=True):
            if sampler.iteration % 500: 
                continue            
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index]=np.mean(tau)
            index +=1
            
            converged = np.all(tau*100<sampler.iteration)
            converged &= np.all(np.abs(old_tau-tau)/tau < 0.01)
            if converged:  
                break
            old_tau = tau
    print('end mcmc')

else:
    print('computed PS without doing mcmc')
    cl_cib = foregrounds_call(var_param, fit_param, fixed_param, color_corr)
    np.savetxt('./cl_cib_Planck_Lenz_corr_pl.txt', cl_cib)
