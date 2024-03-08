#!/usr/bin/env python
# coding: utf-8

import numpy as np
import yaml
from yaml import SafeLoader
from utils import *
from HODS_mod import *
from spectraldependence import *
from power_spectrum import *
from matterPS import * 
from cosmology import *

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------open and read the paramfile, different options available----------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#with open("paramfile_SPIRE.yaml") as f:
#with open("paramfile_Lenz.yaml") as f:
with open("paramfile_Planck.yaml") as f:
    settings = yaml.load(f, Loader=SafeLoader)

#-----------------------------------------------general settings------------------------------------------------
read_matterPS = settings['options']['read_matterPS']
normalization = settings['options']['normalization']
redshift_path = settings['options']['redshift']
redshift      = np.loadtxt(redshift_path)

# set the normalization
if normalization == 1:
    facto = ell * (ell + 1) / (2 * np.pi)  # Dls
else:
    facto = 1.0  # Cls

#------------------------------------------expertiment-related features------------------------------------------
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

#------------------------------------------------paramters setting------------------------------------------------
param               = settings['parameters']
cosmological_param  = param['cosmology']
fixed_param         = param['fixed']
clust_param         = param['clustering']
PS_param            = param['power_spectra']

#compute cosmological parameters, matter power spectrum
cosmo_param = cosmo_param(redshift, cosmological_param, cosmo)

h, dV_dz = cosmo_param.compute_params()

if read_matterPS == True:
    k_array, Pk_array = cosmo_param.read_matter_PS()
else:
    compute_PS = matter_PS.lin_matter_PS()
    k_array = compute_PS()[0]
    Pk_array = cosmo_PS()[2]


#----------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------Other settings-------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# set mass range
logmass = np.arange(2, 15, 0.1)
mh      = 10 ** logmass / (h ** -1)

# set ell range
ell_min = 1
ell_max = 2000
ell     = np.arange(ell_min, ell_max+1)
lenLs   = len(ell)

# set the mass overdensity
delta_200 = 200

# set the hydrostatic mass bias
B = 1.0 / (1 - clust_param['b'])
M_tilde = 10 ** logmass / B


#----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------Frequency dependence and utils computation-----------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# compute frequency dependence
emission = freq_dep(redshift, emissivities, effective_freq, unit, fixed_param)

#compute utils
instance_200 = u_p_nfw_hmf_btSZ_bCIB(
    ell, k_array, Pk_array, mh, M_tilde, redshift, delta_200
)

#----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------CIB power spectrum computation-----------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
def CIB_powerspectrum(clust_param, PS_param, color_corr):
#----------------------------------------------------HOD model----------------------------------------------------
    instance_HOD = hod_ngal(mh, redshift, clust_param, instance_200)

#------------------------------------------Calibration and correlations-------------------------------------------
    nnu          = len(effective_freq)
    nspec        = int(nnu * (nnu - 1) / 2 + nnu)

    if nnu == 3:
        correlation_array = np.array([1., PS_param['corr01'], PS_param['corr02'], 1., PS_param['corr12'], 1.])
        calibration_factors = np.array([PS_param['cal0'], PS_param['cal1'], PS_param['cal2']])
    else:
        correlation_array = np.array([1., PS_param['corr01'], PS_param['corr02'], PS_param['corr03'], 1., PS_param['corr12'], PS_param['corr13'], 1., PS_param['corr23'], 1.])
        calibration_factors = np.array([PS_param['cal0'], PS_param['cal1'], PS_param['cal2'], PS_param['cal3']])

    shot_correlations = np.zeros((nnu, nnu))
    index = 0 
    for i in range(nnu):
        for j in range(i, nnu):
            shot_correlations[i,j] = correlation_array[index]
            index = index + 1 

#--------------------------------------------Power spectra computation---------------------------------------------
    spectra = foregrounds(
        shot_correlations,
        color_corr, 
        calibration_factors,
        ell,
        mh,
        redshift,
        instance_HOD,
        instance_200,
        dV_dz,
        shot_noise,
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

    Cl_cibp = spectra.CIB_poisson()

    cib_1h_EP = Cl_cib_1h_EP * facto 
    cib_1h_LP = Cl_cib_1h_LP * facto
    cib_1h_mix = Cl_cib_1h_mix * facto
    cib_2h_EP = Cl_cib_2h_EP * facto
    cib_2h_LP = Cl_cib_2h_LP * facto
    cib_2h_mix = Cl_cib_2h_mix * facto
    cibc_EP   = (Cl_cib_2h_EP + Cl_cib_1h_EP) * facto 
    cibc_LP   = (Cl_cib_2h_LP + Cl_cib_1h_LP) * facto 
    cibc_mix  = (Cl_cib_1h_mix + Cl_cib_2h_mix) * facto
    cibc      = cibc_EP + cibc_LP + cibc_mix
    cibp      = Cl_cibp * facto
    cib       = cibc + cibp

    return cib

cl_cib = CIB_powerspectrum(clust_param, PS_param, color_corr)