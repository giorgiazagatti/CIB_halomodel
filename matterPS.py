import numpy as np
import camb
from camb import model, initialpower

class matter_PS:
    #def __init__(self, par_h, par_omega_b, par_omega_m, par_tau, par_ns, par_As, par_pivot_scalar, redshift, l2_max, lenLs, dl_CMB):
    def __init__(self, redshift, h, cosmo_param):
        #self.par_h            = par_h
        #self.par_omega_b      = par_omega_b
        #self.par_omega_m      = par_omega_m
        #self.par_tau          = par_tau
        #self.par_ns           = par_ns
        #self.par_As           = par_As
        #self.par_pivot_scalar = par_pivot_scalar
        self.redshift         = redshift
        self.h                = h
        #self.l2_max           = l2_max
        self.cosmo_param = cosmo_param

    #computation of the lin matter PS
    def lin_matter_PS(self):
        par_h = self.h * 100
        par_omega_b = self.cosmo_param.om_b * self.h ** 2
        par_omega_m = self.cosmo_param.om_m * self.h ** 2

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=par_h, ombh2=par_omega_b, omch2=par_omega_m - par_omega_b, tau=self.cosmo_param.par_tau
        )
        pars.InitPower.set_params(ns=self.cosmo_param.par_ns, As=self.cosmo_param.par_As, pivot_scalar=self.cosmo_param.par_pivot_scalar)
        pars.set_matter_power(redshifts=self.redshift, kmax=1e2)
        #pars.set_for_lmax(self.l2_max, lens_potential_accuracy=0)

        pars.NonLinear = model.NonLinear_none
        results        = camb.get_results(pars)
        k_array, z, Pk_array = results.get_matter_power_spectrum(
            minkh=1e-3, maxkh=1e4, npoints=500
        )    
        return k_array, z, Pk_array