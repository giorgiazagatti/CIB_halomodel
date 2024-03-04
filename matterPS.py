import numpy as np
import camb
from camb import model, initialpower

class matter_PS_CMB:
    def __init__(self, par_h, par_omega_b, par_omega_m, par_tau, par_ns, par_As, par_pivot_scalar, redshift, l2_max, lenLs, dl_CMB):
        self.par_h            = par_h
        self.par_omega_b      = par_omega_b
        self.par_omega_m      = par_omega_m
        self.par_tau          = par_tau
        self.par_ns           = par_ns
        self.par_As           = par_As
        self.par_pivot_scalar = par_pivot_scalar
        self.redshift         = redshift
        self.l2_max           = l2_max
        self.lenLs            = lenLs
        self.dl_CMB           = dl_CMB

    #computation of the lin matter PS
    def lin_matter_PS(self):
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.par_h, ombh2=self.par_omega_b, omch2=self.par_omega_m - self.par_omega_b, tau=self.par_tau
        )
        pars.InitPower.set_params(ns=self.par_ns, As=self.par_As, pivot_scalar=self.par_pivot_scalar)
        pars.set_matter_power(redshifts=self.redshift, kmax=1e2)
        pars.set_for_lmax(self.l2_max, lens_potential_accuracy=0)

        pars.NonLinear = model.NonLinear_none
        results        = camb.get_results(pars)
        k_array, z, Pk_array = results.get_matter_power_spectrum(
            minkh=1e-3, maxkh=1e4, npoints=500
        )    
        return k_array, z, Pk_array

    #CMB computation
    def CMB_comp(self):
        pars    = camb.CAMBparams()
        pars.set_cosmology(
             H0=self.par_h, ombh2=self.par_omega_b, omch2=self.par_omega_m - self.par_omega_b, tau=self.par_tau
        )
        pars.set_for_lmax(self.l2_max, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        powers  = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        totCL   = powers["total"]
        self.dl_CMB[:, 1] = totCL[:self.lenLs, 0]
        Dl_CMB = self.dl_CMB[:,1]
        return Dl_CMB
