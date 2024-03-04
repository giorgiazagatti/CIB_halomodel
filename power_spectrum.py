import numpy as np
from scipy import special
from scipy import interpolate
from numpy import trapz


class foregrounds:
    def __init__(
        self,
        shot_correlations,
        color_corr,
        calibration,
        ell,
        mh,
        redshift,
        instance_HOD,
        instance_200,
        dV_dz,
        shot_spire,
        emission
    ):

        self.shot_correlations = shot_correlations
        self.color_corr        = color_corr
        self.calibration       = calibration
        self.ell               = ell
        self.mass              = mh
        self.redshift          = redshift
        self.instance_HOD      = instance_HOD
        self.instance_200      = instance_200
        self.dV_dz             = dV_dz
        self.shot_spire        = shot_spire
        self.emission          = emission


    def halo_terms_CIB_addison(self):
        #compute the conversion factors
        cf_cib = []

        for nu, i in enumerate(self.emission.unit):
            if i=='muK^2':
                cf_cib.append(1 /self.emission.dBdT_num_dust[nu])
            else:
                cf_cib.append(1.0)
        cf_cib = np.array(cf_cib)

        # define the initial
        intmass_2h_EP  = np.zeros([len(self.instance_200.kh), len(self.redshift), len(self.mass)])
        intmass_1h_EP  = np.zeros([len(self.instance_200.kh), len(self.redshift), len(self.mass)])
        intmass_2h_LP  = np.zeros([len(self.instance_200.kh), len(self.redshift), len(self.mass)])
        intmass_1h_LP  = np.zeros([len(self.instance_200.kh), len(self.redshift), len(self.mass)])
        intmass_1h_mix = np.zeros([len(self.instance_200.kh), len(self.redshift), len(self.mass)])
        
        intred_2h_EP  = np.zeros([len(self.instance_200.kh), len(self.redshift)])
        intred_1h_EP  = np.zeros([len(self.instance_200.kh), len(self.redshift)])
        intred_2h_LP  = np.zeros([len(self.instance_200.kh), len(self.redshift)])
        intred_1h_LP  = np.zeros([len(self.instance_200.kh), len(self.redshift)])
        intred_1h_mix = np.zeros([len(self.instance_200.kh), len(self.redshift)])
        intred_2h_mix = np.zeros([len(self.instance_200.kh), len(self.redshift)])

        Cl_2h_EP  = np.zeros([self.emission.n_spec, len(self.instance_200.kh)])
        Cl_1h_EP  = np.zeros([self.emission.n_spec, len(self.instance_200.kh)])
        Cl_2h_LP  = np.zeros([self.emission.n_spec, len(self.instance_200.kh)])
        Cl_1h_LP  = np.zeros([self.emission.n_spec, len(self.instance_200.kh)])
        Cl_2h_mix = np.zeros([self.emission.n_spec, len(self.instance_200.kh)])
        Cl_1h_mix = np.zeros([self.emission.n_spec, len(self.instance_200.kh)])
        # compute the one and two halo terms
        minred = 5  # lower limit on redshift as in Addison 2012 (z>0.25)
        spec = 0

        for nu1 in range(self.emission.n_nu):
            for nu2 in range(nu1, self.emission.n_nu):
                emission_EP  = ( self.emission.j_nu_EP_step[nu1] * self.emission.j_nu_EP_step[nu2]) * 1.0 / (self.instance_HOD.ngal_EP_200c * self.dV_dz) ** 2
                emission_LP  = ( self.emission.j_nu_LP_step[nu1] * self.emission.j_nu_LP_step[nu2])  * 1.0 / (self.instance_HOD.ngal_LP_200c * self.dV_dz) ** 2
                emission_mix = ( self.emission.j_nu_EP_step[nu1]*self.emission.j_nu_LP_step[nu2]  + self.emission.j_nu_EP_step[nu2]*self.emission.j_nu_LP_step[nu1]) * 1.0 / (self.instance_HOD.ngal_EP_200c * self.instance_HOD.ngal_LP_200c * self.dV_dz**2)

                for k in range(len(self.instance_200.kh)):

                    intmass_2h_EP[k, :, :] = (
                        self.instance_200.dndM
                        * self.instance_200.bias_cib
                        * self.instance_HOD.Nbra_EP[np.newaxis, :]
                        * self.instance_200.u_c[:, :, k]
                    )
                    intmass_1h_EP[k, :, :] = self.instance_200.dndM * (
                        2
                        * self.instance_HOD.Ncent_EP[np.newaxis, :]
                        * self.instance_HOD.Nsat_EP[np.newaxis, :]
                        * self.instance_200.u_c[:, :, k]
                        + self.instance_HOD.Nsat_EP[np.newaxis, :] ** 2 * self.instance_200.u_c[:, :, k] ** 2
                    )

                    intmass_2h_LP[k, :, :] = (
                        self.instance_200.dndM
                        * self.instance_200.bias_cib
                        * self.instance_HOD.Nbra_LP[np.newaxis, :]
                        * self.instance_200.u_c[:, :, k]
                    )
                    intmass_1h_LP[k, :, :] = self.instance_200.dndM * (
                        2
                        * self.instance_HOD.Ncent_LP[np.newaxis, :]
                        * self.instance_HOD.Nsat_LP[np.newaxis, :]
                        * self.instance_200.u_c[:, :, k]
                        + self.instance_HOD.Nsat_LP[np.newaxis, :] ** 2 * self.instance_200.u_c[:, :, k] ** 2
                    )
                    intmass_1h_mix[k,:,:] = self.instance_200.dndM * (((
                        self.instance_HOD.Ncent_EP[np.newaxis, :]
                        * self.instance_HOD.Nsat_LP[np.newaxis, :] +
                        self.instance_HOD.Ncent_LP[np.newaxis,:]
                        * self.instance_HOD.Nsat_EP[np.newaxis,:])
                        * self.instance_200.u_k[:, :, k]
                        + self.instance_HOD.Nsat_EP[np.newaxis, :] * self.instance_HOD.Nsat_LP[np.newaxis,:] * self.instance_200.u_k[:, :, k] ** 2 )
                    )             
                    
                    intred_2h_EP[k, :] = (
                        self.dV_dz
                        * self.instance_200.Pk[:, k]
                        * (trapz(intmass_2h_EP[k, :, :], self.mass, axis=-1)) ** 2
                    )
                    intred_1h_EP[k, :] = self.dV_dz * trapz(
                        intmass_1h_EP[k, :, :], self.mass, axis=-1
                    )

                    intred_2h_LP[k, :] = (
                        self.dV_dz
                        * self.instance_200.Pk[:, k]
                        * (trapz(intmass_2h_LP[k, :, :], self.mass, axis=-1)) ** 2
                    )
                    intred_1h_LP[k, :] = self.dV_dz * trapz(
                        intmass_1h_LP[k, :, :], self.mass, axis=-1
                    )
                    intred_1h_mix[k, :] = self.dV_dz * trapz(intmass_1h_mix[k, :, :], self.mass, axis=-1)
                    intred_2h_mix[k, :] = self.dV_dz * self.instance_200.Pk[:, k] * trapz(intmass_2h_EP[k, :, :], self.mass, axis=-1) * trapz(intmass_2h_LP[k, :, :], self.mass, axis=-1)
                           
                    
                    Cl_2h_EP[spec, k] = trapz(
                        emission_EP[minred:] * intred_2h_EP[k, minred:],
                        self.redshift[minred:],
                    ) * cf_cib[nu1] * cf_cib[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2]) 
                    Cl_1h_EP[spec, k] = trapz(
                        emission_EP[minred:] * intred_1h_EP[k, minred:],
                        self.redshift[minred:],
                    ) * cf_cib[nu1] * cf_cib[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2])
                    Cl_2h_LP[spec, k] = trapz(
                        emission_LP[minred:] * intred_2h_LP[k, minred:],
                        self.redshift[minred:],
                    ) * cf_cib[nu1] * cf_cib[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2])
                    Cl_1h_LP[spec, k] = trapz(
                        emission_LP[minred:] * intred_1h_LP[k, minred:],
                        self.redshift[minred:],
                    ) * cf_cib[nu1] * cf_cib[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2])
                    Cl_2h_mix[spec, k] = trapz(
                        emission_mix[minred:] * intred_2h_mix[k,minred],
                        self.redshift[minred:],
                    ) * cf_cib[nu1] * cf_cib[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2])
                    Cl_1h_mix[spec, k] = trapz(
                        emission_mix[minred:] * intred_1h_mix[k,minred],
                        self.redshift[minred:],
                    ) * cf_cib[nu1] * cf_cib[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2])
                    
                spec = spec + 1

        return Cl_1h_EP, Cl_2h_EP, Cl_1h_LP, Cl_2h_LP, Cl_1h_mix, Cl_2h_mix

    def CIB_poisson_spire(self):
        cl_cibp_spire = np.zeros([self.emission.n_spec, len(self.ell)])
        spec = 0
        for nu1 in range(self.emission.n_nu):
            for nu2 in range(nu1, self.emission.n_nu):
                #shot_noise = np.ones(len(self.ell))
                cl_cibp_spire[spec, :] = self.shot_correlations[nu1, nu2] * np.sqrt(
                    self.shot_spire[nu1] * self.shot_spire[nu2] * np.sqrt(self.calibration[nu1] * self.calibration[nu2]) * np.sqrt(self.color_corr[nu1] * self.color_corr[nu2])
                )  # *1./self.conversion[nu1]*1./self.conversion[nu2]
                spec = spec + 1
        return cl_cibp_spire
