from astropy.cosmology import Planck18
import numpy as np

cosmo     = Planck18

class cosmo_param:
    def __init__(self, redshift, cosmological_param, cosmo):
        self.redshift           = redshift
        self.cosmological_param = cosmological_param
        self.cosmo              = cosmo

    def read_params(self):
        T_CMB = self.cosmological_param['T_CMB']
        
        #direi che se non dobbiamo calcolare il matter PS, ma leggerlo solo, questi non servono
        #par_tau          = 0.0544
        #par_ns           = 0.9649
        #par_As           = 2.101e-9
        #par_pivot_scalar = 0.05

        return T_CMB

    def compute_params(self):
        cosmo = self.cosmo
        h     = cosmo.h
        H_0   = cosmo.H(0).value
        om_l  = cosmo.Ode0
        om_m  = cosmo.Om0
        om_b  = cosmo.Ob0

        par_h       = h * 100
        par_omega_m = om_m * h ** 2
        par_omega_b = om_b * h ** 2

        D_a   = cosmo.angular_diameter_distance(self.redshift).value
        D_h   = self.cosmological_param['c'] / H_0
        dV_dz = D_h * ((1 + self.redshift) ** 2 * D_a ** 2) / np.sqrt(om_l + (1 + self.redshift) ** 3 * om_m) / h ** -3
        
        return h, dV_dz

    def read_matter_PS(self):
        matter_PS = np.loadtxt(self.cosmological_param['matter_PS'])
        k_array_T = matter_PS[:,0]
        k_array   = np.transpose(k_array_T)

        Pk_array_T = matter_PS[:,1:]
        Pk_array   = np.transpose(Pk_array_T)

        return k_array, Pk_array

    def read_CMB(self):
        dl_CMB = np.loadtxt(self.cosmological_param['CMB'])

        return dl_CMB

