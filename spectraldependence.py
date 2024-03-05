import numpy as np
from scipy import interpolate

class freq_dep:
    def __init__(self, redshift, emissivities, effective_freq, unit, fixed_param):
        self.redshift       = redshift
        self.emissivities   = emissivities
        self.effective_freq = effective_freq
        self.unit           = unit
        self.fixed_param    = fixed_param
        self.n_nu_spec()
        self.CIB_fdep()
        self.dBdT_dust()

    def n_nu_spec(self):
        
        n_nu   = len(self.effective_freq)
        n_spec = int(n_nu * (n_nu - 1) / 2 + n_nu)

        self.n_nu   = n_nu
        self.n_spec = n_spec

        return

    def CIB_fdep(self):

        emissivities_EP = []
        emissivities_LP = []
        for i in range(len(self.emissivities)):
            emissivities_EP.append(np.loadtxt(self.emissivities[i]['EP_em']))
            emissivities_LP.append(np.loadtxt(self.emissivities[i]['LP_em']))
        emissivities_EP = np.array(emissivities_EP)
        emissivities_LP = np.array(emissivities_LP)

        # interpolate emissivity in the redshift used for halo terms
        z = self.redshift

        j_nu_EP_step = np.zeros([len(self.effective_freq), len(self.redshift)])
        nu_temp = 0
        for nu in range(len(self.effective_freq)):
            z_old_EP    = emissivities_EP[nu,:,0]
            j_nu_old_EP = emissivities_EP[nu,:,1] 
            f_j_nu = interpolate.interp1d(
                z_old_EP,
                j_nu_old_EP,
                kind='cubic',
                bounds_error=False,
                fill_value=0,
            )
            j_nu_EP_step[nu_temp, :] = f_j_nu(z)
            nu_temp = nu_temp + 1
        print(j_nu_EP_step[0,:])

        j_nu_LP_step = np.zeros([len(self.effective_freq), len(self.redshift)])
        nu_temp = 0
        for nu in range(len(self.effective_freq)):
            z_old_LP    = emissivities_LP[nu,:,0]
            j_nu_old_LP = emissivities_LP[nu,:,1] 
            f_j_nu = interpolate.interp1d(
                z_old_LP,
                j_nu_old_LP,
                kind='cubic',
                bounds_error=False,
                fill_value=0,
            )
            j_nu_LP_step[nu_temp, :] = f_j_nu(z)
            nu_temp = nu_temp + 1

        self.j_nu_EP_step = j_nu_EP_step
        self.j_nu_LP_step = j_nu_LP_step

        return

    def dBdT_dust(self):

        nu0_dust  = self.fixed_param['nu0_dust']
        freq_dust = []
        
        for i in self.effective_freq:
            if 'cib' in i:
                freq_dust.append(i['cib'])
        freq_dust = np.array(freq_dust)

        deno     = (h_pl / (k_b * self.fixed_param['T_CMB'])) ** -1 * 1e-9  # GHz
        x        = freq_dust / deno
        x0       = nu0_dust / deno
        dBdT0    = x0 ** 4 * np.exp(x0) / (np.exp(x0) - 1) ** 2
        pref     = 2 * k_b ** 3 * self.fixed_param['T_CMB'] ** 2 / (c ** 2 * h_pl ** 2) * 1e20
        dBdT_num = pref * x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
        dBdT     = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2 / dBdT0

        self.dBdT_dust = dBdT
        self.dBdT_num_dust = dBdT_num

        return