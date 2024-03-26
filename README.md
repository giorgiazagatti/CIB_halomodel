# Repository overview

This repository contains the code implemented and used in:

*TITLE PAPER and AUTHORS*

## Installation

Instructions on how to install and set up the project.

```bash
git clone https://github.com/giorgiazagatti/CIB_halomodel.git
cd CIB_halomodel
```

## How to use



## Parameters description

In this repository, there are three parameter files that mimic the datasets used in the study. The structure of the parameter files is the same and it is composed of three main blocks:

- *Options*: to set up the general settings
- *Frequencies*: instrumental features of the specific experiment
- *Parameters*: cosmological parameters, model parameters, power spectra parameters

In the following we provide for a description of each block of the parameter file.

### Options

In this first block of the parameter file, the user can modify how to deal with the matter power spectrum, the different normalization of the power spectra, the multipole range in which the user wants to compute the CIB power spectra and the redshift.

- *Linear matter power spectrum*. The user can choose between three different options:
  - Use the tabulated matter power spectrum used in this study. This has been computed with CAMB, using the *Planck18* cosmology. To choose this option, set:
    ```bash
    read_matterPS: True
    ```
  - Compute the matter power spectrum for a different cosmology. To compute it, the user has to change the previous flag:
    ```bash
    read_matterPS: False
    ```
  and the linear matter power spectrum is computed by the code with CAMB.
  - Use a linear matter power spectrum provided by the user. In this case, the user has to set the flag at *True*, and insert the path to his matter power spectrum in the third block of the parameter file, and specifically, in the label 'matter_PS':
    ```bash
    parameters:
      cosmology:
        T_CMB: 2.725
        tau: 0.0544
        ns: 0.9649
        As: 2.101e-9
        pivot_scalar: 0.05
        matter_PS: 
    ```
- *Power spectra normalization*. Flag to choose between CIB power spectra computed in $C_\ell$'s or in $D_\ell$'s. In particular, if
    ```bash
    normalization: 0 
    ```
  the resulting power spectra are computed in $C_\ell$'s, otherwise, if
    ```bash
    normalization: 1 
    ```
  the CIB power spectra are computed in $D_\ell$'s.
- *Multipole range*: to set the minimum and the maximum multipoles to compute the CIB power spectra.
    ```bash
    ell_range:          #set the minimum and the maximum ell for the CIB power spectra computation
      ell_min: 1
      ell_max: 2000 
    ```
  Default values are set for the different datasets.
- *Redshift*: path to the redshifts for the analysis.

### Frequencies

The second block contains the information related to each experiment. Specifically:
- the frequency channels of the experiment/dataset;
- the units of the CIB power spectra of the dataset. The default option is $Jy^2$ for all the parameter files, however it is also possible to compute the power spectra in $\mu K^2$ changing:
  ```bash
    units: 'Jy^2'
  ```
  in:
  ```bash
    units: 'muK^2'
  ```
- the effective frequency for the CIB emission;
- the color correction for each frequency channel.

### Paramters
