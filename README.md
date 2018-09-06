# SCAMPy #

SCAMPy (Single Column Atmospheric Model in Python) provides a framework for testing parameterizations of clouds and turbulence.
It is particularly designed to support eddy-diffusivity mass-flux modeling frameworks.

Information about the EDMF parameterization implemented in SCAMPy can be found in:

Tan, Z., C. M. Kaul, K. G. Pressel, Y. Cohen, T. Schneider, and J. Teixeira, 2018:
An extended eddy-diffusivity mass-flux scheme for unified representation of
subgrid-scale turbulence and convection. Journal of Advances in Modeling Earth Systems, 2018.
(see [Tan et al., 2018](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2017MS001162)).

The code is written in Python and Cython.

Code Contributors:
	Colleen Kaul (Caltech)--initial/primary developer. Inquiries may be sent to cmkaul@caltech.edu;
	Yair Cohen (Caltech);
	Anna Jaruga (JPL/Caltech);
	Kyle Pressel (Caltech);
	Zhihong Tan (U. Chicago)

Additional Acknowledgements:
	Tapio Schneider (Caltech);
	Joao Teixeira (JPL)

# installation #

TODO - (Travis instructions in the future?)

# building and running #
```
$ cd scampy
```

Generate the simulation specific parameters (accepted keywords: Soares, DYCOMS_RF01, DYCOMS_RF02, Bomex, life_cycle_Tan2018, Rico, TRMM_LBA, ARM_SGP, GATE_III)
```
$ python generate_namelist.py Soares
```

Generate the turbulence parameters (accepted keywords: defaults, Soares, DYCOMS_RF01, DYCOMS_RF02, Bomex, life_cycle_Tan2018, Rico, TRMM_LBA, ARM_SGP, GATE_III)
```
$ python generate_paramlist.py Soares
```

Compile the source code by running setup.py
```
$ python setup.py build_ext --inplace
```

Serial execution of scampy (both turbulence and case specific parameters need to be passed)
```
$ python main.py Soares.in paramlist_Soares.in
```

# testing  #

To generate the automatic plots try:

```
$ py.test -v tests/
```
