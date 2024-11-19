# LAMMPS-MACE
A script to run LAMMPS through the Python interface with forces set using the MACE-MP-0 ML potential

This runs an example of ZnO2 wall with H2O water above it and overwrites the forces based on the MACE ML potential.

The required software is as follows

     numpy
     scipy    
     Python binding for [lammps](https://docs.lammps.org/Python_head.html)
     [ase](https://databases.fysik.dtu.dk/ase/install.html)
     [MACE](https://mace-docs.readthedocs.io/en/latest/guide/installation.html)

This then runs and plots the difference between the LAMMPS and MACE prediction.
