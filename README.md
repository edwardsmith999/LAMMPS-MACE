# LAMMPS-MACE
A script to run LAMMPS through the Python interface with forces set using the MACE-MP-0 ML potential

This runs an example of ZnO2 wall with H2O water above it and overwrites the forces based on the MACE ML potential.

The required software is as follows

- numpy
- matplotlib
- Python binding for [lammps](https://docs.lammps.org/Python_head.html)
- [ase](https://databases.fysik.dtu.dk/ase/install.html)
- [MACE](https://mace-docs.readthedocs.io/en/latest/guide/installation.html) and the MACE MP 0 model downloaded [here](https://github.com/ACEsuit/mace-mp/releases).

This then runs and plots the difference between the LAMMPS and MACE prediction.

![Figure_1](https://github.com/user-attachments/assets/05cc1d18-b266-463b-b070-60f1cecf1186)

Showing the difference in force between the QM MACE prediction using release https://github.com/ACEsuit/mace-mp/releases/tag/mace_mp_0b2
