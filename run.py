from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from lammps import lammps, PyLammps

import ase
from ase.data import atomic_masses, chemical_symbols
from ase.visualize import view
from ase.calculators.lammps import convert

from mace.calculators import MACECalculator

# Function to determine atom type from mass
def get_atom_element_from_mass(mass, MASS_TOLERANCE=0.1):
    atomic_number = 0
    for ref_mass in atomic_masses[1:]:
        atomic_number += 1
        #print(mass, ref_mass, atomic_names[atomic_number])
        if abs(mass - ref_mass) <= MASS_TOLERANCE:
            return chemical_symbols[atomic_number]
    return "Unknown"

use_nearwall = True
plot = False
convertunits = True

filename = "in.Zr_SPC"
#calculator = MACECalculator(model_paths='../../MACE_MPtrj_2022.9.model', device='cuda')
calculator = MACECalculator(model_paths='../../mace-large-density-agnesi-stress.model', device='cuda')

if plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    plt.ion()
    plt.show()

#Use PyLAMMPS higher level interface
L = PyLammps()
L.file(filename)
lmp = L.lmp

#Get simulation domain size
boxlo, boxhi, xy, xz, yz, pbc, _ = lmp.extract_box()
print("Simulation box = ", boxlo, boxhi, 
      " with periodic boundaries = ", pbc)
Lxyz=[boxhi[i]-boxlo[i] for i in range(3)]

# Get a mapping of atom types to masses
type_to_mass = {L.atoms[i].type: L.atoms[i].mass for i in range(L.atoms.natoms)}

# Deduce the element for each type
atom_type_to_element = {}
for atom_type, mass in type_to_mass.items():
    element = get_atom_element_from_mass(mass)
    atom_type_to_element[atom_type] = element
    print(f"Type {atom_type}: Mass = {mass}, Element = {element}")

print("Mapping of atom types to elements:", atom_type_to_element)

#Extract just near wall region
if use_nearwall:
    ztop_nearwall = 40
    zbot_nearwall = 25
    L_nearwall = ztop_nearwall - zbot_nearwall
    pbc[2] = 0
    Lxyz[2] = L_nearwall


#Get simulation units
if convertunits:
    for l in open(filename,'r'):
        if "units" in l:
            units = l.split()[1]
            break
    #Convert domain size
    for n in range(3):
        Lxyz[n] = convert(Lxyz[n], "distance", units, "ASE")

    if use_nearwall:
        ztop_nearwall = convert(ztop_nearwall, "distance", units, "ASE")
        zbot_nearwall = convert(zbot_nearwall, "distance", units, "ASE")

runstep = 1
Nsteps = 100

# Initialize a dictionary for atom types
atom_data = defaultdict(lambda: {"positions": [], "forces": [], 
                                 "indices": [], "ids": [], "count": 0})

for t in range(1, Nsteps):
    print(t)
    L.run(runstep, "pre yes post yes")

    # Clear the dictionary for new timestep data
    for key in atom_data.keys():
        atom_data[key]["positions"]=[]
        atom_data[key]["forces"]=[]
        atom_data[key]["indices"]=[]
        atom_data[key]["ids"]=[]
        atom_data[key]["count"] = 0

    # Loop over all atoms
    for i in range(L.atoms.natoms):
        atom = L.atoms[i]

        # Take only a subset of molecules (e.g., near surface)
        if use_nearwall:
            if not (zbot_nearwall <= atom.position[2] <= ztop_nearwall):
                continue

        # Determine element type of atom 
        ae = atom_type_to_element[atom.type]

        # Append position and force to the corresponding type
        atom_data[ae]["positions"].append(atom.position)
        atom_data[ae]["forces"].append(atom.force)
        atom_data[ae]["indices"].append(i)
        atom_data[ae]["ids"].append(atom.id)
        atom_data[ae]["count"] += 1

    # Convert positions and forces to numpy arrays
    for key, data in atom_data.items():
        data["positions"] = np.array(data["positions"])
        data["forces"] = np.array(data["forces"])

    # Optionally plot molecules
    if plot:
        plt.cla()
        for k in atom_data.keys():
            positions = atom_data[k]["positions"]
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
        plt.pause(0.01)

    # Prepare ASE atoms object
    chemical_formula = ""
    positions = np.empty((0, 3))
    tags = np.empty(0, dtype=int)
    for ae in atom_data:
        if atom_data[ae]['count'] < 1:
            continue
        chemical_formula = chemical_formula + ae + str(atom_data[ae]['count'])
        positions = np.concatenate([positions, atom_data[ae]["positions"]], axis=0)
        #Save position in atom array as tag so we can compare back to original (could do this
        #with atom id which would be more robust but needs an extra level of loopup id->index)
        tags = np.concatenate([tags, atom_data[ae]["indices"]])

    #Convert positions to ASE units
    if convertunits:
        for i in range(positions.shape[0]):
            for ixyz in range(positions.shape[1]):
                positions[i, ixyz] = convert(positions[i, ixyz], "distance", units, "ASE")

    #Create ASE atom object
    atoms = ase.atoms.Atoms(chemical_formula, positions, 
                            tags=tags, pbc=pbc, cell=Lxyz)

    #Set calculation method in ASE to use MACE MP 0
    atoms.set_calculator(calculator)

    #Get forces with QM accuracy
    forces_QM = atoms.get_forces()

    #Convert forces back from ASE units to LAMMPS
    if convertunits:
        for i in range(forces_QM.shape[0]):
            for ixyz in range(forces_QM.shape[1]):
                forces_QM[i, ixyz] = convert(forces_QM[i, ixyz], "force", "ASE", units)

    #Print the MD and QM values against each other
    forces_diff = []
    for a in atoms:
        forces_diff.append(forces_QM[a.index]-L.atoms[a.tag].force)
    forces_diff = np.array(forces_diff)

    #Plot error between QM and MD as colours
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    cm = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                    c=np.mean(forces_diff,1), cmap=plt.cm.RdYlBu_r)
    plt.colorbar(cm)
    plt.show()

    #Set the LAMMPS values based on QM force
    #How to ensure used in calculation, possibly not
    #if force is recalculated so pre no needed?
    for a in atoms:
        L.atoms[a.tag].force = forces_QM[a.index]

    #Run needs to do no pre calc so QM forces used
    L.run(runstep, "pre no post no")

    #Check updated forces
    #for a in atoms:
    #    print(a.index, a.position, L.atoms[a.tag].position)
    #    print(a.index, forces_QM[a.index], L.atoms[a.tag].force)

