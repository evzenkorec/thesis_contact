from penalty_solver_3D import penalty_solver
from snes_solver_3D import snes_solver
from nitsche_solver_3D import nitsche_solver
from nitscheRFP_solver_3D import nitscheRFP_solver
from auglag_solver_3D import auglag_solver
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Text in command window - text output = True, no text = False
set_log_active(True)

# Geometry parametres
R = 10 # Radius of the half-sphere in [mm]
np.save("parameters/R.npy",R)

# Elasticity parameters
E = 200000. # Young's modulus [MPa]
np.save("parameters/E.npy",E)
nu = 0.25 # Poisson ratio [-]
np.save("parameters/nu.npy",nu)

# penalty parameter
penalty = 100
np.save("parameters/penalty.npy",penalty)
penalty_nitsche = 50
np.save("parameters/penalty_nitsche.npy",penalty_nitsche)
penalty_nitsche_RFP = 50
np.save("parameters/penalty_nitsche_RFP.npy",penalty_nitsche_RFP)
penalty_auglag = 50
np.save("parameters/penalty_auglag.npy",penalty_auglag)
# Parameter of nitscheRFP method
# theta = 1 - symmetric method
theta = 1
np.save("parameters/theta.npy",theta)

penetration = R/10 # penetration in [mm]
np.save("parameters/penetration.npy",penetration)

for i in range(1): 
    # Load mesh
    mesh = Mesh()
    with XDMFFile("mesh/mesh_" + str(i) + ".xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)
    
    
    with XDMFFile("mesh/mf_" + str(i) + ".xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)    
      
    # Solvers list
    
    u = penalty_solver(R,E,nu,penetration,penalty, mesh, boundary_markers)
    with XDMFFile("u_PEN_" + str(i) + ".xdmf") as xdmf:
        xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    u = snes_solver(R,E,nu,penetration,penalty, mesh, boundary_markers)
    with XDMFFile("u_SNES_" + str(i) + ".xdmf") as xdmf:
        xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    u = nitsche_solver(R,E,nu,penetration, penalty_nitsche, mesh, boundary_markers)
    with XDMFFile("u_NITSCHE_" + str(i) + ".xdmf") as xdmf:
        xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    u = nitscheRFP_solver(R,E,nu,penetration, penalty_nitsche_RFP, theta, mesh, boundary_markers)
    with XDMFFile("u_NITSCHE_RFP_" + str(i) + ".xdmf") as xdmf:
        xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False) 

    u, lm = auglag_solver(R,E,nu,penetration, penalty_auglag, mesh, boundary_markers)
    with XDMFFile("u_AUGLAG_" + str(i) + ".xdmf") as xdmf:
        xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)
    with XDMFFile("lm_AUGLAG_" + str(i) + ".xdmf") as xdmf:
        xdmf.write_checkpoint(lm, "lm", 0.0, XDMFFile.Encoding.HDF5, append=False)