from contact_postprocessing_full import postprocessing
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Text in command window - text output = True, no text = False
set_log_active(False)

R = np.load('parameters/R.npy')
R = R.tolist()
penetration = np.load('parameters/penetration.npy')
penetration = penetration.tolist()
E = np.load('parameters/E.npy')
E = E.tolist()
nu = np.load('parameters/nu.npy')
nu = nu.tolist()
penalty = np.load('parameters/penalty.npy')
penalty = penalty.tolist()
penalty_nitsche = np.load('parameters/penalty_nitsche.npy')
penalty_nitsche = penalty_nitsche.tolist()
penalty_nitsche_RFP = np.load('parameters/penalty_nitsche_RFP.npy')
penalty_nitsche_RFP = penalty_nitsche_RFP.tolist()
penalty_auglag = np.load('parameters/penalty_auglag.npy')
penalty_auglag = penalty_auglag.tolist()
theta = np.load('parameters/theta.npy')
theta = theta.tolist()

for i in range(1): 
    # Load mesh
    mesh = Mesh()
    with XDMFFile("mesh/mesh_" + str(i) + ".xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)

    with XDMFFile("mesh/mf_" + str(i) + ".xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    V = VectorFunctionSpace(mesh, "Lagrange", 1) # the same space as the solution space
    u = Function(V)


    # Loading and postprocessing calculated displacements 

    with XDMFFile("u_PEN_" + str(i) + ".xdmf") as infile:
        infile.read_checkpoint(u, "u", 0)            
    postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)
    
    with XDMFFile("u_SNES_" + str(i) + ".xdmf") as infile:
        infile.read_checkpoint(u, "u", 0)            
    postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)

    with XDMFFile("u_NITSCHE_" + str(i) + ".xdmf") as infile:
        infile.read_checkpoint(u, "u", 0)            
    postprocessing(R,E,nu, penetration, penalty_nitsche, u, mesh, boundary_markers)

    with XDMFFile("u_NITSCHE_RFP_" + str(i) + ".xdmf") as infile:
        infile.read_checkpoint(u, "u", 0)            
    postprocessing(R,E,nu, penetration, penalty_nitsche_RFP, u, mesh, boundary_markers)
    
    with XDMFFile("u_AUGLAG_" + str(i) + ".xdmf") as infile:
        infile.read_checkpoint(u, "u", 0)            
    postprocessing(R,E,nu, penetration, penalty_auglag, u, mesh, boundary_markers)