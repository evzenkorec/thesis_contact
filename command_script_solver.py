from penalty_solver_3D import penalty_solver
from snes_solver_3D import snes_solver
from nitsche_solver_3D import nitsche_solver
from nitscheRFP_solver_3D import nitscheRFP_solver
from auglag_solver_3D import auglag_solver
from dolfin import *
from multiphenics import *
import matplotlib.pyplot as plt
import numpy as np

# Text in command window - text output = True, no text = False
set_log_active(True)

# Geometry parametres
R = 10 # Radius of the half-sphere in [mm]
np.save("parameters/R.npy",R)

penetration = 0.1 # penetration in [mm]
np.save("parameters/penetration.npy",penetration)


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

# Number of generated meshes
num_meshes = np.load('parameters/num_meshes.npy')
num_meshes = num_meshes.tolist()

mesh_size = np.zeros(num_meshes)
iter_penalty = np.zeros(num_meshes)
iter_snes = np.zeros(num_meshes)
iter_nitsche = np.zeros(num_meshes)
iter_nitscheRFP = np.zeros(num_meshes)
iter_auglag = np.zeros(num_meshes)
time_penalty = np.zeros(num_meshes)
time_snes = np.zeros(num_meshes)
time_nitsche = np.zeros(num_meshes)
time_nitscheRFP = np.zeros(num_meshes)
time_auglag = np.zeros(num_meshes)

for i in range(num_meshes): 
    # Load mesh
    mesh = Mesh()
    with XDMFFile("mesh/mesh_" + str(i) + ".xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)

    # num_of_elem = len(mesh.cells())
    # num_of_dofs = len(V.dofmap().dofs())
    V = VectorFunctionSpace(mesh, "Lagrange", 1) # Define function space of the problem - piece-wise linear functions
        
    with XDMFFile("mesh/mf_" + str(i) + ".xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    
    # semi-circle moves perpendicularly towards the rigid surface a distance of "penetration" in [mm]
    # penetration = R/50
      
    # Solvers list
    
    # penalty_solver(R,E,nu,penetration, penalty, mesh, boundary_markers)
    num_iter, solution_time, num_of_dofs = penalty_solver(R,E,nu,penetration,penalty, mesh, boundary_markers)
    iter_penalty[i] = num_iter
    time_penalty[i] = solution_time
    mesh_size[i] = num_of_dofs
    # with XDMFFile("u_PEN_" + str(i) + ".xdmf") as xdmf:
    #     xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    # snes_solver(R,E,nu,penetration, penalty)
    num_iter, solution_time, num_of_dofs = snes_solver(R,E,nu,penetration,penalty, mesh, boundary_markers)
    iter_snes[i] = num_iter
    time_snes[i] = solution_time
    # with XDMFFile("u_SNES_" + str(i) + ".xdmf") as xdmf:
    #     xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    num_iter, solution_time, num_of_dofs = nitsche_solver(R,E,nu,penetration, penalty_nitsche, mesh, boundary_markers)
    iter_nitsche[i] = num_iter
    time_nitsche[i] = solution_time
    # with XDMFFile("u_NITSCHE_" + str(i) + ".xdmf") as xdmf:
    #     xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    num_iter, solution_time, num_of_dofs = nitscheRFP_solver(R,E,nu,penetration, penalty_nitsche_RFP, theta, mesh, boundary_markers)
    iter_nitscheRFP[i] = num_iter
    time_nitscheRFP[i] = solution_time
    # with XDMFFile("u_NITSCHE_RFP_" + str(i) + ".xdmf") as xdmf:
    #     xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)

    boundary_restriction = MeshRestriction(mesh, "mesh/contact_restriction_boundary_" + str(i) + ".rtc.xml")
    solution_time, num_of_dofs = auglag_solver(R,E,nu,penetration, penalty_auglag, mesh, boundary_markers, boundary_restriction)
    time_auglag[i] = solution_time
    # with XDMFFile("u_NITSCHE_RFP_" + str(i) + ".xdmf") as xdmf:
    #     xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False) 
        

np.save("parameters/mesh_size.npy", mesh_size)
np.save("parameters/time_penalty.npy", time_penalty)
np.save("parameters/time_snes.npy", time_snes)
np.save("parameters/time_nitsche.npy", time_nitsche)
np.save("parameters/time_nitscheRFP.npy", time_nitscheRFP)
np.save("parameters/time_auglag.npy", time_auglag)

# Save colored mesh partitions in VTK format if running in parallel
# for the last mesh
if MPI.size(mesh.mpi_comm()) > 1:
    File("mesh/partitions.pvd") << MeshFunction("size_t", mesh, mesh.topology().dim(), \
                                           MPI.rank(mesh.mpi_comm()))