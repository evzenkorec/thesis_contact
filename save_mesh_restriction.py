from dolfin import *
from multiphenics import *
import numpy as np
import os
tol = DOLFIN_EPS

penetration = 0.1 # penetration in [mm]

# penetration = np.load('parameters/penetration.npy')
# penetration = penetration.tolist()

# Number of generated meshes
num_meshes = np.load('parameters/num_meshes.npy')
num_meshes = num_meshes.tolist()

for i in range(num_meshes): 
    # Load mesh
    mesh = Mesh()
    with XDMFFile("mesh/mesh_" + str(i) + ".xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)
    
    
    with XDMFFile("mesh/mf_" + str(i) + ".xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    boundary_markers.array()[boundary_markers.array()>300] = 0

    class OnContactBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[2] < penetration + tol

    on_contact_boundary = OnContactBoundary()
    boundary_restriction = MeshRestriction(mesh, on_contact_boundary)
    os.system('rm -rf mesh/contact_restriction_boundary_{0:d}.rtc.xml'.format(i))
    os.system('rm -rf mesh/contact_restriction_boundary_{0:d}.rtc.xdmf'.format(i))

    File("mesh/contact_restriction_boundary_" + str(i) + ".rtc.xml") << boundary_restriction
    #XDMFFile("mesh/contact_restriction_boundary_" + str(i) + ".rtc.xdmf").write(boundary_restriction)