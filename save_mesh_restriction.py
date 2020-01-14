from dolfin import *
from multiphenics import *
import numpy as np

tol = DOLFIN_EPS

penetration = np.load('parameters/penetration.npy')
penetration = penetration.tolist()

for i in range(1): 
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

    File("mesh/contact_restriction_boundary.rtc.xml") << boundary_restriction
    # XDMFFile("mesh/contact_restriction_boundary.rtc.xdmf").write(boundary_restriction)