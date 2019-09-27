"""This program solves contact problem of linear elastic semi-circle
 which is in contact with rigid plane using SNES solver for variational
 inequalities.
 Semi-circle initially "lies" on the rigid surface such that "minimum distance"
 between semi-circle and  rigid plane is zero. Contact is enforced by Dirichlet
 type boundary  condition on the "line" part of the semi-circle, which moves
 perpendicularly towards the rigid surface a distance of "penetration"."""

from dolfin import *
import numpy as np

# To generate mesh
from create_mesh import *
# UFL imports
from ufl import (derivative, dot, dx, grad, Identity, inner, Measure, sym, tr, sqrt)

from dolfin.geometry import BoundingBoxTree
from dolfin import geometry
# IO imports
from dolfin.io import XDMFFile
# Logging imports
from dolfin.log import set_log_level, LogLevel
# FEM imports
from dolfin.fem import (assemble_matrix, assemble_scalar, assemble_vector,
                        apply_lifting, Form, set_bc)
from dolfin.cpp.mesh import Ordering
from petsc4py import PETSc


# Generate mesh
R = 1
L = 2
# 3D mesh
#mesh = generate_cylinder(R,L,res=0.04)
# 2D mesh
mesh = generate_half_circle(R, res=0.01)

# Displacement of top of semi-circle towards the rigid surface [mm]
penetration = R/15

# Plane moved upwards to find collision points
plane = RectangleMesh(MPI.comm_world,
                      [np.array([-1.1,-0.5,0]),
                       np.array([1.2,penetration,0])],
                      [1,1], cpp.mesh.CellType.quadrilateral,
                      cpp.mesh.GhostMode.none)

# Read mesh function
with XDMFFile(MPI.comm_world, "mf.xdmf") as infile:
    mvc = infile.read_mvc_size_t(mesh, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc, 0)

# Define function space and relevant functions
degree = 1
# if degree > 1 :
#     Ordering.order_simplex(mesh)
V = VectorFunctionSpace(mesh, ("Lagrange", degree))
u, v  = Function(V), TestFunction(V)             # Trial and test function

d = u.geometric_dimension()            # Space dimension of u (2 in our case)
B  = Constant(mesh, [0.0,]*d)        # Body force per unit volume
T0 =  Constant(mesh, [0.0,]*d)       # Traction force on the "line" part of the semi-circle
T1 =  Constant(mesh, [0.0,]*d)       # Traction force on the rest of the semi-circle


# FEniCS tolerance
tol = 1e-14

def find_contact():
    """
    Return a list of 1/0 for the edges of the mesh.
       if edge is in contact with plane y=penetration and on boundary
         return 1
       else
         return 0
    """
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
    on_boundary = mesh.topology.on_boundary(1)

    tree_mesh = BoundingBoxTree(mesh, 2)
    tree_plane = BoundingBoxTree(plane, 2)
    ent_mesh, ent_plane = geometry.compute_collisions_bb(tree_mesh, tree_plane)
    coll_cells = set(ent_mesh)
    mesh.create_connectivity(2, 1)
    c21 = mesh.topology.connectivity(2,1)
    edges = []
    for cell in coll_cells:
        edges = np.append(edges, c21.connections(cell))
    edges = set(edges)

    bnd_edges = [i in edges and on_boundary[i] for i in range(len(on_boundary))]
    return np.array(bnd_edges).astype(np.uint64)

def project(value, V):
    """
    Simple implementation of project, needed due to issue 507
    https://github.com/FEniCS/dolfinx/issues/507
    and useful in post-processing.
    """
    u, v = TrialFunction(V), TestFunction(V)
    lhs = inner(u, v)*dx
    rhs = inner(value, v)*dx
    uh = Function(V)
    solve(lhs==rhs, uh)
    return uh

bnd_edges = find_contact()
mf.values[:] += bnd_edges

# Output plane to file (at its actual position)
plane.geometry.points[:,1] -= penetration
XDMFFile(MPI.comm_world, "snes/plane.xdmf").write(plane)

# Todo; Add filter on MeshFunction to return indices of facets with given value
dir_facets = []
for i in range(len(mf.values)):
    if np.isclose(mf.values[i], 1):
        dir_facets.append(i)

# Create Dirichlet-condition for penetration
dirichlet_value = project(Constant(mesh, [0.0,]*(d-1)+[-penetration,]),V)
bc = DirichletBC(V, dirichlet_value , dir_facets)
XDMFFile(MPI.comm_world, "snes/mf.xdmf").write(mf)

# Elasticity parameters
E = Constant(mesh,200000.)   # Young's modulus [MPa]
nu = Constant(mesh, 0.3)     # Poisson ratio [-]
mu = E/(2*(1+nu))              # Elastic parameter mu
lmbda = E*nu/((1+nu)*(1-2*nu)) # Elastic parameter lambda

def epsilon(u): # Definition of deformation tensor
    return sym(grad(u))#0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u): # Definition of Cauchy stress tensor
    return lmbda*tr(epsilon(u))*Identity(d) + 2.0*mu*epsilon(u)



# Total potential energy
ds = Measure('ds', domain=mesh, subdomain_data=mf)
psi = inner(sigma(u), epsilon(u)) # Stored strain energy density (linear elasticity model)
Pi = psi*dx - dot(B, u)*dx - dot(T0, u)*ds(0) - dot(T1, u)*ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)


# The displacement u must be such that the current configuration x+u
# remains in the box [xmin = -inf,xmax = inf] x [ymin = 0,ymax = inf]
class Bounds:
    def __init__(self,x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim

    def eval(self, values, x):
        values[:, 0] = self.x_lim - x[:, 0]
        values[:, 1] = self.y_lim - x[:, 1]
        if d ==3:
            values[:, 1] = self.x_lim-x[:,1]
            values[:, 2] = self.y_lim-x[:,2]

constraint_l = Bounds(-np.infty, 0)
constraint_u = Bounds(np.infty, np.infty)
umax = interpolate(constraint_u.eval, V)
umin = interpolate(constraint_l.eval, V)


class NonlinearPDE_SNESProblem():
    def __init__(self, F, u, bc):
        """
        Nonlinear solver using SNES, lifted from unit tests
        """
        super().__init__()
        V = u.function_space
        du = TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.a_comp = Form(self.a)
        self.bc = bc
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], [[self.bc]], [x], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.a, [self.bc])
        J.assemble()


# Set up the non-linear solver
problem = NonlinearPDE_SNESProblem(F, u, bc)

#problem.set_bounds(umin, umax)
b = dolfin.cpp.la.create_vector(V.dofmap.index_map)
J = dolfin.cpp.fem.create_matrix(problem.a_comp._cpp_object)

# Create Newton solver and solve
snes = PETSc.SNES().create()
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)
snes.setVariableBounds(umin.vector, umax.vector)
snes.setType("vinewtonrsls")
snes.setTolerances(rtol=1.0e-7, max_it=25)

snes.setFromOptions()
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.solve(None, u.vector)
print(snes.getConvergedReason())
assert snes.getConvergedReason() > 0

# Post-processing
with XDMFFile(mesh.mpi_comm(), "snes/u.xdmf") as file:
    file.write(u)

#von Mises stresses
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V0 = FunctionSpace(mesh, ("DG", degree-1)) # Define function space for post-processing
von_Mises = project(von_Mises, V0)
with XDMFFile(mesh.mpi_comm(), "snes/von_mises.xdmf") as file:
    file.write(von_Mises)

# Comparison of Maximum pressure [kPa] and applied force [kN] of FEM solution with analytical Herz solution
p = project(-sigma(u)[d-1, d-1], V0)
a = np.sqrt(R*penetration)
Es = E.value/(1-nu.value**2)
F_p = assemble_scalar(p*ds(2))
if d == 2:
    F_p *= L

with XDMFFile(mesh.mpi_comm(), "snes/von_mises.xdmf") as file:
    file.write(von_Mises)
with XDMFFile(mesh.mpi_comm(), "snes/p.xdmf") as file:
    file.write(p)


F = np.pi/4*Es*penetration*L
p0 = Es*penetration/(2*a)


print("Maximum pressure FE: {0:8.3e} kPa Hertz: {1:8.3e} kPa".format(1e-3*max(np.abs(p.vector.array)), 1e-3*p0))
print("Applied force    FE: {0:8.3e} kN Hertz: {1:8.3e} kN".format(1e-3*F_p, 1e-3*F))
