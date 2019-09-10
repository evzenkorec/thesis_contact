"""This program solves contact problem of linear elastic semi-circle
 which is in contact with rigid plane using the penalty method. Semi-circle initially "lies" on
 the rigid surface such that "minimum distance" between semi-circle and
 rigid plane is zero. Contact is enforced by Dirichlet type boundary 
 condition on the "line" part of the semi-circle, which moves perpendicularly towards 
 the rigid surface a distance of "penetration"."""


from dolfin import *
import numpy as np
from ufl import inner, dx, grad, Measure, tr, nabla_grad, Identity, dot, derivative, SpatialCoordinate, sqrt
# To generate mesh
from create_mesh import *
from petsc4py import PETSc
from dolfin.fem import (assemble_vector, assemble_matrix,
                        assemble_scalar, set_bc, apply_lifting)
from dolfin.io import XDMFFile

R = 1
L = 2
mesh = generate_half_circle(R, res=0.01)
degree = 1
# Function spaces
V = VectorFunctionSpace(mesh, ("CG", degree)) # Define function space of the problem - piece-wise linear functions
u, v = Function(V), TestFunction(V)

d = u.geometric_dimension()    # Space dimension of u (2 in our case)

B  = Constant(mesh, [0,]*d)    # Body force per unit volume
# Traction force on the "line" part of the semi-circle - should be set zero because of Dirichlet boundary condition on the same part of the boundary  
T0 =  Constant(mesh, [0,]*d)
T1 =  Constant(mesh, [0,]*d)     # Traction force on the rest of the semi-circle (except "line" part and contact part) 

# Displacement of top of semi-circle towards the rigid surface [mm]
penetration = R/100
tol = 1e-14

boundary_markers = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0) 

# Definition of Dirichlet type boundary - the line part of the semi-circle
def boundary_D(x, only_boundary):
    return x[:, d-1] > R - tol
# Boundary with traction force T1 - see the description of T1
def bn1(x):
    return np.logical_and(x[:,d-1] < R-tol,x[:,d-1] > penetration + tol)

boundary_markers.mark(bn1, 1)

# Contact search - contact part of the boundary
def bC1(x):
        return x[:,d-1] < penetration - tol
boundary_markers.mark(bC1, 2)

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

# Create Dirichlet-condition for penetration
dirichlet_value = project(Constant(mesh, [0.0,]*(d-1)+[-penetration,]),V)
bc = DirichletBC(V, dirichlet_value , boundary_D)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Elasticity parameters
E = Constant(mesh, 200000.) # Young's modulus [MPa]
nu = Constant(mesh,0.3) # Poisson ratio [-]
mu = E/(2*(1+nu)) # Elastic parameter mu
lmbda = E*nu*((1+nu)*(1-2*nu)) # Elastic parameter lambda
penalty = E*Constant(mesh, 1e2) # penalty parameter  

def epsilon(u): # Definition of deformation tensor
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u): # Definition of Cauchy stress tensor
    return lmbda*tr(epsilon(u))*Identity(d) + 2.0*mu*epsilon(u)
def maculay(x): # Definition of Maculay bracket
    return (x+abs(x))/2

def gap(): # Definition of gap function
    x = SpatialCoordinate(mesh)
    return R-sqrt(R**2-x[0]**2)+u[1]

# Stored strain energy density (linear elasticity model)
psi = inner(sigma(u), epsilon(u))

# Total potential energy
Pi = psi*dx - inner(B, u)*dx  - inner(T0, u)*ds(0) - inner(T1, u)*ds(1)\
     + (penalty/2.0)*inner(maculay(-gap()),maculay(-gap()))*ds(2)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Setup for Newton Solver
class ElasticityEquation(NonlinearProblem):
    def __init__(self, L, u, bc):
        NonlinearProblem.__init__(self)
        V = u.function_space
        du = TrialFunction(V)
        self.L = L
        self.a = derivative(L, u, du)
        self.bc = bc
        self.u = u
        self._F, self._J = None, None

    def F(self, x):
        if self._F is None:
            self._F = assemble_vector(self.L)
        else:
            with self._F.localForm() as f_local:
                f_local.set(0.0)
            self._F = assemble_vector(self._F, self.L)
        apply_lifting(self._F, [self.a], [[self.bc]], [x], -1.0)
        self._F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._F, [self.bc], x, -1.0)
        return self._F

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    def J(self, x):
        if self._J is None:
            self._J = fem.assemble_matrix(self.a, [self.bc])
        else:
            self._J.zeroEntries()
            self._J = fem.assemble_matrix(self._J, self.a, [self.bc])
        self._J.assemble()
        return self._J


problem = ElasticityEquation(F, u, bc)
    
# Set up the non-linear problem and parametres of iterative "newton_solver"
solver = NewtonSolver(MPI.comm_world)

solver.rtol = 1e-6
solver.convergence_criterion = "incremental"
solver.max_it = 25
solver.solve(problem, u.vector)


# Post-processing
with XDMFFile(mesh.mpi_comm(), "penalty/u.xdmf") as file:
    file.write(u)

#von Mises stresses
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V0 = FunctionSpace(mesh, ("DG", degree-1)) # Define function space for post-processing
von_Mises = project(von_Mises, V0)

# Comparison of Maximum pressure [kPa] and applied force [kN] of FEM solution with analytical Herz solution
p = project(-sigma(u)[d-1, d-1], V0)
a = np.sqrt(R*penetration)
Es = E.value/(1-nu.value**2)
F_p = assemble_scalar(p*ds(2))
if d == 2:
    F_p *= L
    
with XDMFFile(mesh.mpi_comm(), "penalty/von_mises.xdmf") as file:
    file.write(von_Mises)
with XDMFFile(mesh.mpi_comm(), "penalty/p.xdmf") as file:
    file.write(p)


F = np.pi/4*Es*penetration*L
p0 = Es*penetration/(2*a)


print("Maximum pressure FE: {0:8.3e} kPa Hertz: {1:8.3e} kPa".format(1e-3*max(np.abs(p.vector.array)), 1e-3*p0))
print("Applied force    FE: {0:8.3e} kN Hertz: {1:8.3e} kN".format(1e-3*F_p, 1e-3*F))
