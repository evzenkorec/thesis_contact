"""This program solves contact problem of linear elastic semi-circle
 which is in contact with rigid plane using SNES solver for variational 
 inequalities.
 Semi-circle initially "lies" on the rigid surface such that "minimum distance" 
 between semi-circle and  rigid plane is zero. Contact is enforced by Dirichlet 
 type boundary  condition on the "line" part of the semi-circle, which moves 
 perpendicularly towards the rigid surface a distance of "penetration"."""

from dolfin import *
from create_mesh import *
from petsc4py import PETSc
import numpy as np


# Generate mesh 
R = 50
mesh = generate_half_circle(50, res=1)

# Define function space and relevant functions
V = VectorFunctionSpace(mesh, ("Lagrange", 1))
u, v  = Function(V), TestFunction(V)             # Trial and test function

d = u.geometric_dimension()            # Space dimension of u (2 in our case)
B  = Constant(mesh, [0.0, 0.0])        # Body force per unit volume
T0 =  Constant(mesh, [0.0, 0.0])       # Traction force on the "line" part of the semi-circle
T1 =  Constant(mesh, [0.0, 0.0])       # Traction force on the rest of the semi-circle

# Displacement of top of semi-circle towards the rigid surface [mm]
penetration = 2.00    

# FEniCS tolerance
tol = 1e-14 

# Definition for marking boundaries
boundary_markers = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0) 

# Definition of Dirichlet type boundary - the line part of the semi-circle
def boundary_D(x, only_boundary):
    return x[:, 1] > R - tol

# Boundary with traction force T1 - see the description of T1
def bn1(x):
    return np.logical_and(x[:,1] < R-tol,x[:,1] > penetration + tol)
boundary_markers.mark(bn1, 1)

# Contact search - contact part of the boundary
def bC1(x):
    return x[:,1] < penetration - tol
boundary_markers.mark(bC1, 2)


from ufl import derivative, dot, dx, grad, Identity, inner, Measure, sym, tr
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
dirichlet_value = project(Constant(mesh, (0.0,-penetration)),V)
bc = DirichletBC(V, dirichlet_value , boundary_D)


# Elasticity parameters
E = Constant(mesh,200000.)   # Young's modulus [MPa]
nu = Constant(mesh, 0.3)     # Poisson ratio [-]
mu = E/(2*(1+nu))              # Elastic parameter mu
lmbda = E*nu/((1+nu)*(1-2*nu)) # Elastic parameter lambda

def epsilon(u): # Definition of deformation tensor
    return sym(grad(u))#0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u): # Definition of Cauchy stress tensor
    return lmbda*tr(epsilon(u))*Identity(d) + 2.0*mu*epsilon(u)
def maculay(x): # Definition of Maculay bracket
    return (x+abs(x))/2



# Total potential energy
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
psi = inner(sigma(u), epsilon(u)) # Stored strain energy density (linear elasticity model)
Pi = psi*dx - dot(B, u)*dx - dot(T0, u)*ds(0) - dot(T1, u)*ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)


# The displacement u must be such that the current configuration x+u
# remains in the box [xmin = -inf,xmax = inf] x [ymin = 0,ymax = inf]
class bounds:
    def __init__(self,x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim

    def eval(self, values, x):
        values[:, 0] = self.x_lim - x[:,0]
        values[:, 1] = self.y_lim - x[:,1]

constraint_u = bounds(np.infty, np.infty)
umax = interpolate(constraint_u.eval, V)
constraint_l = bounds(-np.infty, 0)
umin = interpolate(constraint_l.eval, V)


class NonlinearPDE_SNESProblem():
    def __init__(self, F, u, bc):
        super().__init__()
        V = u.function_space
        du = function.TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.a_comp = dolfin.fem.Form(self.a)
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
        fem.assemble_vector(F, self.L)
        fem.apply_lifting(F, [self.a], [[self.bc]], [x], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        fem.assemble_matrix(J, self.a, [self.bc])
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
snes.setTolerances(rtol=1.0e-9, max_it=10)
snes.setFromOptions()
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.solve(None, u.vector)
assert snes.getConvergedReason() > 0

# Post-processing
from dolfin.io import XDMFFile
with XDMFFile(mesh.mpi_comm(), "output/u.xdmf") as file:
    file.write(u)

#von Mises stresses
from numpy import pi
from ufl import sqrt as uflsqrt
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
von_Mises = uflsqrt(3./2*inner(s, s))
V0 = FunctionSpace(mesh, ("DG", 0)) # Define function space for post-processing
von_Mises = project(von_Mises, V0)
with XDMFFile(mesh.mpi_comm(), "output/von_mises.xdmf") as file:
    file.write(von_Mises)

from dolfin.fem import assemble_scalar
# Comparison of Maximum pressure [kPa] and applied force [kN] of FEM solution with analytical Herz solution
p = Function(V0) # Contact pressure - for post-processing

p = project(-sigma(u)[1, 1], V0)
from numpy import sqrt
a = sqrt(R*penetration)
F = pi/4*E.value*d

p0 = E.value*d/(2*a)

print("Maximum pressure FE: {0:8.5f} kPa Hertz: {1:8.5f} kPa".format(1e-3*max(np.abs(p.vector.array)), 1e-3*p0))
print("Applied force    FE: {0:8.5f} kN Hertz: {1:8.5f} kN".format(1e-3*assemble_scalar(p*ds(2)), 1e-3*F))



