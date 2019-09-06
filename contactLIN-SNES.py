"""This program solves contact problem of linear elastic semi-circle
 which is in contact with rigid plane using SNES solver for variational 
 inequalities.
 Semi-circle initially "lies" on the rigid surface such that "minimum distance" 
 between semi-circle and  rigid plane is zero. Contact is enforced by Dirichlet 
 type boundary  condition on the "line" part of the semi-circle, which moves 
 perpendicularly towards the rigid surface a distance of "penetration"."""

# Copyright (C) 2019 ...
#
# This file is part of DOLFIN.
# 
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License 
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by ... 2019

from dolfin import *
from create_mesh import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
R = 50
mesh = generate_half_circle(50, res=1)
from dolfin.io import XDMFFile
xdmf_file = XDMFFile(mesh.mpi_comm(), "mesh.xdmf")
xdmf_file.write(mesh)
xdmf_file.close()
exit(1)

# Function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 1) # Define function space of the problem - piece-wise linear functions

# Define functions
du = TrialFunction(V)            # Incremental displacement for Jacobi matrix of iterative "newton_solver"
u, v  = Function(V), TestFunction(V)             # Trial and test function

d = u.geometric_dimension()      # Space dimension of u (2 in our case)
B  = Constant((0.0, 0.0))        # Body force per unit volume
T0 =  Constant((0.0, 0.0))       # Traction force on the "line" part of the semi-circle - should be set zero because of Dirichlet boundary condition on the same part of the boundary  
T1 =  Constant((0.0, 0.0))       # Traction force on the rest of the semi-circle (except "line" part and contact part) 

penetration = 2.00 # semi-circle moves perpendicularly towards the rigid surface a distance of "penetration" in [mm]      

tol = DOLFIN_EPS # FEniCS tolerance - necessary for "boundary_markers" and the correct definition of the boundaries 

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # Definition of "boundary_markers" 

boundary_D = CompiledSubDomain('on_boundary && near(x[1], R, tol)', R = R, tol=tol) # Definition of Dirichlet type boundary - the line part of the semi-circle
bc = DirichletBC(V, Constant((0.0,-penetration)), boundary_D) # Setting the Dirichlet type boundary condition

bn0 = CompiledSubDomain('on_boundary && near(x[1], R, tol)', R = R, tol=tol) # Boundary with traction force T0 - see the description of T0
bn0.mark(boundary_markers, 0) 

bn1 = CompiledSubDomain('on_boundary && x[1] < R - tol && x[1] > penetration + tol', R = R, penetration = penetration, tol=tol) # Boundary with traction force T1 - see the description of T1
bn1.mark(boundary_markers, 1)

bC1 = CompiledSubDomain('on_boundary && x[1] < penetration - tol', tol=tol, penetration = penetration) # Contact search - contact part of the boundary
bC1.mark(boundary_markers, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Elasticity parameters
E = Constant(200000.) # Young's modulus [MPa]
nu = Constant(0.3) # Poisson ratio [-]
mu = E/2/(1+nu) # Elastic parameter mu
lmbda = E*nu/(1+nu)/(1-2*nu) # Elastic parameter lambda

def epsilon(u): # Definition of deformation tensor
    return sym(grad(u))#0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u): # Definition of Cauchy stress tensor
    return lmbda*tr(epsilon(u))*Identity(d) + 2.0*mu*epsilon(u)
def maculay(x): # Definition of Maculay bracket
    return (x+abs(x))/2

# Stored strain energy density (linear elasticity model)
psi = inner(sigma(u), epsilon(u))

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T0, u)*ds(0) - dot(T1, u)*ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F for iterative "newton_solver"
J = derivative(F, u, du)

# The displacement u must be such that the current configuration x+u
# remains in the box [xmin = -inf,xmax = inf] x [ymin = 0,ymax = inf]
constraint_u = Expression(("xmax - x[0]","ymax - x[1]"),
                         xmax=np.infty,  ymax=np.infty, degree=1)
constraint_l = Expression(("xmin - x[0]","ymin - x[1]"),
                          xmin=-np.infty, ymin=0, degree=1)
umin = interpolate(constraint_l, V)
umax = interpolate(constraint_u, V)

# Define the solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                                          "maximum_iterations": 30,
                                          "report": True,
                                          "error_on_nonconvergence": False}}

# Set up the non-linear solver
problem = NonlinearVariationalProblem(F, u, bc, J=J)
problem.set_bounds(umin, umax)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)
info(solver.parameters, True)

# Solve the problem
(iter, converged) = solver.solve()

# Check for convergence
if not converged:
    warning("This demo is a complex nonlinear problem. Convergence is not guaranteed when modifying some parameters or using PETSC 3.2.")

# Post-processing

# Save solution in VTK format
file = File("displacement.pvd")
file << u

#von Mises stresses
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))

V = FunctionSpace(mesh, 'Lagrange', 1)
von_Mises = project(von_Mises, V)

#displacement
# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)

# Plot the deformed configuration 
plt.figure()
graph1 = plot(u, mode="displacement", wireframe=True, title="Displacement field - deformed configuration")
matplotlib.rcParams['interactive'] == True
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(graph1)
plt.savefig("displacementL-SNES.pdf", format="pdf")

# Plot mesh
plt.figure()
graph2 = plot(mesh, title="MeshL")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig("meshL-SNES.pdf", format="pdf")


# Plot von Mises stress
plt.figure()
plt.xlabel('$x$')
plt.ylabel('$y$')
graph3 = plot(von_Mises, title='Von Mises stress [kPa]')
plt.colorbar(graph3)
plt.savefig("stressL-SNES.pdf", format="pdf")

#Plot magnitude of displacement
plt.figure()
plt.xlabel('$x$')
plt.ylabel('$y$')
graph4 = plot(u_magnitude, title='Displacement magnitude [mm]')
plt.colorbar(graph4)
plt.savefig("MAGdisplacementL-SNES.pdf", format="pdf")

# Comparison of Maximum pressure [kPa] and applied force [kN] of FEM solution with analytical Herz solution
V0 = FunctionSpace(mesh, "DG", 0) # Define function space for post-processing of stress
p = Function(V0, name="Contact pressure") # Contact pressure - for post-processing
p.assign(-project(sigma(u)[1, 1], V0))
a = sqrt(R*penetration)
F = pi/4*float(E)*d
p0 = float(E)*d/(2*a)
print("Maximum pressure FE: {0:8.5f} kPa Hertz: {1:8.5f} kPa".format(1e-3*max(np.abs(p.vector().get_local())), 1e-3*p0))
print("Applied force    FE: {0:8.5f} kN Hertz: {1:8.5f} kN".format(1e-3*assemble(p*ds(2)), 1e-3*F))

#print max and min for magnitude of displacement
print('min/max u [mm]:',
u_magnitude.vector().get_local().min(),
u_magnitude.vector().get_local().max())

#print max and min for von Mises stress
print('min/max von Mises stress [kPa]:',
1e-3*von_Mises.vector().get_local().min(),
1e-3*von_Mises.vector().get_local().max())



