"""This program solves contact problem of linear elastic semi-circle
 which is in contact with rigid plane using the Augmented Lagrange method. 
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
from mshr import *
import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt
import numpy as np

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
# tell the form to apply optimization strategies in the code generation
# phase and the use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory).

# Define mesh geometry using "mshr"
R = 50 # Radius of the semi-circle in [mm] - can be changed
rect_help = Rectangle(Point(-R, 2*R), Point(R, R))
circle = Circle(Point(0.0, R), R)
half_circle = circle - rect_help

# Create mesh
mesh_half_circle = generate_mesh(half_circle, 40)
mesh = mesh_half_circle

# Define mixed function space 
LG = VectorFunctionSpace(mesh,"Lagrange",  1) # Define function space for displacement - piece-wise linear functions
RE =FunctionSpace(mesh,"R", 0) # Define function space for lagrange multipliers - piece-wise linear functions
LG2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
RE2 = FiniteElement("R", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, LG2*RE2) # Define mixed function space
V0 = FunctionSpace(mesh, "DG", 0) # Define function space for post-processing of stress

# Define trial and test functions 
v = TestFunction(LG)
varLm = TestFunction(RE)
w = Function(V)
u, Lm = split(w)
v, varLm = TestFunctions(V)
varw = TestFunction(V)
dw = TrialFunction(V)
p = Function(V0, name="Contact pressure") # Contact pressure for post-processing 
d = u.geometric_dimension() # Space dimension of u (2 in our case)

# Body force per unit volume and tractions
B  = Constant((0.0, 0.0))        # Body force per unit volume
T0 =  Constant((0.0, 0.0))       # Traction force on the "line" part of the semi-circle - should be set zero because of Dirichlet boundary condition on the same part of the boundary  
T1 =  Constant((0.0, 0.0))       # Traction force on the rest of the semi-circle (except "line" part and contact part) 

penetration = 2.00 # semi-circle moves perpendicularly towards the rigid surface a distance of "penetration" in [mm]      
penalty = Constant(1e9) # penalty parameter - !!!is divided by 2 in the total potential energy functional!!!                 
sphere = Expression("(x[0]*x[0])/(2*R)", R=R, degree=2)  # Quadratic approximation of sphere for easier implementation  

tol = DOLFIN_EPS # FEniCS tolerance - necessary for "boundary_markers" and the correct definition of the boundaries 

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # Definition of "boundary_markers" 

boundary_D = CompiledSubDomain('on_boundary && near(x[1], R, tol)', R = R, tol=tol) # Definition of Dirichlet type boundary - the line part of the semi-circle
bc = DirichletBC(V.sub(0), Constant((0.0,-penetration)), boundary_D) # Setting the Dirichlet type boundary condition

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
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
#return sym(nabla_grad(u))
def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(d) + 2.0*mu*epsilon(u)
def maculay(x): # Definition of Maculay bracket
    return (x+abs(x))/2
def gap(): # Definition of gap function
    return sphere+u[1]

# Stored strain energy density (linear elasticity model)
psi = inner(sigma(u), epsilon(u))

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T0, u)*ds(0) - dot(T1, u)*ds(1) -1./(2.*penalty)*(dot(Lm,Lm)-dot(maculay(-Lm-penalty*(gap())),maculay(-Lm-penalty*(gap()))))*ds(2)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, w, varw)

# Compute Jacobian of F for iterative "newton_solver"
J = derivative(F, w, dw)

# Set up the non-linear problem and parametres of iterative "newton_solver" 
solve(F == 0, w, bc, J = J, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

# Post-processing

#von Mises stresses
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
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
plt.savefig("displacementL-AUGLAG.pdf", format="pdf")

# Plot mesh
plt.figure()
graph2 = plot(mesh, title="MeshL")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig("meshL-AUGLAG.pdf", format="pdf")


# Plot von Mises stress
plt.figure()
plt.xlabel('$x$')
plt.ylabel('$y$')
graph3 = plot(von_Mises, title='Von Mises stress [kPa]')
plt.colorbar(graph3)
plt.savefig("stressL-AUGLAG.pdf", format="pdf")

#Plot magnitude of displacement
plt.figure()
plt.xlabel('$x$')
plt.ylabel('$y$')
graph4 = plot(u_magnitude, title='Displacement magnitude [mm]')
plt.colorbar(graph4)
plt.savefig("MAGdisplacementL-AUGLAG.pdf", format="pdf")

# Comparison of Maximum pressure [kPa] and applied force [kN] of FEM solution with analytical Herz solution 
p.assign(-project(sigma(u)[1, 1], V0))
# p.assign(-project(Lm, V0))
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