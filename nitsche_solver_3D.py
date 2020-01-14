from dolfin import *
# from ufl import sign
import numpy as np
# from IPython import embed; embed()

def nitsche_solver(R,E,nu,penetration,penalty, mesh, boundary_markers):  
    
    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
       
    boundary_markers.array()[boundary_markers.array()>300] = 0
    
    # Characteristic size of the elements of the mesh 
    hMESH = CellDiameter(mesh)
    meshSIZE = 1/2*(mesh.hmax() + mesh.hmin())
    # h_avg = (hMESH('+') + hMESH('-')) / 2.0

    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 1) # Define function space of the problem - piece-wise linear functions
    
    # Define functions
    du = TrialFunction(V)            # Incremental displacement for Jacobi matrix of iterative "newton_solver"
    v  = TestFunction(V)             # Test function
    u  = Function(V)                 # Trial function
    # d = u.geometric_dimension()      # Space dimension of u (3 in our case)

    # Definition of normal vectors to the elastic body and to the rigid half-space
    nRIGID = Constant((0.0,0.0,1.0))
    n = FacetNormal(mesh)
    
    # Definition of tolerance
    tol = DOLFIN_EPS # FEniCS tolerance - necessary for "boundary_markers" and the correct definition of the boundaries 

    
    # Setting the Dirichlet type boundary condition
    bc1 = DirichletBC(V.sub(2), Constant((-penetration)), boundary_markers, 2) 
    bP = CompiledSubDomain('sqrt(x[0]*x[0]+x[1]*x[1]) < meshSIZE && near(x[2], R, tol) && on_boundary',meshSIZE = meshSIZE, R=R, tol = tol)
    bP.mark(boundary_markers, 4)    
    bc2 = DirichletBC(V.sub(0), Constant((0)), boundary_markers, 4)
    bc3 = DirichletBC(V.sub(1), Constant((0)), boundary_markers, 4)
    bc = [bc1,bc2,bc3]
    
    # Definition of contact zones
    bC1 = CompiledSubDomain('on_boundary && x[2] < penetration + tol', tol=tol, penetration = penetration) # Contact search - contact part of the boundary
    bC1.mark(boundary_markers, 3)

    # Definition of surface integral subdomains
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)


    # Elasticity parameters
    mu = E/2/(1+nu) # Elastic parameter mu
    lmbda = E*nu/(1+nu)/(1-2*nu) # Elastic parameter lambda

    # Spatial coordinates of mesh
    x = SpatialCoordinate(mesh)
    
    
    # Definition of functions for variational formulation of the contact problem 
    def epsilon(u): # Definition of deformation tensor
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    def sigma(u): # Definition of Cauchy stress tensor
        return lmbda*tr(epsilon(u))*Identity(3) + 2.0*mu*epsilon(u)
    def gap(u): # Definition of gap function
        x = SpatialCoordinate(mesh)
        return x[2]+u[2]
    def maculay(x): # Definition of Maculay bracket
        return (x+abs(x))/2
    def pN(u):
        return dot(dot(nRIGID,sigma(u)),nRIGID) 
    def convert_to_integer(a):
        "Convert to a 32-bit int. Raise exception if this will cause an overflow"
        if abs(a) <= np.iinfo(np.int32).max:
            return np.int32(a)
        else:
            raise OverflowError("Cannot safely convert to a 32-bit int.")

    # Stored strain energy density (linear elasticity model)
    psi = 1/2*inner(sigma(u), epsilon(u))

    # Total potential energy
    Pi = psi*dx - dot(pN(u),-maculay(-gap(u)))*ds(3) + 1/2*penalty*E/hMESH*dot(-maculay(-gap(u)),-maculay(-gap(u)))*ds(3)
    
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)
    # F = inner(sigma(u), epsilon(v))*dx - dot(pN(v),-maculay(gap(u)))*ds(3) - dot(pN(u),-v[2]*1/2*(1+sign(-gap(u))))*ds(3) + penalty*E/hMESH*dot(maculay(-gap(u)),-v[2]*1/2*(1+sign(-gap(u))))*ds(3) # - dot(pN(v),-maculay(-gap(u)))*ds(3) - dot(pN(u),v[2]*1/2*(1+sign(-gap(u))))*ds(3) + penalty*E/hMESH*dot(maculay(-gap(u)),-v[2]*1/2*(1+sign(-gap(u))))*ds(3)       
    
    # Compute Jacobian of F for iterative "newton_solver"
    J = derivative(F, u, du)

    # Solve nonlinear problem
    problem = NonlinearVariationalProblem(F, u, bc, J)
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-9
    prm["newton_solver"]["relative_tolerance"] = 1E-9
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["relaxation_parameter"] = 1.0
    prm["newton_solver"]["linear_solver"] = "lu"
    # prm["newton_solver"]["preconditioner"] = "petsc_amg"
    # prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1E-9
    # prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1E-7
    # prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 1000

    solver.parameters.update(prm)

    solver.solve()    
    
    return u


