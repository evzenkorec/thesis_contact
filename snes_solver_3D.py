from dolfin import *
import time
from dolfin import Timer as t
import numpy as np
# from IPython import embed; embed()

def snes_solver(R,E,nu,penetration,penalty, mesh, boundary_markers):
     
    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
       
    boundary_markers.array()[boundary_markers.array()>300] = 0
    
    # Characteristic size of the elements of the mesh 
    # hMESH = CellDiameter(mesh)
    # meshSIZE = 1/2*(mesh.hmax() + mesh.hmin())
    meshSIZE = mesh.hmax()
    # h_avg = (hMESH('+') + hMESH('-')) / 2.0

    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 1) # Define function space of the problem - piece-wise linear functions
    num_of_dofs = V.dim()
    # num_of_dofs = len(V.dofmap().dofs())
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
    Pi = psi*dx 

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)
    
    # Compute Jacobian of F for iterative "newton_solver"
    J = derivative(F, u, du)

    # The displacement u must be such that the current configuration x+u
    # remains in the box [xmin = -inf,xmax = inf] x [ymin = 0,ymax = inf]
    constraint_u = Expression(("xmax - x[0]","ymax - x[1]","zmax - x[2]"),
                            xmax=np.infty,  ymax=np.infty, zmax=np.infty, degree=1)
    constraint_l = Expression(("xmin - x[0]","ymin - x[1]","zmin - x[2]"),
                            xmin=-np.infty, ymin=-np.infty, zmin=0, degree=1)
    umin = interpolate(constraint_l, V)
    umax = interpolate(constraint_u, V)

    # Define the solver parameters
    
    problem = NonlinearVariationalProblem(F, u, bc, J=J)
    problem.set_bounds(umin, umax)
    solver  = NonlinearVariationalSolver(problem)
    
    
    
    prm = solver.parameters
  
    #solver.parameters.update(snes_solver_parameters)
    
    # prm['nonlinear_solver'] = 'snes'
    # #prm["snes_solver"]["linear_solver"] = "lu"
    # prm["snes_solver"]["linear_solver"] = "lu"
    # #prm["snes_solver"]["preconditioner"] = "ilu"
    # prm["snes_solver"]["maximum_iterations"] = 100
    # prm["snes_solver"]["absolute_tolerance"] = 1E-10
    # prm["snes_solver"]["relative_tolerance"] = 1E-10
    # prm["snes_solver"]["report"] = True
    # prm["snes_solver"]["error_on_nonconvergence"] = True
    #prm["snes_solver"]["divergence_tolerance"] = 1e30
    
    
    snes_solver_parameters = {"nonlinear_solver": "snes",
                            "snes_solver": {"linear_solver": "gmres",
                                            "preconditioner": "ilu",
                                            "maximum_iterations": 100,
                                            "absolute_tolerance": 1E-9,
                                            "relative_tolerance": 1E-9,
                                            "report": False,
                                            "method":'vinewtonrsls', #vinewtonssls fails to converge
                                            "error_on_nonconvergence": True,
                                            "krylov_solver": {"divergence_limit": 1e10},
                                            "krylov_solver": {"absolute_tolerance": 1E-9},
                                            "krylov_solver": {"relative_tolerance": 1E-9},
                                            "krylov_solver": {"maximum_iterations": 1000}}}
    solver.parameters.update(snes_solver_parameters)
    
    # Set up the non-linear solver
    
    # Solve the nonlinear problem
    # start = time.time()
    with Timer() as t:
        (iter, converged) = solver.solve()
    # end = time.time()

    # solution_time = end-start
    solution_time = t.elapsed()[0]

    # print(iter)
    info(prm, False)
        
    return iter, solution_time, num_of_dofs