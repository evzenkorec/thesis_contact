from dolfin import *
from multiphenics import *
# from ufl import transpose
import time
from dolfin import Timer as t
import numpy as np

def auglag_solver(R,E,nu,penetration,penalty_auglag, mesh, boundary_markers, boundary_restriction):

    class CustomSolver(NewtonSolver):
        def __init__(self):
            NewtonSolver.__init__(self, mesh.mpi_comm(),
                                PETScKrylovSolver(), PETScFactory.instance())
            self.problem = problem

        def solver_setup(self, A, P, problem, iteration):
            self.linear_solver().set_operator(A)
            # as_backend_type(A).set_near_nullspace(null_space)

            # # Choose GMRES method for the Krylov solver.
            PETScOptions.set("ksp_type", "gmres")
            # # Output information from the krylov solver process.
            # # PETScOptions.set("ksp_view")
            PETScOptions.set("pc_type", "ilu")
            
            # PETScOptions.set("ksp_atol", 1e-9)
            # PETScOptions.set("ksp_rtol", 1e-9)
            PETScOptions.set("ksp_rtol", 1e-9)
            PETScOptions.set("ksp_max_it", 100000)  
            # # PETScOptions.set("ksp_divtol", 1e10)
            # # Monitor the iterations of the KSP solver to check convergence.
            # # PETScOptions.set("ksp_monitor_true_residual")                           

            self.linear_solver().set_from_options()
            
        def solve(self):
            NewtonSolver.solve(self, self.problem, self.problem.block_solution.block_vector())
            # Keep subfunctions up to date
            self.problem.block_solution.apply("to subfunctions")

    # Set backend to PETSC
    parameters["linear_algebra_backend"] = "PETSc"

    def build_nullspace(V, x):
        """Function to build null space for 3D elasticity"""

        # Create list of vectors for null space
        nullspace_basis = [x.copy() for i in range(6)]

        # Build translational null space basis
        V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
        V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
        V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

        # Build rotational null space basis
        V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
        V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
        V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
        V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
        V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
        V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

        for x in nullspace_basis:
            x.apply("insert")

        # Create vector space basis and orthogonalize
        basis = VectorSpaceBasis(nullspace_basis)
        basis.orthonormalize()

        return basis                
    
    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
      
    boundary_markers.array()[boundary_markers.array()>300] = 0
        
    hMESH = CellDiameter(mesh)
    # meshSIZE = 1/2*(mesh.hmax() + mesh.hmin())
    meshSIZE = mesh.hmax()
    
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    Vlm = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Vlm], restrict=[None, boundary_restriction])
    num_of_dofs = V.dim()

    # TRIAL/TEST FUNCTIONS #
    dul = BlockTrialFunction(W) # Incremental displacement for Jacobi matrix of iterative "newton_solver"
    (du, dlm) = block_split(dul)
    ul = BlockFunction(W)
    (u, lm) = block_split(ul)
    vm = BlockTestFunction(W)
    (v, vlm) = block_split(vm)

    null_space = build_nullspace(V, u.vector())

    # Definition of normal vectors to the elastic body and to the rigid half-space
    nRIGID = Constant((0.0,0.0,1.0))
    n = FacetNormal(mesh)

    # Definition of tolerance
    tol = DOLFIN_EPS # FEniCS tolerance - necessary for "boundary_markers" and the correct definition of the boundaries 

    
    # Setting the Dirichlet type boundary condition
    # bc1 = DirichletBC(W.sub(0).sub(2), Constant((-penetration)), boundary_markers, 2) 
    bc1 = DirichletBC(W.sub(0).sub(2), Constant((-penetration)), boundary_markers, 2) 
    bP = CompiledSubDomain('sqrt(x[0]*x[0]+x[1]*x[1]) < meshSIZE && near(x[2], R, tol) && on_boundary',meshSIZE = meshSIZE, R=R, tol = tol)
    bP.mark(boundary_markers, 4)    
    bc2 = DirichletBC(W.sub(0).sub(0), Constant((0)), boundary_markers, 4)
    bc3 = DirichletBC(W.sub(0).sub(1), Constant((0)), boundary_markers, 4)
    bc = BlockDirichletBC([bc1,bc2,bc3])
    
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
  
    
    F = [inner(sigma(u), epsilon(v))*dx + 1/(penalty_auglag*E/hMESH)*maculay(-lm-penalty_auglag*E/hMESH*(gap(u)))*(-penalty_auglag*E/hMESH*v[2])*ds(3), 
     - 1/(penalty_auglag*E/hMESH)*dot(lm,vlm)*ds(3) + 1/(penalty_auglag*E/hMESH)*maculay(-lm-penalty_auglag*E/hMESH*(gap(u)))*(-vlm)*ds(3)]  
    
    # Compute Jacobian of F for iterative "newton_solver"
    J = block_derivative(F, ul, dul)
    # from IPython import embed;embed()
    
    problem = BlockNonlinearProblem(F, ul, bc, J)
    custom_solver = CustomSolver()

    prm = custom_solver.parameters
    prm["absolute_tolerance"] = 1E-9
    prm["relative_tolerance"] = 1E-9
    prm["maximum_iterations"] = 100
    prm["relaxation_parameter"] = 1.0
    # prm["linear_solver"] = "gmres"
    # prm["preconditioner"] = "ilu"
    # prm["krylov_solver"]["absolute_tolerance"] = 1E-9
    # prm["krylov_solver"]["relative_tolerance"] = 1E-9
    # prm["krylov_solver"]["maximum_iterations"] = 1000

    custom_solver.parameters.update(prm)
    
    with Timer() as t:
        custom_solver.solve()
    
    solution_time = t.elapsed()[0]
    
    # u, lm = ul.block_split(deepcopy=False)
        
    return solution_time, num_of_dofs