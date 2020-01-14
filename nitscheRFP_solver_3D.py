from dolfin import *
import time
from dolfin import Timer as t
import numpy as np
# from IPython import embed; embed()

def nitscheRFP_solver(R,E,nu,penetration,penalty,theta, mesh, boundary_markers):

    class Problem(NonlinearProblem):
        def __init__(self, J, F, bcs):
            self.bilinear_form = J
            self.linear_form = F
            self.bcs = bcs
            NonlinearProblem.__init__(self)

        def F(self, b, x):
            assemble(self.linear_form, tensor=b)
            for bc in self.bcs:
                bc.apply(b, x)

        def J(self, A, x):
            assemble(self.bilinear_form, tensor=A)
            for bc in self.bcs:
                bc.apply(A)


    class CustomSolver(NewtonSolver):
        def __init__(self):
            NewtonSolver.__init__(self, mesh.mpi_comm(),
                                PETScKrylovSolver(), PETScFactory.instance())
        
        def solver_setup(self, A, P, problem, iteration):
            self.linear_solver().set_operator(A)
            as_backend_type(A).set_near_nullspace(null_space)

            # Choose conjugate gradient method for the Krylov solver.
            PETScOptions.set("ksp_type", "gmres")
            # Output information from the krylov solver process.
            # PETScOptions.set("ksp_view")
            PETScOptions.set("pc_type", "fieldsplit")
            PETScOptions.set("pc_fieldsplit_block_size", 3)
            PETScOptions.set("fieldsplit_pc_type", "hypre")
            PETScOptions.set("fieldsplit_pc_hypre_type", "boomeramg")
            PETScOptions.set("fieldsplit_pc_hypre_boomeramg_strong_threshold", 0.7)
            PETScOptions.set("pc_fieldsplit_0", "0,1,2")
            # PETScOptions.set("ksp_atol", 1e-11)
            PETScOptions.set("ksp_rtol", 1e-9)
            PETScOptions.set("ksp_max_it", 100000)  
            # PETScOptions.set("ksp_divtol", 1e10)
            PETScOptions.set("pc_fieldsplit_type", "symmetric_multiplicative")
            # Set the smoother to be applied tot he MG levels.
            PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
            PETScOptions.set("mg_levels_ksp_type", "chebyshev")
            PETScOptions.set("mg_levels_pc_type", "jacobi")
            PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)
            PETScOptions.set("mg_levels_esteig_ksp_max_it", 50)
            PETScOptions.set("mg_levels_ksp_chebyshev_esteig_random")
            PETScOptions.set("mg_levels_ksp_max_it", 3)
            # Set the coarse grid solver.
            # Solve the coarse grid in parallel across all processes.
            PETScOptions.set("mg_coarse_ksp_type", "preonly")
            PETScOptions.set("mg_coarse_pc_type", "lu")
            PETScOptions.set("mg_coarse_pc_factor_mat_solver_package", "superlu_dist")
            # Monitor the iterations of the KSP solver to check convergence.
            # PETScOptions.set("ksp_monitor_true_residual")
                           

            self.linear_solver().set_from_options()

            # # get number of levels and prolongations by getMG
            # ksp = self.linear_solver().ksp()
            # pc = ksp.getPC()
            
            # nlevel = pc.getMGLevels() 
            # print(nlevel)
            # for ih in range(nlevel-1, 0, -1):
            #     mat = pc.getMGInterpolation(ih)
            #     print(mat.size)


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
    
    # Characteristic size of the elements of the mesh 
    hMESH = CellDiameter(mesh)
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
    null_space = build_nullspace(V, u.vector())

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
    Pi = psi*dx - theta*(hMESH/E)*(1/penalty)*1/2*dot(pN(u),pN(u))*ds(3)\
        + 1/2*penalty*(E/hMESH)*dot(maculay(-gap(u)-(1/penalty)*(hMESH/E)*pN(u)),maculay(-gap(u)-(1/penalty)*(hMESH/E)*pN(u)))*ds(3)\

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)
    
    # Compute Jacobian of F for iterative "newton_solver"
    J = derivative(F, u, du)

    # Solve nonlinear problem
    problem = Problem(J, F, bc)
    custom_solver = CustomSolver()

    prm = custom_solver.parameters
    # prm["absolute_tolerance"] = 1E-9
    prm["relative_tolerance"] = 1E-9
    prm["maximum_iterations"] = 100
    prm["relaxation_parameter"] = 1.0
    # prm["linear_solver"] = "gmres"
    # prm["preconditioner"] = "petsc_amg"
    # prm["krylov_solver"]["absolute_tolerance"] = 1E-9
    # prm["krylov_solver"]["relative_tolerance"] = 1E-9
    # prm["krylov_solver"]["maximum_iterations"] = 1000

    custom_solver.parameters.update(prm)  

    # start = time.time()
    with Timer() as t:
        custom_solver.solve(problem, u.vector())
    # end = time.time()
    
    # solution_time = end-start  
    solution_time = t.elapsed()[0]      
    
    return solution_time, num_of_dofs