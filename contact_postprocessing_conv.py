from dolfin import *
# from IPython import embed; embed()

def postprocessing(R,E,nu,penetration,penalty,u):
    
    import matplotlib
    matplotlib.get_backend()
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    #mesh = Mesh()
    #with XDMFFile("mesh/mesh.xdmf") as infile:
    #    infile.read(mesh)
    mesh = u.function_space().mesh()
    mvc = MeshValueCollection("size_t", mesh, 1)
        
    with XDMFFile("mesh/mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc) 
    boundary_markers.array()[boundary_markers.array()>300] = 0 
  
    # Characteristic size of elements
    # hMESH = CellDiameter(mesh)
    meshSIZE = 1/2*(mesh.hmax() + mesh.hmin())
    # h_avg = (hMESH('+') + hMESH('-')) / 2.0

    # Define and load function u (displacement) and test function v
    V = VectorFunctionSpace(mesh, "Lagrange", 1) # the same space as the solution space
    v  = TestFunction(V)
          
    # Elasticity parameters
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)

    # Functions for postprocessing
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
    
    # Definition of normal vectors to the elastic body and to the rigid half-space
    nRIGID = Constant((0.0,0.0,1.0))
    n = FacetNormal(mesh)
    
    # Definition of tolerance for "boundary_markers"
    tol = DOLFIN_EPS 

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

    File("contact_sphere/boundary_markers.pvd") << boundary_markers

    # Definition of surface integral subdomains
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # Analytical Hertz solution
    def Hertz_solution():
        Econ = E/(1-nu**2) # Auxiliary Hertz modulus
        nucon = nu
        a = np.sqrt(R*penetration) # Hertz contact radius
        F = 4*a**3*Econ/(3*R) # Hertz contact force
        p0 = 3*F/(2*np.pi*a**2) # Maximum contact pressure - traction component in z-direction

        # Contact pressure over x axis in contact zone
        def pANALYTIC(x):
            return p0*(1-x**2/a**2)**(1/2)
        
        # Points for evaluation 
        sizeVALUE = 2*a/meshSIZE*10
        sizeVALUE = convert_to_integer(sizeVALUE)
        value = np.zeros((sizeVALUE,1))
        xAUX = np.linspace(-a + tol, a - tol, sizeVALUE)
        zAUX = R-np.sqrt(R**2-xAUX**2) + meshSIZE/10
        yAUX = np.zeros((sizeVALUE,)) 
        coordAUX = np.array([xAUX,yAUX,zAUX])

        # Definition of contact pressure function from FEM solution (from sigma_z) 
        V0 = FunctionSpace(mesh, "DG", 0)
        p = Function(V0) # Contact pressure - for post-processing
        p.assign(-project(sigma(u)[2, 2], V0))

        for i in range(0, coordAUX.shape[1]): 
            value[i] = p(coordAUX[:,i])

        # plot .pdf figure
        plt.figure()
        graph4 = plt.plot(xAUX, pANALYTIC(xAUX),'k', linewidth=2)
        graph3 = plt.plot(xAUX,value, 'b--', linewidth=2)
        plt.axhline(color = 'k')
        plt.title('Contact pressure', loc='center')
        plt.xlabel('$x$')
        plt.ylabel('$p(x)$')
        plt.legend(['Herz solution', 'FEM'], loc='upper left')
        plt.savefig("contact_sphere/contact_pressure.pdf", format="pdf")

        return p0, F
    
    # # Maximum contact pressure for penalty approach - precise value in nodes
    # def max_normal_contact_traction_NODAL():
        
    #     # Find value and coordinate of max contact pressure
    #     (ux,uy,uz) = u.split(deepcopy=True)
    #     Vz = FunctionSpace(mesh, "CG", 1)
    #     X_ = Vz.tabulate_dof_coordinates()
    #     z_ = X_[:,2]
    #     contact_pressure = Function(Vz)
    #     contact_pressure.vector()[:] = penalty*E/meshSIZE*1/2*(np.abs(-z_ - uz.vector()) - z_ - uz.vector())    
    #     max_pressure_1 = max(contact_pressure.vector().get_local())

    #     V1 = FunctionSpace(mesh, "CG", 1)
    #     contact_pressure_function = interpolate(contact_pressure,V1)
        
    #     pos = np.argwhere(contact_pressure_function.vector().get_local() == max_pressure_1)
    #     X = V1.tabulate_dof_coordinates()
    #     coord_1 = X[pos,:]
        
    #     # Find contact pressure in (0,0,0) - theoretical coordinate of maximum
    #     point_max_pressure = (0,0,0)
    #     max_pressure_2 = contact_pressure_function(point_max_pressure)

    #     return max_pressure_1, coord_1, max_pressure_2   

    # # # Maximum contact pressure for penalty approach - projection
    # # def max_normal_contact_traction_PROJECTION():
        
    # #     # Find value and coordinate of max contact pressure
    # #     V0 = FunctionSpace(mesh, "DG", 0)
    # #     contact_pressure2 = penalty*E/meshSIZE*maculay(-gap(u))
    # #     contact_pressure2 = project(contact_pressure2, V0)
    # #     contact_pressure_function2 =  interpolate(contact_pressure2,V0)
    # #     max_pressure2 = max(contact_pressure2.vector().get_local())

    # #     # Find contact pressure in (0,0,0) - theoretical coordinate of maximum
    # #     point_max_pressure = (0,0,0)
    # #     value2 = contact_pressure_function2(point_max_pressure)
        
    # #     return max_pressure2


    # # Calculation of contact force from reactions in nodes -  (bottom - contact) subdomain
    # def contact_subdomain_reaction():
        
    #     bc_func_bottom = DirichletBC(V, Constant((1,2,3)), boundary_markers, 3)

    #     boundary_func_bottom = Function(V)
        
    #     bc_func_bottom.apply(boundary_func_bottom.vector())

    #     reaction = assemble(inner(epsilon(v),sigma(u))*dx)
    #     reaction_func = Function(V)
    #     reaction_func.vector()[:] = reaction

    #     # x component of force
    #     x_r_sum_bottom =0
    #     for dof in np.argwhere(boundary_func_bottom.vector().get_local()==1):
    #         x_r_sum_bottom+= reaction.get_local()[dof]

    #     # y component of force
    #     y_r_sum_bottom =0
    #     for dof in np.argwhere(boundary_func_bottom.vector().get_local()==2):
    #         y_r_sum_bottom+= reaction.get_local()[dof]
        
    #     # z component of force
    #     z_r_sum_bottom =0
    #     for dof in np.argwhere(boundary_func_bottom.vector().get_local()==3):
    #         z_r_sum_bottom+= reaction.get_local()[dof]

    #     # Save reactions in nodes as .pvd
    #     File('contact_sphere/reaction.pvd').write(reaction_func)
        
    #     return x_r_sum_bottom, y_r_sum_bottom, z_r_sum_bottom



    # Calculation of contact force from reactions in nodes -  (top - contact) subdomain
    def top_subdomain_reaction():

        bc_func_top1 = DirichletBC(V, Constant((1,2,3)), boundary_markers, 2)
        bc_func_top2 = DirichletBC(V, Constant((1,2,3)), boundary_markers, 4)
        
        boundary_func_top = Function(V)

        
        bc_func_top1.apply(boundary_func_top.vector())
        bc_func_top2.apply(boundary_func_top.vector())
        reaction = assemble(inner(epsilon(v),sigma(u))*dx)
        reaction_func  = Function(V)
        reaction_func.vector()[:] = reaction

        # x component of force
        x_r_sum_top =0
        for dof in np.argwhere(boundary_func_top.vector().get_local()==1):
            x_r_sum_top+= reaction.get_local()[dof]

        # y component of force
        y_r_sum_top =0
        for dof in np.argwhere(boundary_func_top.vector().get_local()==2):
            y_r_sum_top+= reaction.get_local()[dof]

        # z component of force      
        z_r_sum_top =0
        for dof in np.argwhere(boundary_func_top.vector().get_local()==3):
            z_r_sum_top+= reaction.get_local()[dof]
          
        return x_r_sum_top, y_r_sum_top, z_r_sum_top

    

    #Calculation of von Mises stresses
    # def von_mises():
    #     V0 = FunctionSpace(mesh, "DG", 0)
    #     s = sigma(u) - (1./3)*tr(sigma(u))*Identity(3) # deviatoric stress
    #     von_Mises = sqrt(3./2*inner(s, s))
    #     # V = FunctionSpace(mesh, 'DG', 0)
    #     von_Mises = project(von_Mises, V0)

    #     min_von_Mises = von_Mises.vector().get_local().min()
    #     max_von_Mises = von_Mises.vector().get_local().max()
        
    #     # Save von Mises stress as .pvd
    #     file = File("contact_sphere/VMstress.pvd")
        
    #     return min_von_Mises, max_von_Mises

    # # Calculation of the norm of displacement
    # def mag_displacement():
    #     V0 = FunctionSpace(mesh, "DG", 0)
        
    #     u_magnitude = sqrt(dot(u, u))
    #     u_magnitude = project(u_magnitude, V0)

    #     min_displ_mag = u_magnitude.vector().get_local().min()
    #     max_displ_mag = u_magnitude.vector().get_local().max()

    #     #Save the normm of displacement as .pvd
    #     file = File("contact_sphere/MAGdisplacement.pvd")
    #     file << u_magnitude

    #     return min_displ_mag, max_displ_mag


    # # Calculation of contact force and maximum pressure from sigma_z 
    # def contact_stress():
        
    #     # Definition of contact pressure function from FEM solution (from sigma_z) 
    #     V0 = FunctionSpace(mesh, "DG", 0)
    #     p = Function(V0, name="Contact pressure") # Contact pressure - for post-processing
    #     p.assign(-project(sigma(u)[2, 2], V0))

    #     # Save value and coordinate of max stress value
    #     maxp = max(p.vector().get_local())
    #     val = np.argwhere(p.vector().get_local() == maxp)
    #     Xproject = V0.tabulate_dof_coordinates()
    #     coord = Xproject[val,:]

    #     # Save value of contact pressure at (0,0,0) - theoretical coordinate of maximum
    #     p_zero = p(0,0,0)

    #     # Calculation of contact force
    #     F_top = 1e-3*assemble(p*(ds(2)+ds(4)))
    #     F_bottom = 1e-3*assemble(maculay(p)*ds(3))

    #     # Save contact pressure as .pvd
    #     file = File("contact_sphere/pressure.pvd")
    #     file << p

    #     return maxp, coord, p_zero, F_top, F_bottom

    # # Calculation of contact force and maximum pressure from traction force 
    # def contact_traction():
    #     V0 = FunctionSpace(mesh, "DG", 0)

    #     # Definition of traction force
    #     traction = dot(n,sigma(u))
    #     contact_traction = dot(traction,nRIGID)

    #     # Projection of traction on the surface of the body
    #     uB, v = TrialFunction(V0), TestFunction(V0)
    #     a = inner(uB, v)*ds
    #     l = inner(contact_traction,v)*ds
    #     A = assemble(a, keep_diagonal=True)
    #     L = assemble(l)
    #     A.ident_zeros()
    #     trac = Function(V0)
    #     solve(A, trac.vector(), L)

    #     # Calculation of contact force
    #     F_top = 1e-3*assemble(-contact_traction*(ds(2)+ds(4)))
    #     F_bottom = 1e-3*assemble(maculay(contact_traction)*ds(3))
        
    #     # Maximum contact pressure and its coordinate
    #     max_cont_traction = max(trac.vector().get_local())
        
    #     val = np.argwhere(trac.vector().get_local() == max_cont_traction)
    #     Xproject = V0.tabulate_dof_coordinates()
    #     coord = Xproject[val,:]

    #     # Contact pressure in (0,0,0) - coordinate of theroetical maximum 
    #     cont_trac_zero = trac(0,0,0)
        
    #     File('contact_sphere/boundary_traction.pvd') << trac

    #     return F_top, F_bottom, max_cont_traction, cont_trac_zero, coord

    # # Norm of the "jump" on inter-elemental facets
    # def jumpNorm():
    #     JN = np.sqrt(assemble(dot(jump(u), jump(u))*dS))
    #     return JN

    # # Penetration norm 
    # def penetrationNORM():
    #     PN = np.sqrt(assemble(inner(maculay(-gap(u)),maculay(-gap(u)))*ds(3)))
    #     return PN

    # print("===========================================================================")
    # print("CALCULATION INFO")
    # print("===========================================================================")
    # print("Jump norm: {0:8.3e}".format(jumpNorm()))
    # print("Penetration norm: {0:8.3e}".format(penetrationNORM()))
    # print("===========================================================================")
    # print("Contact force - calculation from nodal reactions")
    # print("Contact force BOTTOM - FE: {0:8.5f} kN, Hertz: {1:8.5f} kN".format(1e-3*contact_subdomain_reaction()[2][0],1e-3*Hertz_solution()[1]))
    # print("Frictional force_x BOTTOM - FE: {0:8.5f} kN, Hertz: {1:8.5f} kN".format(1e-3*contact_subdomain_reaction()[0][0],0))
    # print("Frictional force_y BOTTOM - FE: {0:8.5f} kN, Hertz: {1:8.5f} kN".format(1e-3*contact_subdomain_reaction()[1][0],0))
    # print("Contact force TOP - FE: {0:8.5f} kN, Hertz: {1:8.5f} kN".format(1e-3*top_subdomain_reaction()[2][0],1e-3*Hertz_solution()[1]))
    # print("Frictional force_x TOP - FE: {0:8.5f} kN, Hertz: {1:8.5f} kN".format(1e-3*top_subdomain_reaction()[0][0],0))
    # print("Frictional force_y TOP - FE: {0:8.5f} kN, Hertz: {1:8.5f} kN".format(1e-3*top_subdomain_reaction()[1][0],0))
    # print("---------------------------------------------------------------------------")
    # print("Contact force - calculation from stress")
    # print("Contact force - stress sigma_z BOTTOM   FE: {0:8.5f} kN Hertz: {1:8.5f} kN".format(contact_stress()[4], 1e-3*Hertz_solution()[1]))
    # print("Contact force - stress sigma_z TOP   FE: {0:8.5f} kN Hertz: {1:8.5f} kN".format(contact_stress()[3], 1e-3*Hertz_solution()[1]))
    # print("---------------------------------------------------------------------------")
    # print("Contact force - calculation from tractions")
    # print("Contact force - traction BOTTOM  FE: {0:8.5f} kN Hertz: {1:8.5f} kN".format(contact_traction()[1], 1e-3*Hertz_solution()[1]))
    # print("Contact force - traction TOP  FE: {0:8.5f} kN Hertz: {1:8.5f} kN".format(contact_traction()[0], 1e-3*Hertz_solution()[1]))
    # print("===========================================================================")
    # print("Maximum tractions - penalty approach nodal")
    # print("Maximum contact traction FE - penalty approach nodal: {0:8.5f} MPa Hertz: {1:8.5f} MPa".format(max_normal_contact_traction_NODAL()[0], Hertz_solution()[0]))
    # print("Maximum contact traction FE - penalty approach nodal - coordinate x: {0:8.5f}, coordinate z: {1:8.5f}".format(max_normal_contact_traction_NODAL()[1][0][0][0],max_normal_contact_traction_NODAL()[1][0][0][1]))
    # print("Contact traction FE in [0,0]- penalty approach nodal: {0:8.5f} MPa Hertz: {1:8.5f} MPa".format(max_normal_contact_traction_NODAL()[2], Hertz_solution()[0]))
    # print("---------------------------------------------------------------------------")  
    # print("Maximum tractions - stress approach")
    # print("Maximum contact traction - stress approach - FE: {0:8.5f} MPa Hertz: {1:8.5f} MPa".format(contact_stress()[0], Hertz_solution()[0]))
    # print("Maximum contact traction FE - stress approach - coordinate x: {0:8.5f}, coordinate z: {1:8.5f}".format(contact_stress()[1][0][0][0],contact_stress()[1][0][0][1]))
    # print("Contact traction FE in [0,0]- stress approach: {0:8.5f} MPa Hertz: {1:8.5f} MPa".format(contact_stress()[2], Hertz_solution()[0]))
    # print("---------------------------------------------------------------------------")  
    # print("Maximum tractions - tractions approach")
    # print("Maximum contact traction - tractions approach - FE: {0:8.5f} MPa Hertz: {1:8.5f} MPa".format(contact_traction()[2], Hertz_solution()[0]))
    # print("Maximum contact traction FE - tractions approach - coordinate x: {0:8.5f}, coordinate z: {1:8.5f}".format(contact_traction()[4][0][0][0],contact_traction()[4][0][0][1]))
    # print("Contact traction FE in [0,0]- tractions approach: {0:8.5f} MPa Hertz: {1:8.5f} MPa".format(contact_traction()[3], Hertz_solution()[0]))
    # print("===========================================================================")
    # print('Displacement norm - min: {0:8.5f} mm, max: {1:8.5f} mm'.format(mag_displacement()[0],mag_displacement()[1]))
    # print('Von Mises stress - min:{0:8.5f} MPa, max:{1:8.5f} MPa'.format(von_mises()[0], von_mises()[1]))
    plt.close("all")
    return 1e-3*top_subdomain_reaction()[2][0], 1e-3*Hertz_solution()[1]


# # for i in range(3):
#     f_name = "u_" + str(i) + ".xdmf"
#     nitsche_solver(R,E,nu,penetration,penalty, f_name)

#     u = nitsche_solver(R,E,nu,penetration,penalty)
#     with XDMFFile("u_" + str(i) + ".xdmf") as xdmf:
#         xdmf.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, append=False)


