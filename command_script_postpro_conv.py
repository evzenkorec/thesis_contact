from contact_postprocessing_conv import postprocessing
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Text in command window - text output = True, no text = False
set_log_active(False)

array_penetration = np.load('parameters/array_penetration.npy')
num_points = np.load('parameters/num_points.npy')
num_points = num_points.tolist()
max_penetration = np.load('parameters/max_penetration.npy')
max_penetration = max_penetration.tolist()
min_penetration = np.load('parameters/min_penetration.npy')
min_penetration = min_penetration.tolist()
R = np.load('parameters/R.npy')
R = R.tolist()
E = np.load('parameters/E.npy')
E = E.tolist()
nu = np.load('parameters/nu.npy')
nu = nu.tolist()
penalty = np.load('parameters/penalty.npy')
penalty = penalty.tolist()
penalty_nitsche = np.load('parameters/penalty_nitsche.npy')
penalty_nitsche = penalty_nitsche.tolist()
penalty_nitsche_RFP = np.load('parameters/penalty_nitsche_RFP.npy')
penalty_nitsche_RFP = penalty_nitsche_RFP.tolist()
penalty_auglag = np.load('parameters/penalty_auglag.npy')
penalty_auglag = penalty_auglag.tolist()
theta = np.load('parameters/theta.npy')
theta = theta.tolist()

array_rel_penetration = np.zeros((num_points))
array_contact_force_penalty_top = np.zeros((num_points))
array_contact_force_snes_top = np.zeros((num_points))
array_contact_force_nitsche_top = np.zeros((num_points))
array_contact_force_nitscheRFP_top = np.zeros((num_points))
array_contact_force_auglag_top = np.zeros((num_points))
array_contact_force_Hertz = np.zeros((num_points))

array_rel_contact_force_penalty_top = np.zeros((num_points))
array_rel_contact_force_snes_top = np.zeros((num_points))
array_rel_contact_force_nitsche_top = np.zeros((num_points))
array_rel_contact_force_nitscheRFP_top = np.zeros((num_points))
array_rel_contact_force_auglag_top = np.zeros((num_points))

array_contact_force_penalty_bottom = np.zeros((num_points))
array_contact_force_snes_bottom = np.zeros((num_points))
array_contact_force_nitsche_bottom = np.zeros((num_points))
array_contact_force_nitscheRFP_bottom = np.zeros((num_points))
array_contact_force_auglag_bottom = np.zeros((num_points))
array_contact_force_Hertz = np.zeros((num_points))

array_rel_contact_force_penalty_bottom = np.zeros((num_points))
array_rel_contact_force_snes_bottom = np.zeros((num_points))
array_rel_contact_force_nitsche_bottom = np.zeros((num_points))
array_rel_contact_force_nitscheRFP_bottom = np.zeros((num_points))
array_rel_contact_force_auglag_bottom = np.zeros((num_points))

# Arrays for OOFEM comparison

array_penetration_OOFEM = np.array([R/200, R/50])
array_rel_penetration_OOFEM = array_penetration_OOFEM/R

array_contact_force_OOFEM = np.array([0.78127, 6.5387])
array_contact_force_Hertz_OOFEM = np.zeros(array_contact_force_OOFEM.shape[0])
array_rel_contact_force_OOFEM = np.zeros(array_contact_force_Hertz_OOFEM.shape[0])



for i in range(1): 
    # Load mesh
    mesh = Mesh()
    with XDMFFile("mesh/mesh_" + str(i) + ".xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)
    
    with XDMFFile("mesh/mf_" + str(i) + ".xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    V = VectorFunctionSpace(mesh, "Lagrange", 1) # the same space as the solution space
    u = Function(V)

    # Data for TOP and BOTTOM reaction

    for i in range(array_penetration.shape[0]):
        
        penetration = array_penetration[i]
        array_rel_penetration[i] = array_penetration[i]/R
        
        with XDMFFile("u_PEN_" + str(i) + ".xdmf") as infile:
            infile.read_checkpoint(u, "u", 0)            
        array_contact_force_penalty_top[i] = -postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[0]
        array_contact_force_Hertz[i] = postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[1]
        array_rel_contact_force_penalty_top[i] = 100*(array_contact_force_penalty_top[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]
        array_contact_force_penalty_bottom[i] = postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[2]
        array_rel_contact_force_penalty_bottom[i] = 100*(array_contact_force_penalty_bottom[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]
        
        with XDMFFile("u_SNES_" + str(i) + ".xdmf") as infile:
            infile.read_checkpoint(u, "u", 0)            
        array_contact_force_snes_top[i] = -postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[0]
        array_rel_contact_force_snes_top[i] = 100*(array_contact_force_snes_top[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]
        array_contact_force_snes_bottom[i] = postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[2]
        array_rel_contact_force_snes_bottom[i] = 100*(array_contact_force_snes_bottom[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]

        with XDMFFile("u_NITSCHE_" + str(i) + ".xdmf") as infile:
            infile.read_checkpoint(u, "u", 0)            
        array_contact_force_nitsche_top[i] = -postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[0]
        array_rel_contact_force_nitsche_top[i] =100*(array_contact_force_nitsche_top[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]
        array_contact_force_nitsche_bottom[i] = postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[2]
        array_rel_contact_force_nitsche_bottom[i] =100*(array_contact_force_nitsche_bottom[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]

        with XDMFFile("u_NITSCHE_RFP_" + str(i) + ".xdmf") as infile:
            infile.read_checkpoint(u, "u", 0)            
        array_contact_force_nitscheRFP_top[i] = -postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[0]
        array_rel_contact_force_nitscheRFP_top[i] = 100*(array_contact_force_nitscheRFP_top[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]
        array_contact_force_nitscheRFP_bottom[i] = postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[2]
        array_rel_contact_force_nitscheRFP_bottom[i] = 100*(array_contact_force_nitscheRFP_bottom[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]

        with XDMFFile("u_AUGLAG_" + str(i) + ".xdmf") as infile:
            infile.read_checkpoint(u, "u", 0)            
        array_contact_force_auglag_top[i] = -postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[0]
        array_rel_contact_force_auglag_top[i] = 100*(array_contact_force_auglag_top[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]
        array_contact_force_auglag_bottom[i] = postprocessing(R,E,nu, penetration, penalty, u, mesh, boundary_markers)[2]
        array_rel_contact_force_auglag_bottom[i] = 100*(array_contact_force_auglag_bottom[i] - array_contact_force_Hertz[i])/array_contact_force_Hertz[i]

    # # Get data for OOFEM comparison

    for i in range(array_contact_force_Hertz_OOFEM.shape[0]):
        
        array_contact_force_Hertz_OOFEM[i] = postprocessing(R,E,nu, array_penetration_OOFEM[i], penalty, u, mesh, boundary_markers)[1]
        array_rel_contact_force_OOFEM[i] = 100*(array_contact_force_OOFEM[i] - array_contact_force_Hertz_OOFEM[i])/array_contact_force_Hertz_OOFEM[i]

        # # Savu u as .pvd 
        # file = File("contact_sphere/displacement_NITSCHE_RFP_" + str(i) + ".pvd");
        # file << u
    
    # # Print arrays TOP reaction - debugging for paralell 
    # print(array_rel_penetration)       
    # print(array_contact_force_penalty_top)
    # print(array_contact_force_snes_top)
    # print(array_contact_force_nitsche_top)
    # print(array_contact_force_nitscheRFP_top)
    # print(array_contact_force_auglag_top)
    
    # Reactions vs. penetrations graph TOP reaction - plot .pdf figure
    plt.figure()
    plt.xscale('log', basex=10)
    plt.yscale('log', basey=10)
    graph1 = plt.plot(array_rel_penetration, array_contact_force_penalty_top, '+-', linewidth=0.5, markersize=3)
    graph2 = plt.plot(array_rel_penetration, array_contact_force_snes_top,'x-', linewidth=0.5, markersize=3)
    graph3 = plt.plot(array_rel_penetration, array_contact_force_nitsche_top,'o-', linewidth=0.5, markersize=3)
    graph4 = plt.plot(array_rel_penetration, array_contact_force_nitscheRFP_top,'v-', linewidth=0.5, markersize=3)
    graph5 = plt.plot(array_rel_penetration, array_contact_force_auglag_top,'d-', linewidth=0.5, markersize=3)
    graph6 = plt.plot(array_rel_penetration, array_contact_force_Hertz,'s-', linewidth=0.5, markersize=3)
    graph7 = plt.plot(array_rel_penetration_OOFEM, array_contact_force_Hertz_OOFEM,'*', linewidth=0.5, markersize=5)
    # plt.xlim(left=1e-4)
    # plt.ylim(bottom=1e-4)
    plt.legend(['Penalty', 'SNES', 'Nitsche', 'NitscheRFP', 'Aug.Lag', 'Hertz', 'OOFEM'], loc='upper left')
    #plt.legend(['Penalty', 'Nitsche', 'NitscheRFP', 'Hertz'], loc='upper left')
    # graph2 = plt.plot(xAUX,value, 'b--', linewidth=2)
    # plt.axhline(color = 'k')
    plt.title('Convergence TOP reaction', loc='center')
    plt.xlabel('$penetration/R$')
    plt.ylabel('$Contact  force$')
    # plt.legend(['Herz solution', 'FEM'], loc='upper left')
    plt.savefig("contact_sphere/convergence_top.pdf", format="pdf")


    # Reactions vs. penetrations graph - plot .pdf figure
    plt.figure()
    # plt.xscale('log', basex=10)
    # plt.yscale('log', basey=10)
    graph1 = plt.plot(array_rel_penetration, array_rel_contact_force_penalty_top, '+-', linewidth=0.5, markersize=3)
    graph2 = plt.plot(array_rel_penetration, array_rel_contact_force_snes_top,'x-', linewidth=0.5, markersize=3)
    graph3 = plt.plot(array_rel_penetration, array_rel_contact_force_nitsche_top,'o-', linewidth=0.5, markersize=3)
    graph4 = plt.plot(array_rel_penetration, array_rel_contact_force_nitscheRFP_top,'v-', linewidth=0.5, markersize=3)
    graph5 = plt.plot(array_rel_penetration, array_rel_contact_force_auglag_top,'d-', linewidth=0.5, markersize=3)
    graph6 = plt.plot(array_rel_penetration_OOFEM, array_rel_contact_force_OOFEM,'*', linewidth=0.5, markersize=5)
    # plt.xlim(left=1e-4)
    # plt.ylim(bottom=1e-4)
    plt.legend(['Penalty', 'SNES', 'Nitsche', 'NitscheRFP', 'Aug.Lag.', 'OOFEM'], loc='upper left')
    #plt.legend(['Penalty', 'Nitsche', 'NitscheRFP'], loc='upper left')
    # graph2 = plt.plot(xAUX,value, 'b--', linewidth=2)
    # plt.axhline(color = 'k')
    plt.title('Convergence TOP raction - relative values', loc='center')
    plt.xlabel('penetration/R')
    plt.ylabel('100*(Contact force - Hertz)/Contact force')
    # plt.legend(['Herz solution', 'FEM'], loc='upper left')
    plt.savefig("contact_sphere/convergence_relative_top.pdf", format="pdf")






    # Reactions vs. penetrations graph BOTTOM reaction - plot .pdf figure
    plt.figure()
    plt.xscale('log', basex=10)
    plt.yscale('log', basey=10)
    graph1 = plt.plot(array_rel_penetration, array_contact_force_penalty_bottom, '+-', linewidth=0.5, markersize=3)
    graph2 = plt.plot(array_rel_penetration, array_contact_force_snes_bottom,'x-', linewidth=0.5, markersize=3)
    graph3 = plt.plot(array_rel_penetration, array_contact_force_nitsche_bottom,'o-', linewidth=0.5, markersize=3)
    graph4 = plt.plot(array_rel_penetration, array_contact_force_nitscheRFP_bottom,'v-', linewidth=0.5, markersize=3)
    graph5 = plt.plot(array_rel_penetration, array_contact_force_auglag_bottom,'d-', linewidth=0.5, markersize=3)
    graph6 = plt.plot(array_rel_penetration, array_contact_force_Hertz,'s-', linewidth=0.5, markersize=3)
    graph7 = plt.plot(array_rel_penetration_OOFEM, array_contact_force_Hertz_OOFEM,'*', linewidth=0.5, markersize=5)
    # plt.xlim(left=1e-4)
    # plt.ylim(bottom=1e-4)
    plt.legend(['Penalty', 'SNES', 'Nitsche', 'NitscheRFP', 'Aug.Lag', 'Hertz', 'OOFEM'], loc='upper left')
    #plt.legend(['Penalty', 'Nitsche', 'NitscheRFP', 'Hertz'], loc='upper left')
    # graph2 = plt.plot(xAUX,value, 'b--', linewidth=2)
    # plt.axhline(color = 'k')
    plt.title('Convergence BOTTOM reaction', loc='center')
    plt.xlabel('$penetration/R$')
    plt.ylabel('$Contact  force$')
    # plt.legend(['Herz solution', 'FEM'], loc='upper left')
    plt.savefig("contact_sphere/convergence_bottom.pdf", format="pdf")


    # Reactions vs. penetrations graph - plot .pdf figure
    plt.figure()
    # plt.xscale('log', basex=10)
    # plt.yscale('log', basey=10)
    graph1 = plt.plot(array_rel_penetration, array_rel_contact_force_penalty_bottom, '+-', linewidth=0.5, markersize=3)
    graph2 = plt.plot(array_rel_penetration, array_rel_contact_force_snes_bottom,'x-', linewidth=0.5, markersize=3)
    graph3 = plt.plot(array_rel_penetration, array_rel_contact_force_nitsche_bottom,'o-', linewidth=0.5, markersize=3)
    graph4 = plt.plot(array_rel_penetration, array_rel_contact_force_nitscheRFP_bottom,'v-', linewidth=0.5, markersize=3)
    graph5 = plt.plot(array_rel_penetration, array_rel_contact_force_auglag_bottom,'d-', linewidth=0.5, markersize=3)
    graph6 = plt.plot(array_rel_penetration_OOFEM, array_rel_contact_force_OOFEM,'*', linewidth=0.5, markersize=5)
    # plt.xlim(left=1e-4)
    # plt.ylim(bottom=1e-4)
    plt.legend(['Penalty', 'SNES', 'Nitsche', 'NitscheRFP', 'Aug.Lag.', 'OOFEM'], loc='upper left')
    #plt.legend(['Penalty', 'Nitsche', 'NitscheRFP'], loc='upper left')
    # graph2 = plt.plot(xAUX,value, 'b--', linewidth=2)
    # plt.axhline(color = 'k')
    plt.title('Convergence BOTTOM raction - relative values', loc='center')
    plt.xlabel('penetration/R')
    plt.ylabel('100*(Contact force - Hertz)/Contact force')
    # plt.legend(['Herz solution', 'FEM'], loc='upper left')
    plt.savefig("contact_sphere/convergence_relative_bottom.pdf", format="pdf")


    plt.close("all")

    print("===========================================================================")
    print("GRAPH INFO")
    print("===========================================================================")
    print("Number of elements: {0:8.3e}".format(len(mesh.cells())))
    print("Max mesh size [mm]: {0:8.3e}".format(mesh.hmax()))
    print("Min mesh size [mm]: {0:8.3e}".format(mesh.hmin()))
    print("Max penetration [mm]: {0:8.3e}".format(max_penetration))
    print("Min penetration [mm]: {0:8.3e}".format(min_penetration))
    print("Radius of half-sphere [mm]: {0:8.3e}".format(R))
    print("Young's modulus [MPa]: {0:8.3e}".format(E))
    print("Poisson ratio [-]: {0:8.3e}".format(nu))
    print("Penalty for penalty method [-]: {0:8.3e}".format(penalty))
    print("Penalty for Nitsche method [-]: {0:8.3e}".format(penalty_nitsche))
    print("Penalty for Nitsche-RFP method [-]: {0:8.3e}".format(penalty_nitsche_RFP))
    print("Penalty for Aug. Lag. method [-]: {0:8.3e}".format(penalty_auglag))
    print("===========================================================================")
