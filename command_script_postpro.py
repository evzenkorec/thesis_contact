from contact_postprocessing_conv import postprocessing
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

penetration = np.load('parameters/penetration.npy')
penetration = penetration.tolist()
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

time_penalty = np.load('parameters/time_penalty.npy')
time_snes = np.load('parameters/time_snes.npy')
time_nitsche = np.load('parameters/time_nitsche.npy')
time_nitscheRFP = np.load('parameters/time_nitscheRFP.npy')
time_auglag = np.load('parameters/time_auglag.npy')

mesh_size = np.load('parameters/mesh_size.npy')

# Reactions vs. penetrations graph - plot .pdf figure
plt.figure()
# plt.xscale('log', basex=10)
# plt.yscale('log', basey=10)
graph1 = plt.plot(mesh_size, time_penalty, '+-', linewidth=0.5, markersize=3)
# graph2 = plt.plot(mesh_size, time_snes,'x-', linewidth=0.5, markersize=3)
graph3 = plt.plot(mesh_size, time_nitsche,'o-', linewidth=0.5, markersize=3)
graph4 = plt.plot(mesh_size, time_nitscheRFP,'v-', linewidth=0.5, markersize=3)
# graph4 = plt.plot(mesh_size, time_auglag,'d-', linewidth=0.5, markersize=3)
# plt.legend(['Penalty', 'SNES', 'Nitsche', 'NitscheRFP', 'Aug. Lag.'], loc='upper left')
# plt.legend(['Penalty', 'Nitsche', 'NitscheRFP', 'Aug. Lag.'], loc='upper left')
plt.legend(['Penalty', 'Nitsche', 'NitscheRFP'], loc='upper left')


# plt.legend(['Penalty', 'Nitsche', 'NitscheRFP'], loc='upper left')
# graph2 = plt.plot(xAUX,value, 'b--', linewidth=2)
# plt.axhline(color = 'k')
plt.title('Complexity', loc='center')
plt.xlabel('Number of DOFs')
plt.ylabel('Time [s]')
# plt.legend(['Herz solution', 'FEM'], loc='upper left')
plt.savefig("contact_sphere/convergence.pdf", format="pdf")

plt.close("all")

print("===========================================================================")
print("GRAPH INFO")
print("===========================================================================")
print("Penetration [mm]: {0:8.3e}".format(penetration))
print("Radius of half-sphere [mm]: {0:8.3e}".format(R))
print("Young's modulus [MPa]: {0:8.3e}".format(E))
print("Poisson ratio [-]: {0:8.3e}".format(nu))
print("Penalty for penalty method [-]: {0:8.3e}".format(penalty))
print("Penalty for Nitsche method [-]: {0:8.3e}".format(penalty_nitsche))
print("Penalty for Nitsche_RFP method [-]: {0:8.3e}".format(penalty_nitsche_RFP))
print("Penalty for Aug. Lag. method [-]: {0:8.3e}".format(penalty_auglag))
print("===========================================================================")