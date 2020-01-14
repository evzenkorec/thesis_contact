# thesis_contact
These programmes allow plotting the convergence of numerical contact solvers to the Hertz solution for the decreasing penetration values.


Programmes for calculation of the contact of elastic hemisphere with the rigid plane Nitsche and NitscheRFP solvers are not working properly, improvements are necessary!!! To run, first create three folders with the names "parameters", "mesh" and "contact_sphere", then run "generate_mesh.py" (generates mesh) and "save_mesh_restriction.py" (restricts the field of the Lagrange multipliers for the Augmented Lagrangian method). Finally, run "command_scripts" for solvers and postprocessing
