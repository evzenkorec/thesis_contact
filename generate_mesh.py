try:
    from pygmsh.built_in import Geometry
    from pygmsh.opencascade import Geometry as geooc

    from pygmsh import generate_mesh
    import meshio
    import numpy as np
except ImportError:
    pass
__all__ = ['R','res']
#__all__ = ["generate_half_circle", "generate_cylinder"]

for i in range(1):
    def generate_sphere(R, res):
        geo = geooc()
        from numpy import pi
        #cylinder = geo.add_cylinder([0,0,R], [0,L,0], R, char_length=res)
        #box = geo.add_box([-R,-0.1*L,R], [2.1*R,1.2*L,1.1*R], char_length=res)
        #diff = geo.boolean_difference([cylinder], [box])
        # Uniform refinement of points
        # geo.add_raw_code("pts_bo1[] = PointsOf{Volume{bo1};};")
        # geo.add_raw_code("Characteristic Length{pts_bo1[]} ="+"{0};".format(res))
        box = geo.add_box([-R,-R,R], [2*R,2*R, 1.1*R])
        sphere = geo.add_ball([0,0,R],R)
        diff = geo.boolean_difference([sphere], [box])
        geo.add_raw_code("Physical Surface(2) = {2};")
        geo.add_raw_code("Physical Surface(1) = {1};")

        # Minimum field at line of contact
        geo.add_raw_code("Point(25) = {0,0,0};")
        geo.add_raw_code("Field[1] = Distance; Field[1].NodesList = {25};")
        geo.add_raw_code("Field[2] = Threshold; Field[2].IField=1; Field[2].LcMin = {0}; Field[2].LcMax = {1}; Field[2].DistMin={2}; Field[2].DistMax={3};".format(res, 5*res, 0.3*R, 0.6*R)) #5
        geo.add_raw_code("Field[3] = Min; Field[3].FieldsList = {2};")
        geo.add_raw_code("Background Field = 2;")
        geo.add_physical(diff, label="1")

        gmesh = generate_mesh(geo, geo_filename='mesh/sphere.geo')
        points, cells = gmesh.points, gmesh.cells
        meshio.write("mesh/mesh_" + str(i) + ".xdmf", meshio.Mesh(points=gmesh.points, cells={"tetra": gmesh.cells["tetra"]}))
        meshio.write("mesh/mf_" + str(i) + ".xdmf", meshio.Mesh(points=gmesh.points, cells={"triangle": gmesh.cells["triangle"]},
                                        cell_data={"triangle": {"name_to_read": gmesh.cell_data["triangle"]["gmsh:physical"]}}))
        
    R = 10
    res = 0.5 - i*0.1
    if __name__=='__main__':
        #generate_half_circle(R, res=res)
        generate_sphere(R, res)

     
