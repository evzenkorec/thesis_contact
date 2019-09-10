from pygmsh.built_in import Geometry
from pygmsh.opencascade import Geometry as geooc

from pygmsh import generate_mesh

__all__ = ["generate_half_circle", "generate_cylinder"]

def generate_half_circle(R, res=1):
    geo = Geometry()
    center = geo.add_point((0,R,0), lcar=4*res)
    p1 = geo.add_point((-R,R,0), lcar=4*res)
    p2 = geo.add_point((R,R,0), lcar=4*res)
    p3 = geo.add_point((0,0,0), lcar=res)
    l1 = geo.add_line(p1,p2)
    curve1 = geo.add_circle_arc(p1, center, p3)
    curve2 = geo.add_circle_arc(p3, center, p2)
    loop = geo.add_line_loop([l1, -curve1, -curve2])
    surf = geo.add_plane_surface(loop)
    geo.add_physical(surf, label="1")

    gmesh = generate_mesh(geo, prune_z_0=True)#, geo_filename="test.geo")
    points, cells = gmesh.points, gmesh.cells
    
    from dolfin import Mesh, MPI, cpp
    from dolfin.fem import create_coordinate_map
    from dolfin.cpp.mesh import CellType
    mesh = Mesh(MPI.comm_world, CellType.triangle, gmesh.points, gmesh.cells["triangle"],  [], cpp.mesh.GhostMode.none)
    # Fix for saving functions to file
    cmap = create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    return mesh


def generate_cylinder(R, L, res):
    geo = geooc()
    from numpy import pi
    cylinder = geo.add_cylinder([0,0,R], [0,L,0], R, char_length=res)
    box = geo.add_box([-R,-0.1*L,R], [2.1*R,1.2*L,1.1*R], char_length=res)
    diff = geo.boolean_difference([cylinder], [box])
    # Uniform refinement of points
    # geo.add_raw_code("pts_bo1[] = PointsOf{Volume{bo1};};")
    # geo.add_raw_code("Characteristic Length{pts_bo1[]} ="+"{0};".format(res))

    # Minimum field at line of contact
    geo.add_raw_code("Point(25) = {0,0,0};")
    geo.add_raw_code("Point(26) = {0,"+str(L)+",0};")
    geo.add_raw_code("Line(27) = {25,26};")
    geo.add_raw_code("Field[1] = Distance; Field[1].EdgesList = {27};")
    geo.add_raw_code("Field[2] = Threshold; Field[2].IField=1; Field[2].LcMin = {0}; Field[2].LcMax = {1}; Field[2].DistMin={2}; Field[2].DistMax={3};".format(res, 3*res, 0.3*R, 0.6*R))
    geo.add_raw_code("Field[3] = Min; Field[3].FieldsList = {2};")
    geo.add_raw_code("Background Field = 2;")
    geo.add_physical(diff, label="1")

    gmesh = generate_mesh(geo)
    points, cells = gmesh.points, gmesh.cells
    
    from dolfin import Mesh, MPI, cpp
    from dolfin.fem import create_coordinate_map
    from dolfin.cpp.mesh import CellType
    mesh = Mesh(MPI.comm_world, CellType.tetrahedron, gmesh.points, gmesh.cells["tetra"],  [], cpp.mesh.GhostMode.none)
    # Fix for saving functions to file
    cmap = create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    return mesh
