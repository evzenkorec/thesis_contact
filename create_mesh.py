from pygmsh.built_in import Geometry
from pygmsh import generate_mesh

__all__ = ["generate_half_circle"]

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
