import trimesh
from trimesh import PointCloud

mesh = trimesh.load('torus.stl')
vertices = mesh.vertices
print(len(vertices))
max_points = 10000
if len(vertices) > max_points:
    step = len(vertices) // max_points
    vertices = vertices[::step]
point_cloud = trimesh.points.PointCloud(vertices)

# Get the bounding box of the mesh
bbox = mesh.bounding_box
bbox_dimensions = bbox.extents
# Print the dimensions of the bounding box
print("Bounding Box Dimensions (Largo, Ancho, Alto):", bbox_dimensions)
point_cloud.show()




