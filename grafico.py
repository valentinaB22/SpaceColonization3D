import numpy as np
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trimesh import PointCloud
from stl import mesh

def plot_stl_mesh(stl_filename):
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_filename)

    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the wireframe of the mesh
    # Define the face and edge colors with transparency
    face_color = (0.5, 0.5, 0.5, 0.5)  # Gray with 50% transparency
    edge_color = (0.5, 0.5, 0.5, 0.5)  # Gray with full opacity

    # Plot the mesh with specified face and edge colors
    for triangle in your_mesh.vectors:
        ax.add_collection3d(Poly3DCollection([triangle], facecolors=[face_color], edgecolors=[edge_color], lw=0))
    ax.grid(False)
    # Remove axis lines and ticks
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Auto-scale the plot to fit the mesh
    scale = your_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Show the plot
    plt.show()

# Usage
stl_filename = 'corazonMAX.stl'  # Replace with your STL file's path
#plot_stl_mesh(stl_filename)
mesh = trimesh.load('corazonMAX.stl')
print(mesh.contains([[-10,-10,-10]]))
a=PointCloud(mesh.vertices)
a.show()
mesh.show()
print("vos")
#f=mesh.voxelized(4.0).points
#j=mesh.contains(f)
print("listo")
print(sum(bool(x) for x in j))
print(len(f))