from __future__ import division
import random
from stl import mesh
import numpy as np
import math
import time
import plotly.graph_objects as go

class Leaf:
  reached = False

  def __init__(self,x,y,z):
    self.pos = [x,y,z]

  def reachedM(self):
    self.reached = True

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('Estomac.stl')


x=[]
y=[]
z=[]
vectores=[]
for i in your_mesh.vectors:
  for j in i:
    vectores.append(j)
    x.append(j[0])
    y.append(j[1])
    z.append(j[2])
cant_puntos_inicial = int(20*len(x)/100)
print(cant_puntos_inicial)
indices = random.sample(range(len(x)),cant_puntos_inicial)

leaves  = []
for i in indices:
  leaf = Leaf(x[i], y[i], z[i])
  leaves.append(leaf)

fx = np.array([])
fy = np.array([])
fz = np.array([])
for i in range(len(leaves)):
  fx = np.append(fx, leaves[i].pos[0])
  fy = np.append(fy, leaves[i].pos[1])
  fz = np.append(fz, leaves[i].pos[2])

fig = go.Figure()

# Scatter plot for leaves
fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='markers', marker=dict(color='#900040', size=1.5)))

# Extract vertices and faces from the STL mesh (Replace 'your_mesh' with your actual mesh data)
vertices = your_mesh.vectors

# Create a 3D surface mesh from the STL data
x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
surface = go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='gray')

# Add the surface mesh to the figure
fig.add_trace(surface)

# Hide the Cartesian axis lines (optional)
fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))

# Show the figure
fig.show()