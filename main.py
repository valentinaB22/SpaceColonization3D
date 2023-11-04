from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from stl import mesh
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import trimesh

import plotly.graph_objects as go

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('rabbit_free.stl')
triiiimesh = trimesh.load('rabbit_free.stl')
print(triiiimesh.vertices[1])
####################################parametros
max_dist=1000
min_dist =20
apertura_max=40.0
apertura_min = 20.0
grosor_dibujo = 10.0
delta= 30 #coeficiente de variacion de apertura
sigma = 0.01 # coeficiente de convergencia
porcentaje_ocupacion= 100.0 #el arbol va a crecer hasta ese porcentaje de ocupacion, dependiendo las leaves qe queden.
cant_converger = 4 #cant de iteraciones iguales para llegar a la convergencia
porcentaje_sampleo = 35 # porcentaje de puntos de atraccion

#puntos de la imagen
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
cant_puntos_inicial = int(porcentaje_sampleo*len(x)/100)
print(cant_puntos_inicial)
indices = random.sample(range(len(x)),cant_puntos_inicial)
print(len(x))
#print(indices)

###################################Clase Leaf
class Leaf:
  reached = False

  def __init__(self,x,y,z):
    self.pos = [x,y,z]

  def reachedM(self):
    self.reached = True

###################################Clase branch
class Branch:
  count = 0
  len_aux = 2
  parent=None
  pos=None
  dir=None
  saveDir=None
  depth=1

  def __init__(self, v, d, p):
    if p is None:
     self.parent=None
     self.pos=v
     self.dir=d
     self.saveDir=self.dir
     self.depth = 1
    else:
      self.parent=p
      self.pos=self.parent.next()
      self.dir=self.parent.dir
      self.saveDir=self.dir
      self.depth = p.get_depth()+1

  def get_depth(self):
    return self.depth

  def reset(self):
    self.count=0
    self.dir=self.saveDir

  def next(self):
    a=self.dir[0]*self.len_aux
    b=self.dir[1]*self.len_aux
    c=self.dir[2]*self.len_aux
    v= np.array([a,b,c])
    next = self.pos+v
    return next

####################################Clase tree
class Tree:
  cont = 0
  branches = []
  leaves = []

  def __init__(self):
    # cargo los puntos
    for i in indices:
      leaf = Leaf(x[i], y[i], z[i])
      self.leaves.append(leaf)
    #v1 = np.array([6 ,10 ,30.66895485]) #corazon
    v1 = np.array([0, 0, 20])  # rabbit_free
    #v1 = np.array([0, 0, 0])
    #v1 = np.array([50, 0, 0]) #esfera
    #v1 = np.array([27, -14, 14]) #para el stanford_bunny
    #v1 = np.array([10.5,13,5]) #hand
    #v1 = np.array([3,88,3]) #hand desde dedo
    #v1 = np.array([116,17, -15]) #torus
    #v1 = np.array([6,160,5])  #human
    #v1 = np.array([186, 131, 172])  # aorta
    #v1 = np.array([206, 154, 221])  # pancreas
    #v1 = np.array([177,100,199])  # colon
    #v1 = np.array([153,91,143]) #higado
    #v1 = np.array([897, 1677, -12])  # brain
    v2 = np.array([1,1,1])

    # Crea la raiz del arbol y agrega la rama. Al ser el inicio, no tiene parent.
    root = Branch(v1, v2, None)
    self.branches.append(root)
    current = Branch(None, None, root)

    # Crea la primer rama usando root como parent. Donde pos=root.pos.next(), dir=root.dir, parent=trunk
    while not self.closeEnough(current):
      trunk = Branch(None, None, current)
      self.branches.append(trunk)
      current = trunk

  # Busca los puntos que esten lo suficientemente cerca
  def closeEnough(self, b):
    for l in self.leaves:
      d = (b.pos[0] - l.pos[0]) ** 2 + (b.pos[1] - l.pos[1]) ** 2 + (b.pos[2] - l.pos[2]) ** 2
      if (d < max_dist ** 2):
        return True
    return False

  # Método para setear la magnitud de un vector
  def setMag(self, rand, mag):
    coef = (mag / np.linalg.norm(rand))
    return [rand[0] * coef, rand[1] * coef, rand[2] * coef]

  def fun_apertura(self,branch):
    aper = apertura_max - (delta * branch.get_depth())
    if (aper < apertura_min ):
      return apertura_min
    else:
      return aper

  def fun_apertura_automatico(self,branch,ocupacion_actual):
    aper = apertura_max - (ocupacion_actual/100) * apertura_max
    if (aper < apertura_min):
      return apertura_min
    else:
      return aper

  def converge (self, ocupacion_actual,ocupacion_anterior):
    if ((ocupacion_actual-ocupacion_anterior) <= sigma):
      self.cont = self.cont +1
      if(self.cont > cant_converger):
        return True
    else:
      self.cont =0
      return False
    return False

  #control de contorno
  def ray_intersect(self,triangle, ray_origin, ray_direction):
    # Calculate the normal of the triangle
    triangle_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    triangle_normal /= np.linalg.norm(triangle_normal)
    # Check if the ray is parallel to the triangle
    dot_product = np.dot(triangle_normal, ray_direction)
    if np.abs(dot_product) < 1e-6:
      return False
    # Calculate the intersection point
    t = np.dot(triangle[0] - ray_origin, triangle_normal) / dot_product
    if t < 0:
      return False
    intersection_point = ray_origin + t * ray_direction
    # Check if the intersection point is inside the triangle
    u = np.dot(np.cross(triangle[1] - triangle[0], intersection_point - triangle[0]), triangle_normal)
    v = np.dot(np.cross(triangle[2] - triangle[1], intersection_point - triangle[1]), triangle_normal)
    w = np.dot(np.cross(triangle[0] - triangle[2], intersection_point - triangle[2]), triangle_normal)
    return u >= 0 and v >= 0 and w >= 0

  def is_point_inside_mesh(self, point):
    # Check ray intersection with each triangle in the mesh
    ray_direction = np.array([1, 0, 0])  # You can set your desired ray direction
    intersections = 0
    for triangle in your_mesh.vectors:
      if self.ray_intersect(triangle, point, ray_direction):
        intersections += 1
    # If the number of intersections is odd, the point is inside the mesh
    return intersections % 2 == 1

  def grow(self):
    iter= 0
    ocupacion_actual = 0
    while ocupacion_actual < porcentaje_ocupacion:
      ocupacion_anterior = ocupacion_actual
      print("iter:", iter, " - leaves: ", len(self.leaves), " - branches: ", len(self.branches) ," - porcentaje_ocupacion: ", ocupacion_actual)
      iter = iter + 1
      for l in self.leaves:
        if (l.reached == False):
          closest = "null"
          closestDir = "null"
          record = -1.0
          for b in self.branches:
            # calcula la distancia de la leaf a la branch
            dir = np.array([l.pos[0] - b.pos[0], l.pos[1] - b.pos[1], l.pos[2] - b.pos[2]])
            d = (l.pos[0] - b.pos[0]) ** 2 + (l.pos[1] - b.pos[1]) ** 2 + (l.pos[2] - b.pos[2]) ** 2
            # si es menor q la distancia minima, lo descarta
            if (d < min_dist ** 2):
              l.reachedM()
              closest = "null"
              break
            # busca la branch mas cercana
            elif ((d <= max_dist ** 2) and (closest == "null" or d < record)):
              """
              f = b.dir / np.linalg.norm(b.dir)
              o = (l.pos - b.pos) / np.linalg.norm(l.pos - b.pos)
              c = np.array([f]).dot(o)
              rad = math.acos(float(round(c[0], 6)))
              grado = rad * (360 / math.pi)
              #aper = self.fun_apertura(b)
              aper = self.fun_apertura_automatico(b,ocupacion_actual)
              if (grado < aper):
              """
              closest = b
              closestDir = dir
              record = d
          # si encontró el punto mas cercano, lo normaliza, calcula la nueva direccion y suma una Leaf al count
          if (closest != "null"):
            closestDir = closestDir / np.linalg.norm(closestDir)
            closest.dir = np.array(
              [closest.dir[0] + closestDir[0], closest.dir[1] + closestDir[1], closest.dir[2] + closestDir[2]])
            closest.count = closest.count + 1
      # elimina las puntos que fueron descubiertos
      for i in range(len(self.leaves) - 1, 0, -1):
        if (self.leaves[i].reached):
          self.leaves.pop(i)
      cant_leaves_ocupadas = cant_puntos_inicial - len(self.leaves)
      ocupacion_actual = (cant_leaves_ocupadas * 100) / cant_puntos_inicial
      # se usa para hacer el cremiento de las ramas sobre los 3 ejes, sino lo hace en 2d
      for i in range(len(self.branches)):
        b = self.branches[i]
        if (b.count > 0):
          b.dir = b.dir / b.count
          rand = np.random.random((3))
          mag = 0.3
          rand = self.setMag(rand, mag)
          b.dir = b.dir + rand
          b.dir = b.dir / np.linalg.norm(b.dir)
          #si esta dentro del contorno crece, sino no.
          #if self.is_point_inside_mesh(b.next()):
          if triiiimesh.contains([b.next()]):
            newB = Branch(None, None, b)
            self.branches.append(newB)
            b.reset()
      if (self.converge(ocupacion_actual, ocupacion_anterior)):
        print("converge")
        break

  def show(self):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for leaves
    fx = np.array([leaf.pos[0] for leaf in self.leaves])
    fy = np.array([leaf.pos[1] for leaf in self.leaves])
    fz = np.array([leaf.pos[2] for leaf in self.leaves])
    ax.scatter(fx, fy, fz, c='black', marker='o', s=0.01)

    # Scatter plot for branches
    x1 = np.array([branch.pos[0] for branch in self.branches[1:] if branch.parent is not None])
    y1 = np.array([branch.pos[1] for branch in self.branches[1:] if branch.parent is not None])
    z1 = np.array([branch.pos[2] for branch in self.branches[1:] if branch.parent is not None])
    x2 = np.array([branch.parent.pos[0] for branch in self.branches[1:] if branch.parent is not None])
    y2 = np.array([branch.parent.pos[1] for branch in self.branches[1:] if branch.parent is not None])
    z2 = np.array([branch.parent.pos[2] for branch in self.branches[1:] if branch.parent is not None])
    ax.scatter(x1, y1, z1, c='#900040', marker='o', s=1)
    ax.scatter(x2, y2, z2, c='#900040', marker='o', s=1)

    # Scatter plot for lines (branches)
    for i in range(len(x1)):
      #grosor = grosor_dibujo / self.branches[i + 1].get_depth()
      ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], color='#900040', linewidth=1.5)

    face_color = (0.5, 0.5, 0.5, 0.1)  # Gray with 50% transparency
    edge_color = (0.5, 0.5, 0.5, 0.1)  # Gray with full opacity

    # Plot the mesh with specified face and edge colors
    ax.plot_trisurf(triiiimesh.vertices[:, 0], triiiimesh.vertices[:,1], triangles=triiiimesh.faces, Z=triiiimesh.vertices[:,2], alpha=0.1)
    #for triangle in your_mesh.vectors:
    #  ax.add_collection3d(Poly3DCollection([triangle], facecolors=[face_color], edgecolors=[edge_color], lw=0))
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
    # Hide the axis labels and ticks (optional)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(azim=-50, elev=10)  # Adjust the angles as needed

    plt.show()
"""
  def show(self):
    fx = np.array([])
    fy = np.array([])
    fz = np.array([])
    for i in range(len(self.leaves)):
      fx = np.append(fx, self.leaves[i].pos[0])
      fy = np.append(fy, self.leaves[i].pos[1])
      fz = np.append(fz, self.leaves[i].pos[2])
    fig = go.Figure()
    x1 = np.array([])
    y1 = np.array([])
    z1 = np.array([])
    x2 = np.array([])
    y2 = np.array([])
    z2 = np.array([])
    for i in range(1, len(self.branches)):
      if self.branches[i].parent is not None:
        x1 = np.append(x1, self.branches[i].pos[0])
        y1 = np.append(y1, self.branches[i].pos[1])
        z1 = np.append(z1, self.branches[i].pos[2])
        x2 = np.append(x2, self.branches[i].parent.pos[0])
        y2 = np.append(y2, self.branches[i].parent.pos[1])
        z2 = np.append(z2, self.branches[i].parent.pos[2])
    # Scatter plot for branches
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=dict(color='black', size=1)))
    # Scatter plot for parent branches
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=dict(color='black', size=1)))
    lines = []
    for i in range(len(x1)):
      grosor = grosor_dibujo / self.branches[i].get_depth()
      lines.append(
        go.Scatter3d(
          x=[x1[i], x2[i]],
          y=[y1[i], y2[i]],
          z=[z1[i], z2[i]],
          mode='lines',
          line=dict(color='#900040', width=grosor)
        )
      )
    for line in lines:
      fig.add_trace(line)
    # Extract vertices and faces from the STL mesh
    vertices = your_mesh.vectors
    # Create a 3D surface mesh from the STL data
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    surface = go.Surface(x=x, y=y, z=z,opacity=0.1,colorscale='gray')
    # Add the surface mesh to the figure
    fig.add_trace(surface)
    # Hide the Cartesian axis lines (optional)
    fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    # Show the figure
    fig.show()      
"""

###############################MAIN
tree = Tree()
start = time.time()
tree.grow()
end = time.time()
print("tiempo del grow: ", end - start)
tree.show()