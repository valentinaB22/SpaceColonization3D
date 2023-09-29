from __future__ import division

from matplotlib import pyplot as plt
from stl import mesh
import numpy as np
import math
import time
import plotly.graph_objects as go

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('rabbit_free.stl')

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

puntos=[x,y,z]

#x=x[:5000]
#y=y[:5000]
#z=z[:5000]

####################################parametros
max_dist=40
min_dist=5

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
  len_aux = 5.0
  parent=None
  pos=None
  dir=None
  saveDir=None

  def __init__(self, v, d, p):
    if p is None:
     self.parent=None
     self.pos=v
     self.dir=d
     self.saveDir=self.dir
    else:
      self.parent=p
      self.pos=self.parent.next()
      self.dir=self.parent.dir
      self.saveDir=self.dir

  def reset(self):
    self.count=0
    self.dir=self.saveDir

  def next(self):
    a=self.dir[0]*self.len_aux
    b=self.dir[1]*self.len_aux
    c=self.dir[2]*self.len_aux
    v= np.array([a,b,c])
    #v=self.dir*len
    next = self.pos+v
    return next

####################################Clase tree
class Tree:
  branches = []
  leaves = []

  def __init__(self):
    # cargo los puntos
    for i in range(len(x)):
      leaf = Leaf(x[i], y[i], z[i])
      self.leaves.append(leaf)

    v1 = np.array([0, 0, 20])
    v2 = np.array([0, 0, 20])

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
      # d=np.linalg.norm(b.pos-l.pos)
      d = (b.pos[0] - l.pos[0]) ** 2 + (b.pos[1] - l.pos[1]) ** 2 + (b.pos[2] - l.pos[2]) ** 2
      if (d < max_dist ** 2):
        return True
    return False

  # Método para setear la magnitud de un vector
  def setMag(self, rand, mag):
    coef = (mag / np.linalg.norm(rand))
    return [rand[0] * coef, rand[1] * coef, rand[2] * coef]

  def grow(self):
    iter = 50
    while iter > 0:
      print("iter:", iter, " - leaves: ", len(self.leaves), " - branches: ", len(self.branches))
      iter = iter - 1
      for l in self.leaves:
        if (l.reached == False):
          closest = "null"
          closestDir = "null"
          record = -1.0
          for b in self.branches:
            # calcula la distancia de la leaf a la branch
            dir = np.array([l.pos[0] - b.pos[0], l.pos[1] - b.pos[1], l.pos[2] - b.pos[2]])
            # d = np.linalg.norm(dir)
            d = (l.pos[0] - b.pos[0]) ** 2 + (l.pos[1] - b.pos[1]) ** 2 + (l.pos[2] - b.pos[2]) ** 2
            # si es menor q la distancia minima, lo descarta
            if (d < min_dist ** 2):
              l.reachedM()
              closest = "null"
              break
            # busca la branch mas cercana
            elif ((d <= max_dist ** 2) and (closest == "null" or d < record)):
              # grado=math.degrees(self.angle(l.pos,b.pos))%360
              # if((grado > angulo_min) & (grado < angulo_max)):
              closest = b;
              closestDir = dir;
              record = d;
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

      # se usa para hacer el cremiento de las ramas sobre los 3 ejes, sino lo hace en 2d
      for i in range(len(self.branches)):
        b = self.branches[i]
        if (b.count > 0):
          b.dir = b.dir / b.count
          rand = np.random.random((3))
          mag = 0.3  #########################
          rand = self.setMag(rand, mag)
          b.dir = b.dir + rand
          b.dir = b.dir / np.linalg.norm(b.dir)
          newB = Branch(None, None, b)
          self.branches.append(newB)
          b.reset()
  '''
  # gráfico
  def show(self):
    fx = np.array([])
    fy = np.array([])
    fz = np.array([])  # Array for z-coordinates
    for i in range(len(self.leaves)):
      fx = np.append(fx, self.leaves[i].pos[0])
      fy = np.append(fy, self.leaves[i].pos[1])
      fz = np.append(fz, self.leaves[i].pos[2])  # Append z-coordinate

    fig = go.Figure()

    x1 = np.array([])
    y1 = np.array([])
    z1 = np.array([])  # Array for z-coordinates of branches
    x2 = np.array([])
    y2 = np.array([])
    z2 = np.array([])  # Array for z-coordinates of parent branches

    for i in range(1, len(self.branches)):
      if self.branches[i].parent is not None:
        x1 = np.append(x1, self.branches[i].pos[0])
        y1 = np.append(y1, self.branches[i].pos[1])
        z1 = np.append(z1, self.branches[i].pos[2])
        x2 = np.append(x2, self.branches[i].parent.pos[0])
        y2 = np.append(y2, self.branches[i].parent.pos[1])
        z2 = np.append(z2, self.branches[i].parent.pos[2])

    # Scatter plot for branches
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=dict(color='blue',size=1)))

    # Scatter plot for parent branches
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=dict(color='green',size=1)))

    lines = []
    for i in range(len(x1)):
      lines.append(
        go.Scatter3d(
          x=[x1[i], x2[i]],
          y=[y1[i], y2[i]],
          z=[z1[i], z2[i]],
          mode='lines',
          line=dict(color='red')
        )
      )

    for line in lines:
      fig.add_trace(line)

    fig.update_layout(scene=dict(aspectmode='data'))  # Set aspect mode to 'data' for uniform scaling
    fig.show()

  '''
  def show(self):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for leaves
    fx = np.array([leaf.pos[0] for leaf in self.leaves])
    fy = np.array([leaf.pos[1] for leaf in self.leaves])
    fz = np.array([leaf.pos[2] for leaf in self.leaves])
    ax.scatter(fx, fy, fz, c='black', marker='o', s=0.1)

    # Scatter plot for branches
    x1 = np.array([branch.pos[0] for branch in self.branches[1:] if branch.parent is not None])
    y1 = np.array([branch.pos[1] for branch in self.branches[1:] if branch.parent is not None])
    z1 = np.array([branch.pos[2] for branch in self.branches[1:] if branch.parent is not None])
    x2 = np.array([branch.parent.pos[0] for branch in self.branches[1:] if branch.parent is not None])
    y2 = np.array([branch.parent.pos[1] for branch in self.branches[1:] if branch.parent is not None])
    z2 = np.array([branch.parent.pos[2] for branch in self.branches[1:] if branch.parent is not None])
    ax.scatter(x1, y1, z1, c='black', marker='o', s=1)
    ax.scatter(x2, y2, z2, c='black', marker='o', s=1)

    # Scatter plot for lines (branches)
    for i in range(len(x1)):
      ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], color='#900040')

    ax.grid(False)
    # Remove axis lines and ticks
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Hide the axis labels and ticks (optional)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()
############################main
tree = Tree()
start = time.time()
tree.grow()
end = time.time()
print("tiempo del grow: ", end - start)
tree.show()