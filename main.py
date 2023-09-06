from __future__ import division
import random
from stl import mesh
import numpy as np
import math
import time
import plotly.graph_objects as go

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('rabbit_free.stl')

####################################parametros
max_dist=40
min_dist=5
apertura_max=90.0
apertura_min = 10.0
grosor_dibujo = 30.0
delta= 5 #coeficiente de variacion de apertura
cant_puntos_inicial = 3000
sigma = 0.01 # coeficiente de convergencia
porcentaje_ocupacion= 30.0 #el arbol va a crecer hasta ese porcentaje de ocupacion, dependiendo las leaves qe queden.
cant_converger =3 #cant de iteraciones iguales para llegar a la convergencia

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

indices = random.sample(range(len(x)),cant_puntos_inicial)
#print(len(x))
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
  len_aux = 5.0
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
    v1 = np.array([0, 0, 20])
    v2 = np.array([1, 1, 1])

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
              f = b.dir / np.linalg.norm(b.dir)
              o = (l.pos - b.pos) / np.linalg.norm(l.pos - b.pos)
              c = np.array([f]).dot(o)
              rad = math.acos(float(round(c[0], 6)))
              grado = rad * (360 / math.pi)
              #aper = self.fun_apertura(b)
              aper = self.fun_apertura_automatico(b,ocupacion_actual)
              if (grado < aper):
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
          newB = Branch(None, None, b)
          self.branches.append(newB)
          b.reset()
      if (self.converge(ocupacion_actual, ocupacion_anterior)):
        print("convergeeee")
        break

  # gráfico
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
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=dict(color='blue',size=1)))
    # Scatter plot for parent branches
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=dict(color='green',size=1)))
    lines = []

    for i in range(len(x1)):
      grosor = grosor_dibujo/self.branches[i].get_depth()
      lines.append(
        go.Scatter3d(
          x=[x1[i], x2[i]],
          y=[y1[i], y2[i]],
          z=[z1[i], z2[i]],
          mode='lines',
          line=dict(color='red',width=grosor)
        )
      )
    for line in lines:
      fig.add_trace(line)
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


###############################MAIN
tree = Tree()
start = time.time()
tree.grow()
end = time.time()
print("tiempo del grow: ", end - start)
tree.show()