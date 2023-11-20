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
import vtk
import vmtk.vmtkmarchingcubes as vmtkm
import vmtk.vmtkcenterlineattributes as vmtkcenterlineattributes
import gmsh
from vmtk import vmtkscripts
from vmtk import vmtkmeshgenerator, vmtksurfaceviewer
import vmtk.vmtkcenterlinemodeller as clm
import vmtk.vmtkimagewriter as vi
import vmtk.vmtksurfacewriter as vmtks
import  vmtk.vmtkcenterlines as vmtkc
import vmtk.vmtknumpytocenterlines as vmtkn
import vmtk.vmtkscripts as vmtksc
import vmtk.vmtkcenterlineviewer as view
import vmtk.vmtkimageviewer as imviewer

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('torus.stl')
triiiimesh = trimesh.load('torus.stl')
print(triiiimesh.vertices[1])
####################################parametros
max_dist=100
min_dist =5
apertura_max=90.0
apertura_min = 90.0
grosor_max = 1
delta= 2 #coeficiente de variacion de apertura
sigma = 0.01 # coeficiente de convergencia
porcentaje_ocupacion= 50.0 #el arbol va a crecer hasta ese porcentaje de ocupacion, dependiendo las leaves qe queden.
cant_converger = 4 #cant de iteraciones iguales para llegar a la convergencia
porcentaje_sampleo = 50 # porcentaje de puntos de atraccion

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
  len_aux = 5
  parent=None
  pos=None
  dir=None
  saveDir=None
  depth=1
  grosor=1

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
      self.depth = p.get_depth() + 1
      self.parent.incrementar_grosor()

  def incrementar_grosor(self):
    self.grosor = self.grosor + 1
    if self.parent:
      self.parent.incrementar_grosor()

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
    #v1 = np.array([0, 0, 20])  # rabbit_free
    #v1 = np.array([0, 0, 0])
    #v1 = np.array([50, 0, 0]) #esfera
    #v1 = np.array([27, -14, 14]) #para el stanford_bunny
    #v1 = np.array([10.5,13,5]) #hand
    #v1 = np.array([3,88,3]) #hand desde dedo
    v1 = np.array([116,17, -15]) #torus
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
    ax.scatter(fx, fy, fz, c='black', marker='o', s=0.001)

    # Scatter plot for branches
    x1 = np.array([branch.pos[0] for branch in self.branches[1:] if branch.parent is not None])
    y1 = np.array([branch.pos[1] for branch in self.branches[1:] if branch.parent is not None])
    z1 = np.array([branch.pos[2] for branch in self.branches[1:] if branch.parent is not None])
    x2 = np.array([branch.parent.pos[0] for branch in self.branches[1:] if branch.parent is not None])
    y2 = np.array([branch.parent.pos[1] for branch in self.branches[1:] if branch.parent is not None])
    z2 = np.array([branch.parent.pos[2] for branch in self.branches[1:] if branch.parent is not None])
    ax.scatter(x1, y1, z1, c='#900040', marker='o', s=0.01)
    ax.scatter(x2, y2, z2, c='#900040', marker='o', s=0.01)

    grosores = []
    #grosoresunitario = []
    nor = np.array([])
    for g in range(len(x1)):
      nor = np.append(nor,self.branches[g+1].grosor)

    # Scatter plot for lines (branches)
    for i in range(len(x1)):
      grosordib = self.branches[i+1].grosor/np.linalg.norm(nor)*grosor_max
      ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], color='#900040', linewidth=grosordib)
      grosores.append((grosordib))
      #grosores.append(self.branches[i+1].grosor)
      #grosoresunitario.append(1)
    print(grosores)

    face_color = (0.5, 0.5, 0.5, 0.1)  # Gray with 50% transparency
    edge_color = (0.5, 0.5, 0.5, 0.1)  # Gray with full opacity

    # Plot the mesh with specified face and edge colors
    ax.plot_trisurf(triiiimesh.vertices[:, 0], triiiimesh.vertices[:,1], triangles=triiiimesh.faces, Z=triiiimesh.vertices[:,2], alpha=0.1)
    #for triangle in your_mesh.vectors:
     #ax.add_collection3d(Poly3DCollection([triangle], facecolors=[face_color], edgecolors=[edge_color], lw=0))
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

    #------------------------------------------------------------------MALLA 3d

    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a VTK polydata object to represent the lines
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    radii = vtk.vtkDoubleArray()
    radii.SetName("TubeRadius")
    radii.SetNumberOfValues(len(grosores))

    # Add points and lines to the polydata
    for i in range(len(x1)):
      points.InsertNextPoint(x1[i], y1[i], z1[i])
      points.InsertNextPoint(x2[i], y2[i], z2[i])
      lines.InsertNextCell(2)
      lines.InsertCellPoint(2 * i)
      lines.InsertCellPoint(2 * i + 1)
      radius = grosores[i]  # Assuming you have a 'radius' value in your data
      radii.InsertNextValue(radius)

    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().AddArray(radii)
    polydata.GetPointData().SetActiveScalars("TubeRadius")

    #writer = vtk.vtkPolyDataWriter()
    #writer.SetFileName("pruebaaa.vtk")
    #writer.SetInputData(polydata)
    #writer.Write()

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    z = np.concatenate((z1, z2))

    radii = np.array(grosores)
    ArrayDict = {
      'Points': np.column_stack((x,y,z)).astype(np.float),
      'PointData': {'Radii': radii.astype(np.float)},  # Add other point data as needed
      'CellData': {'CellPointIds': np.arange(len(x)).reshape(-1, 2)}  # Assuming each segment is a separate line
    }
    numpy_cent = vmtkn.vmtkNumpyToCenterlines()
    numpy_cent.ArrayDict= ArrayDict
    numpy_cent.Execute()
    #v = view.vmtkCenterlineViewer()
    #v.Centerlines = polydata
    #v.Execute()

    # Use vmtkcenterlinemodeller to convert centerlines to an image
    centerlineModeler = clm.vmtkCenterlineModeller()
    centerlineModeler.Centerlines = polydata
    #centerlineModeler.Centerlines = numpy_cent.Centerlines
    centerlineModeler.RadiusArrayName = "TubeRadius"
    centerlineModeler.NegateFunction = True
    centerlineModeler.SampleDimensions = [100,100,100]
    # Execute the algorithm
    centerlineModeler.Execute()

    #vv = imviewer.vmtkImageViewer()
    #vv.Image = centerlineModeler.Image
    #vv.Execute()

    # Get the output image data
    outputImageData = centerlineModeler.Image

    marching = vmtkm.vmtkMarchingCubes()
    marching.Level= -3
    marching.Image = outputImageData
    marching.Execute()

    #v = view.vmtkCenterlineViewer()
    #v.Centerlines = marching.Surface
    #v.Execute()

    sur = marching.Surface
    write_v = vmtks.vmtkSurfaceWriter()
    write_v.Format = "stl"
    write_v.Surface = sur
    write_v.OutputFileName = "outMarching2.stl"
    write_v.Execute()


    # Save the result as a .vtk file

    #vtk_writer = vi.vmtkImageWriter()
    #vtk_writer.Image = outputImageData
    #vtk_writer.Format = "vtk"
    #vtk_writer.OutputFileName = "outputImage2.vtk"
    #vtk_writer.Execute()



    """
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a VTK polydata object to represent the lines
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Add points and lines to the polydata
    for i in range(len(x1)):
      points.InsertNextPoint(x1[i], y1[i], z1[i])
      points.InsertNextPoint(x2[i], y2[i], z2[i])
      lines.InsertNextCell(2)
      lines.InsertCellPoint(2 * i)
      lines.InsertCellPoint(2 * i + 1)

    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Create a tube filter to add thickness to the lines
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(polydata)
    tube_filter.SetRadius(0.2)  # Adjust the radius as needed for thickness
    tube_filter.SetNumberOfSides(32)  # Adjust the number of sides for the tube

    # Create a mapper and actor for the tubes
    tube_mapper = vtk.vtkPolyDataMapper()
    tube_mapper.SetInputConnection(tube_filter.GetOutputPort())
    tube_actor = vtk.vtkActor()
    tube_actor.SetMapper(tube_mapper)
    tube_actor.GetProperty().SetColor(0.56, 0, 0.25)

    # Add the tube actor to the renderer
    renderer.AddActor(tube_actor)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("centerlinevtp.vtp")
    writer.SetInputConnection(tube_filter.GetOutputPort())
    writer.Write()
"""
    """
    # -------------------COMO GRAFICAR CON TUBOS ------------------------------
    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Your existing code here

    # Create a VTK polydata object to represent the lines
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Add points and lines to the polydata
    for i in range(len(x1)):
      points.InsertNextPoint(x1[i], y1[i], z1[i])
      points.InsertNextPoint(x2[i], y2[i], z2[i])
      lines.InsertNextCell(2)
      lines.InsertCellPoint(2 * i)
      lines.InsertCellPoint(2 * i + 1)

    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Create a tube filter to add thickness to the lines
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(polydata)
    tube_filter.SetRadius(0.2)  # Adjust the radius as needed for thickness
    tube_filter.SetNumberOfSides(32)  # Adjust the number of sides for the tube

    # Create a mapper and actor for the tubes
    tube_mapper = vtk.vtkPolyDataMapper()
    tube_mapper.SetInputConnection(tube_filter.GetOutputPort())
    tube_actor = vtk.vtkActor()
    tube_actor.SetMapper(tube_mapper)
    tube_actor.GetProperty().SetColor(0.56, 0, 0.25)

    # Add the tube actor to the renderer
    renderer.AddActor(tube_actor)

    # Create a writer for the tubes' STL file
    tube_stl_writer = vtk.vtkSTLWriter()
    tube_stl_writer.SetFileName("tubes.stl")
    tube_stl_writer.SetInputConnection(tube_filter.GetOutputPort())
    tube_stl_writer.SetFileTypeToBinary()  # Prefer binary format for STL
    tube_stl_writer.Write()

    # Set up a render window and interactor
    render_window.SetSize(800, 800)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    # Start the rendering loop
    render_window.Render()
    iren.Start()
    """
    plt.show()
"""
    #------------------------------------------------------------------MALLA 3d
    # --------------------------------------CON TRIANGULOS-------------------------
    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a VTK polydata object to represent the mesh
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    # Add points to the polydata
    for i in range(len(x1)):
      # Add two points at different Z positions to create a surface
      z1_lower = z1[i]
      z2_lower = z2[i] - 0.1  # Adjust the Z difference as needed
      z1_upper = z1[i]
      z2_upper = z2[i] + 0.1  # Adjust the Z difference as needed

      points.InsertNextPoint(x1[i], y1[i], z1_lower)
      points.InsertNextPoint(x2[i], y2[i], z2_lower)
      points.InsertNextPoint(x1[i], y1[i], z1_upper)
      points.InsertNextPoint(x2[i], y2[i], z2_upper)

      # Create two triangles to form a surface
      triangles.InsertNextCell(3)
      triangles.InsertCellPoint(4 * i)
      triangles.InsertCellPoint(4 * i + 1)
      triangles.InsertCellPoint(4 * i + 2)

      triangles.InsertNextCell(3)
      triangles.InsertCellPoint(4 * i + 1)
      triangles.InsertCellPoint(4 * i + 2)
      triangles.InsertCellPoint(4 * i + 3)

    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    # Create a mapper and actor for the polydata
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.56, 0, 0.25)

    # Add the actor to the renderer
    renderer.AddActor(actor)

    # Set up a render window and interactor
    render_window.SetSize(800, 800)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    # Save the scene as an STL file

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName("3d_scene.stl")
    stl_writer.SetInputData(polydata)

    # Ensure binary format (MeshLab prefers this)
    stl_writer.SetFileTypeToBinary()
    stl_writer.Write()
    render_window.Render()
    iren.Start()

    plt.show()       
"""

###############################MAIN
tree = Tree()
start = time.time()
tree.grow()
end = time.time()
print("tiempo del grow: ", end - start)
tree.show()