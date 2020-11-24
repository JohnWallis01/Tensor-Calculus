import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
import numpy as np
from D3Geometries import *
# surface visulisation
N=50


Graph = Eliptical_Paraboloid_Map()


u = Graph.c1
v = Graph.c2
u,v = np.meshgrid(u,v)


x = Graph.x(u,v)
y = Graph.y(u,v)
z = Graph.z(u,v)



fig,axes = plt.subplots(subplot_kw=dict(projection='3d'))
ax = axes
ax.plot_surface(x, y, z)


max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)



plt.show()
