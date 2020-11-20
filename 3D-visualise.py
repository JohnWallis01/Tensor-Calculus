import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
import numpy as np
# surface visulisation
N=50

u = np.linspace(0, np.pi, N)
v = np.linspace(-np.pi, np.pi, N)
u,v = np.meshgrid(u,v)

# #sphere
# x = np.cos(v)*np.sin(u)
# y = np.sin(v)*np.sin(u)
# z = np.cos(u)

#hump looking thing
x = u*np.sin(v)
y = u*np.cos(v)
z = np.exp(-u)

#exotic surface
x = u*np.cos(v)
y = u*np.sin(v)
z = np.sin(u)*np.sin(v)


fig,axes = plt.subplots(subplot_kw=dict(projection='3d'))
ax = axes
ax.plot_surface(x, y, z)
plt.show()
