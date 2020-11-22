import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
import numpy as np
# surface visulisation
N=50





u = np.linspace(-1, 1, N)
v = np.linspace(0, 2*np.pi, N)
u,v = np.meshgrid(u,v)





# #sphere
# x = np.cos(v)*np.sin(u)
# y = np.sin(v)*np.sin(u)
# z = np.cos(u)

# #recangular sphere
#
# x = u
# y = v
# z = np.sqrt(1-x**2-y**2)



# #Torus
#
# x = (2+1*np.cos(u))*np.cos(v)
# y = (2+1*np.cos(u))*np.sin(v)
# z = 1*np.sin(u)


# #hump looking thing
# x = u*np.sin(v)
# y = u*np.cos(v)
# z = np.exp(-u)
#
# #exotic surface
# x = u*np.cos(v)
# y = u*np.sin(v)
# z = np.sin(u)*np.sin(v)

# #saddle
#
# x = u
# y = v
# z = (u*u -v*v)/50

#mobeius strip

x = (1-u*np.sin(v/2))*np.cos(v)
y = (1-u*np.sin(v/2))*np.sin(v)
z = u*np.cos(v/2)



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
