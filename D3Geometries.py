from sympy import *
import numpy as np

#these work best when Z = f(x,y)
t = symbols('t')
u = Function('u')(t)
v = Function('v')(t)
N=50
# Geometrical Maps
class Sphere_Map():
    """docstring for Sphere."""

    def __init__(self):
        super(Sphere_Map, self).__init__()
        self.X = cos(v)*sin(u)
        self.Y = sin(v)*sin(u)
        self.Z = cos(u)
        self.c1 = np.linspace(0, np.pi, N)
        self.c2 = np.linspace(0, 2*np.pi, N)
    def x(self,u,v):
         return np.cos(v)*np.sin(u)
    def y(self,u,v):
        return np.sin(v)*np.sin(u)
    def z(self,u,v):
        return np.cos(u)
    def boundary(self,condition):
        return False

class Sphere_Map_XY():
    """docstring for Sphere."""

    def __init__(self):
        super(Sphere_Map_XY, self).__init__()
        self.X = u
        self.Y = v
        self.Z = sqrt(2-u**2-v**2)
        self.c1 = np.linspace(-1, 1, N)
        self.c2 = np.linspace(-1, 1, N)
    def x(self,u,v):
        return u
    def y(self,u,v):
        return v
    def z(self,u,v):
        return np.sqrt(2-u**2-v**2)
    def boundary(self,condition):
        if condition[0][0]**2 + condition[0][1]**2 > 2:
            return True
        else:
            return False

class Plane_Map():
    """docstring for Plane_Map."""

    def __init__(self):
        super(Plane_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = 0
        self.c1 = np.linspace(-10, 10, N)
        self.c2 = np.linspace(-10, 10, N)
    def x(self,u,v):
        return u
    def y(self,u,v):
        return v
    def z(self,u,v):
        return 0.5*v-u
    def boundary(self,condition):
        return False

class Exotic_Map():
    """docstring for Exotic_Map."""

    def __init__(self):
        super(Exotic_Map, self).__init__()
        self.X = u*cos(v)
        self.Y = u*sin(v)
        self.Z = sin(u)*sin(v)
        self.c1 = np.linspace(0, 2*np.pi, N)
        self.c2 = np.linspace(0, 2*np.pi, N)
    def x(self,u,v):
        return u*np.cos(v)
    def y(self,u,v):
        return u*np.sin(v)
    def z(self,u,v):
        return np.sin(u)*np.sin(v)
    def boundary(self,condition):
        if condition[0][0]<0 or condition[0][0]>2*np.pi:
            return True
        else:
            return False
            
class Hump_Map():
    """docstring for Exotic_Map."""

    def __init__(self):
        super(Hump_Map, self).__init__()
        self.X = u*sin(v)
        self.Y = u*cos(v)
        self.Z = exp(-u)
        self.c1 = np.linspace(0, 1, N)
        self.c2 = np.linspace(0, np.pi*2, N)
    def x(self,u,v):
        return u*np.sin(v)
    def y(self,u,v):
        return u*np.cos(v)
    def z(self,u,v):
        return np.exp(-u)
    def boundary(self,condition):
        if condition[0][0]<0 or condition[0][0]>1:
            return True
        else:
            return False

class Torus_Map(object):
    """docstring for Torus."""

    def __init__(self):
        super(Torus_Map, self).__init__()
        self.X = (2+1*cos(u))*cos(v)
        self.Y = (2+1*cos(u))*sin(v)
        self.Z = 1*sin(u)
        self.c1 = np.linspace(0, 2*np.pi, N)
        self.c2 = np.linspace(0, 2*np.pi, N)
    def x(self,u,v):
         return (2+1*np.cos(u))*np.cos(v)
    def y(self,u,v):
         return (2+1*np.cos(u))*np.sin(v)
    def z(self,u,v):
         return 1*np.sin(u)
    def boundary(self,condition):
        return False

class Saddle_Map(object):
    """docstring for Saddle_Map."""

    def __init__(self):
        super(Saddle_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = (u*u -v*v)/50
        self.c1 = np.linspace(-25, 25, N)
        self.c2 = np.linspace(-25, 25, N)
    def x(self,u,v):
        return u
    def y(self,u,v):
        return v
    def z(self,u,v):
        return (u*u -v*v)/50
    def boundary(self,condition):
        if condition[0][0]**2 + condition[0][1]**2 > 10000:
            return True
        else:
            return False

class Mobeius_Map(object):
    """docstring for Mobeius_Map."""

    def __init__(self, arg):
        super(Mobeius_Map, self).__init__()
        self.X = (1-u*sin(v/2))*cos(v)
        self.Y = (1-u*sin(v/2))*sin(v)
        self.Z = u*cos(v/2)
        self.c1 = np.linspace(-0.5, 0.5, N)
        self.c2 = np.linspace(0, 2*np.pi, N)
    def x(self,u,v):
        return (1-u*np.sin(v/2))*np.cos(v)
    def y(self,u,v):
        return (1-u*np.sin(v/2))*np.sin(v)
    def z(self,u,v):
        return u*np.cos(v/2)
    def boundary(self,condition):
        if condition[0][0] > 0.5 or condition[0][0] < -0.5: #or condition[0][1] > np.pi*2 or condition[0][1] < 0:
            return True
        else:
            return False
