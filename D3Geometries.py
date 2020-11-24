from sympy import *
import numpy as np

#these work best when Z = f(x,y)
t = symbols('t')
u = Function('u')(t)
v = Function('v')(t)
N=50
# Geometrical Maps
class Sphere_Map():


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

class Hypbolic_Paraboloid_Map(object):


    def __init__(self):
        super(Hypbolic_Paraboloid_Map, self).__init__()
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

class Hyperboloid_Map(object):


    def __init__(self):
        super(Hyperboloid_Map, self).__init__()
        self.X = sqrt(1+u**2)*cos(v)
        self.Y = sqrt(1+u**2)*sin(v)
        self.Z = u
        self.c1 = np.linspace(-3,3,N)
        self.c2 = np.linspace(0,np.pi*2,N)
    def x(self,u,v):
         return np.sqrt(1+u**2)*np.cos(v)
    def y(self,u,v):
         return np.sqrt(1+u**2)*np.sin(v)
    def z(self,u,v):
         return u
    def boundary(self,condition):
        if condition[0][0] > 3 or condition[0][0] < -3: #or condition[0][1] > np.pi*2 or condition[0][1] < 0:
            return True
        else:
            return False


class Singularity_Map(object):


    def __init__(self):
        super(Singularity_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = -1/sqrt(u**2+v**2)
        self.c1 = np.linspace(-3,3,N)
        self.c2 = np.linspace(-3,3,N)
    def x(self,u,v):
         return u
    def y(self,u,v):
         return v
    def z(self,u,v):
         return -1/np.sqrt(u**2+v**2)
    def boundary(self,condition):
        if condition[0][0] > 3 or condition[0][0] < -3 or condition[0][1] > 3 or condition[0][1] < -3:
            return True
        else:
            return False

class Eliptical_Paraboloid_Map(object):


    def __init__(self):
        super(Eliptical_Paraboloid_Map, self).__init__()
        self.X = (u**2-4)*sin(v)
        self.Y = (u**2-4)*cos(v)
        self.Z = u
        self.c1 = np.linspace(-2,2,N)
        self.c2 = np.linspace(0,2*np.pi,N)
    def x(self,u,v):
         return  (u**2-4)*np.sin(v)
    def y(self,u,v):
         return (u**2-4)*np.cos(v)
    def z(self,u,v):
         return u
    def boundary(self,condition):
        if condition[0][0] > 2 or condition[0][0] < -2:
            return True
        else:
            return False


class Distortion_Map(object):

    def __init__(self):
        super(Distortion_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = -1/(exp((u)**2+(v)**2))
        self.c1 = np.linspace(-3,3,N)
        self.c2 = np.linspace(-3,3,N)
    def x(self,u,v):
         return u
    def y(self,u,v):
         return v
    def z(self,u,v):
         return -1/(np.exp((u)**2+(v)**2))
    def boundary(self,condition):
        if condition[0][0] > 3 or condition[0][0] < -3 or condition[0][1] > 3 or condition[0][1] < -3:
            return True
        else:
            return False


class Multi_Distortion_Map(object):
#this needs some performance improvements
    def __init__(self):
        super(Multi_Distortion_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = simplify(-1/(exp((u-2)**2+(v-2)**2))-1/(exp((u+2)**2+(v+2)**2))-1/(exp((u+2)**2+(v-2)**2))-1/(exp((u-2)**2+(v+2)**2)))
        self.c1 = np.linspace(-5,5,N)
        self.c2 = np.linspace(-5,5,N)
    def x(self,u,v):
         return u
    def y(self,u,v):
         return v
    def z(self,u,v):
        return -1/(np.exp((u-2)**2+(v-2)**2))-1/(np.exp((u+2)**2+(v+2)**2))-1/(np.exp((u+2)**2+(v-2)**2))-1/(np.exp((u-2)**2+(v+2)**2))
    def boundary(self,condition):
        if condition[0][0] > 5 or condition[0][0] < -5 or condition[0][1] > 5 or condition[0][1] < -5:
            return True
        else:
            return False

class Saddle_Map(object):
    def __init__(self):
        super(Saddle_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = u*v/50
        self.c1 = np.linspace(-25,25,N)
        self.c2 = np.linspace(-25,25,N)
    def x(self,u,v):
         return u
    def y(self,u,v):
         return v
    def z(self,u,v):
        return u*v/50
    def boundary(self,condition):
        if condition[0][0]**2 + condition[0][1]**2 > 10000:
            return True
        else:
            return False


class D1_Wave_Map(object):
    def __init__(self):
        super(D1_Wave_Map, self).__init__()
        self.X = u
        self.Y = v
        self.Z = sin(u)
        self.c1 = np.linspace(-5,5,N)
        self.c2 = np.linspace(-5,5,N)
    def x(self,u,v):
         return u
    def y(self,u,v):
         return v
    def z(self,u,v):
        return np.sin(u)
    def boundary(self,condition):
        if condition[0][0] > 5 or condition[0][0] < -5 or condition[0][1] > 5 or condition[0][1] < -5:
            return True
        else:
            return False


class Spherical_Wave_Map(object):
    def __init__(self):
        super(Spherical_Wave_Map, self).__init__()
        self.X = u*cos(v)
        self.Y = u*sin(v)
        self.Z = sin(u)
        self.c1 = np.linspace(0,10,N)
        self.c2 = np.linspace(0,np.pi*2,N)
    def x(self,u,v):
         return u*np.cos(v)
    def y(self,u,v):
         return u*np.sin(v)
    def z(self,u,v):
        return np.sin(u)
    def boundary(self,condition):
        if condition[0][0] > 10 or condition[0][0] <-0:
            return True
        else:
            return False

class Boys_Surface_Map(object):
    #this is never gonna find its own metric tensor
    def __init__(self):
        super(Boys_Surface_Map, self).__init__()
        w = u*exp(v*I)
        self.g1 = (-3/2)*im(w*(1-w**4)/(w**6+sqrt(5)*w**3-1))
        self.g2 = (-3/2)*re(w*(1+w**4)/(w**6+sqrt(5)*w**3-1))
        self.g3 = im((1+w**6)/(w**6+sqrt(5)*w**3-1))-(1/2)
        self.X = self.g1/(self.g1**2+self.g2**2+self.g3**2)
        self.Y = self.g2/(self.g1**2+self.g2**2+self.g3**2)
        self.Z = self.g3/(self.g1**2+self.g2**2+self.g3**2)
        self.c1 = np.linspace(0,1,N) # u is radius
        self.c2 = np.linspace(0,np.pi*2,N) # v is angle
    def x(self,u,v):
         w = u*np.exp(1j*v)
         g1 = (-3/2)*np.imag(w*(1-w**4)/(w**6+np.sqrt(5)*w**3-1))
         g2 = (-3/2)*np.real(w*(1+w**4)/(w**6+np.sqrt(5)*w**3-1))
         g3 = np.imag((1+w**6)/(w**6+np.sqrt(5)*w**3-1))-(1/2)
         return g1/(g1**2+g2**2+g3**2)
    def y(self,u,v):
          w = u*np.exp(1j*v)
          g1 = (-3/2)*np.imag(w*(1-w**4)/(w**6+np.sqrt(5)*w**3-1))
          g2 = (-3/2)*np.real(w*(1+w**4)/(w**6+np.sqrt(5)*w**3-1))
          g3 = np.imag((1+w**6)/(w**6+np.sqrt(5)*w**3-1))-(1/2)
          return g2/(g1**2+g2**2+g3**2)
    def z(self,u,v):
          w = u*np.exp(1j*v)
          g1 = (-3/2)*np.imag(w*(1-w**4)/(w**6+np.sqrt(5)*w**3-1))
          g2 = (-3/2)*np.real(w*(1+w**4)/(w**6+np.sqrt(5)*w**3-1))
          g3 = np.imag((1+w**6)/(w**6+np.sqrt(5)*w**3-1))-(1/2)
          return g3/(g1**2+g2**2+g3**2)
    def boundary(self,condition):
        if condition[0][0] > 1 or condition[0][0] <0:
            return True
        else:
            return False
