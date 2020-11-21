from sympy import *
from sympy.vector import CoordSys3D
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
import numpy as np


N = CoordSys3D("N")
t = symbols('t')
u = Function('u')(t)
v = Function('v')(t)

c = Array([u,v])

# Map to Space

#these work best when Z = f(x,y)

# #sphere
# X = cos(v)*sin(u)
# Y = sin(v)*sin(u)
# Z = cos(u)




# #plane
# X = u
# Y = v
# Z = 0

# # exotic surface
# X = u*cos(v)
# Y = u*sin(v)
# Z = sin(u)*sin(v)

# #hump looking thing
# X = u*sin(v)
# Y = u*cos(v)
# Z = exp(-u)

# #Torus
#
# X = (2+1*cos(u))*cos(v)
# Y = (2+1*cos(u))*sin(v)
# Z = 1*sin(u)


#saddle
X = u
Y = v
Z = (u*u -v*v)/50



PR_PC = []
for coord in c:
    PR_PC.append(diff(X,coord)*N.i + diff(Y,coord)*N.j + diff(Z,coord)*N.k)
PR_PC = Array(PR_PC)


PR_PC2 = []
for c1 in c:
    R = []
    for c2 in c:
        R.append(simplify((diff(X,c1)*N.i + diff(Y,c1)*N.j + diff(Z,c1)*N.k).diff(c2)))
    PR_PC2.append(R)
PR_PC2 = Array(PR_PC2)


#calculating metrix tensor
G = []
for c1 in PR_PC:
    R = []
    for c2 in PR_PC:
        R.append(c1.dot(c2))
    G.append(R)
G = Matrix(G)
G = simplify(G)
G_I = G.inv()



#velcocity_acceleration dots (PR/PU dot PR2/PU2 ect.)    [c1,c2] c1 = [[]]
V_Adots = []
for c1 in PR_PC:
    A1 = []
    for c2 in PR_PC2:
        A2 = []
        for c3 in c2:
            A2.append(c1.dot(c3))
        A1.append(A2)
    V_Adots.append(A1)
V_Adots = simplify(Array(V_Adots))



Gamma = MutableDenseNDimArray([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]])

for k in range(len(c)):
    for i in range(len(c)):
        for j in range(len(c)):
            E = 0
            for c1 in range(len(c)):
                E += V_Adots[c1,i,j]*G_I.col(c1)[k]
            Gamma[i,j,k] = E
eqs = []
for k in range(len(c)):
    eq = c[k].diff(t).diff(t)
    for i in range(len(c)):
        for j in range(len(c)):
            eq += Gamma[i,j,k]*c[i].diff(t)*c[j].diff(t)
    eqs.append(simplify(Eq(eq,0)))
# print(dsolve(eqs, c))

x = Function('x')(t)
y = Function('y')(t)


pprint(eqs,use_unicode=False)


second_Ds =[]
for i in range(len(c)):
    second_Ds.append(lambdify((u,v,u.diff(t),v.diff(t)),solve(eqs[i],c[i].diff(t,2))))





#visulisation for a sphere
def x(u,v):
     return np.cos(v)*np.sin(u)
def y(u,v):
    return np.sin(v)*np.sin(u)
def z(u,v):
    return np.cos(u)

#hump looking thing
# def x(u,v):
#     return u*np.sin(v)
# def y(u,v):
#     return u*np.cos(v)
# def z(u,v):
#     return np.exp(-u)

# #exotic surface
# def x(u,v):
#     return u*np.cos(v)
# def y(u,v):
#     return u*np.sin(v)
# def z(u,v):
#     return np.sin(u)*np.sin(v)

# #Torus
#
# def x(u,v):
#      return (2+1*np.cos(u))*np.cos(v)
# def y(u,v):
#      return (2+1*np.cos(u))*np.sin(v)
# def z(u,v):
#      return 1*np.sin(u)

#saddle

def x(u,v):
    return u
def y(u,v):
    return v
def z(u,v):
    return (u*u -v*v)/50


N=50

u = np.linspace(-10, 10, N)
v = np.linspace(-10, 10, N)
u,v = np.meshgrid(u,v)


Initial_Conditions = [[np.array([np.pi/2,np.pi/2]),np.array([0.1,0.3])],[np.array([0.1,0.2]),np.array([-0.4,0.2])]] #(R,Rp) #maybe normalise the rp vectors
D3Paths = []

#diff eq solver
fig = plt.figure()
ax = fig.add_subplot(111)
for condition in Initial_Conditions:
    Path = []
    delta = 0.01
    for i in range(5000):
        # print(R)
        Path.append(condition[0].copy())
        for c in range(len(second_Ds)):
            condition[1][c] += delta*second_Ds[c](condition[0][0],condition[0][1],condition[1][0],condition[1][1])[0]
            # print(Rp)
        condition[0] += condition[1]*delta
    Path = np.array(Path)
    plt.plot(Path.T[0],Path.T[1])
    # plt.polar(Path.T[1],Path.T[0])
    D3Paths.append([x(Path.T[0],Path.T[1]),y(Path.T[0],Path.T[1]),z(Path.T[0],Path.T[1])])
    ax.set_aspect('equal')
plt.show()


fig,axes = plt.subplots(subplot_kw=dict(projection='3d'))
ax = axes
ax.plot_wireframe(x(u,v), y(u,v), z(u,v),color=(0,1,0,0.15))
for Path in D3Paths:
    ax.plot(Path[0],Path[1],Path[2])


plt.show()
