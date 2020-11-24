from sympy import *
from sympy.vector import CoordSys3D
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
import numpy as np
from D3Geometries import *


N = CoordSys3D("N")
t = symbols('t')
u = Function('u')(t)
v = Function('v')(t)

c = Array([u,v])

Geometery = Eliptical_Paraboloid_Map()
Visulisation = Eliptical_Paraboloid_Map()
X= Geometery.X
Y= Geometery.Y
Z= Geometery.Z

PR_PC = []
for coord in c:
    PR_PC.append(diff(X,coord)*N.i + diff(Y,coord)*N.j + diff(Z,coord)*N.k)
PR_PC = Array(PR_PC)


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

pprint(G,use_unicode=False)



#using metric tensor to find christoffel symbols

Gamma = MutableDenseNDimArray([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]])
for k in range(len(c)):
    for i in range(len(c)):
        for j in range(len(c)):
            E = 0
            for l in range(len(c)):
                E += 0.5*G_I.row(k)[l]*((G.row(l)[i]).diff(c[j])+(G.row(j)[l]).diff(c[i])-(G.row(i)[j]).diff(c[l]))
            Gamma[i,j,k] = E

print("\n")
Gamma = simplify(Gamma)
pprint(Gamma,use_unicode=False)

eqs = []
for k in range(len(c)):
    eq = c[k].diff(t).diff(t)
    for i in range(len(c)):
        for j in range(len(c)):
            eq += Gamma[i,j,k]*c[i].diff(t)*c[j].diff(t)
    eqs.append(simplify(Eq(eq,0)))


pprint(eqs,use_unicode=False)


second_Ds =[]
for i in range(len(c)):
    second_Ds.append(lambdify((u,v,u.diff(t),v.diff(t)),solve(eqs[i],c[i].diff(t,2))))


x = Visulisation.x
y = Visulisation.y
z = Visulisation.z
boundary = Visulisation.boundary

u = Visulisation.c1
v = Visulisation.c2
u,v = np.meshgrid(u,v)


Initial_Conditions = [[np.array([0.2,0.3]),np.array([0.2,0.1])],[np.array([1.0,0.5]),np.array([-0.85,-0.5])]] #(R,Rp) #maybe normalise the rp vectors
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
        if boundary(condition):
            break
    Path = np.array(Path)
    plt.plot(Path.T[0],Path.T[1])
    D3Paths.append([x(Path.T[0],Path.T[1]),y(Path.T[0],Path.T[1]),z(Path.T[0],Path.T[1])])
    ax.set_aspect('equal')

plt.show()


fig,axes = plt.subplots(subplot_kw=dict(projection='3d'))
ax = axes
ax.plot_wireframe(x(u,v), y(u,v), z(u,v),color=(0,1,0,0.15))
for Path in D3Paths:
    ax.plot(Path[0],Path[1],Path[2])

max_range = np.array([x(u,v).max()-x(u,v).min(), y(u,v).max()-y(u,v).min(), z(u,v).max()-z(u,v).min()]).max() / 2.0

mid_x = (x(u,v).max()+x(u,v).min()) * 0.5
mid_y = (y(u,v).max()+y(u,v).min()) * 0.5
mid_z = (z(u,v).max()+z(u,v).min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)



plt.show()
