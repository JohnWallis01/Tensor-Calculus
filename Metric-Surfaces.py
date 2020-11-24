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
x = Function("x")(u,v)
y = Function("y")(u,v)
z = Function("z")(u,v)
s = Function("s")(x,y,z)

c = Array([u,v])


#self defined metric
#-------------------
G = [[1,0],[0,sin(u)**2]]
G = Matrix(G)
G = simplify(G)
G_I = G.inv()
pprint(G,use_unicode=False)
#-----------------


#g_ij = d_i(s).d_j(s)







#
Equations = []
for i in range(len(c)):
    for j in range(len(c)):
        Equations.append(Eq((simplify(diff(x,c[i])*N.i + diff(y,c[i])*N.j + diff(z,c[i])*N.k).dot(diff(x,c[j])*N.i + diff(y,c[j])*N.j + diff(z,c[j])*N.k)),G.row(i)[j]))
for Equation in Equations:
    pprint(Equation,use_unicode=False)
    print("\n")


def Format(Equations):
    for Equation in (Equations[0:2]+[Equations[3]]):
        print("\\begin{equation}")
        print("\\begin{split}")
        print(latex(Equation))
        print("\\end{split}")
        print("\\end{equation}")

#https://latexbase.com
Format(Equations)
