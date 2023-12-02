import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

R = 4
Omega = 1
t = sp.Symbol('t')

r = 2 + sp.sin(8 * t)
fi = t + 0.5 * sp.sin(4 * t)

x = r * sp.cos(fi)
y = r * sp.sin(fi)

xC = R*Omega*t

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)

a_x = sp.diff(Vx, t)
a_y = sp.diff(Vy, t)


T = np.linspace(0, 10, 1000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
XC = np.zeros_like(T)
YC = R
VX = np.zeros_like(T)
VY = np.zeros_like(T)
A_x = np.zeros_like(T)
A_y = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    XC[i] = sp.Subs(xC, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    A_x[i] = sp.Subs(a_x, t, T[i])
    A_y[i] = sp.Subs(a_y, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-5, 5], ylim=[-5, 5])

ax1.plot(X, Y)
ax1.plot([X.min(), X.max()], [0, 0], 'black')

Phi = np.linspace(0, 2*math.pi, 100)
#Circ, = ax1.plot(XC[0]+R*np.cos(Phi), YC+R*np.sin(Phi), 'g')

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], (X[0]+ VX[0])], [Y[0], (Y[0]+VY[0])], 'r')
ALine, = ax1.plot([X[0], (X[0]+A_x[0])], [Y[0], (Y[0]+A_y[0])], 'blue')
RLine, = ax1.plot([0, X[0]], [0, Y[0]], 'g')

ArrowX = np.array([-0.1*R, 0, -0.1*R])
ArrowY = np.array([0.1*R, 0, -0.1*R])

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX+X[0]+VX[0], RArrowY+Y[0]+VY[0], 'r')

RVecArrowX_A, RVecArrowY_A = Rot2D(ArrowX, ArrowY, math.atan2(A_y[0], A_x[0]))
A_Arrow, = ax1.plot(RVecArrowX_A+X[0]+A_x[0], RVecArrowY_A+Y[0] + A_y[0], 'blue')

RVecArrowX_R, RVecArrowY_R = Rot2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
R_Arrow, = ax1.plot(RVecArrowX_R + VX[0] + X[0], RVecArrowY_R + VY[0] + Y[0], 'g')

def anima(i):
    P.set_data(X[i], Y[i])

    VLine.set_data([X[i], (X[i]+VX[i])], [Y[i], (Y[i]+VY[i])])
    ALine.set_data([X[i], (X[i] + A_x[i])], [Y[i], (Y[i] + A_y[i])])
    RLine.set_data([0, X[i]], [0, Y[i]])

    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX+(X[i]+VX[i]), RArrowY+(Y[i]+VY[i]))

    RArrowX_A, RArrowY_A = Rot2D(ArrowX, ArrowY, math.atan2(A_y[i], A_x[i]))
    A_Arrow.set_data(RArrowX_A+(X[i]+A_x[i]), RArrowY_A+(Y[i]+A_y[i]))

    RVecArrowX_R, RVecArrowY_R = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    R_Arrow.set_data(RVecArrowX_R + X[i], RVecArrowY_R + Y[i])
    return P, VLine, ALine, RLine, VArrow, A_Arrow, R_Arrow

anim = FuncAnimation(fig, anima, frames=1000, interval=50, repeat=False)

plt.show()

