import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import math
from scipy.integrate import odeint


def formY(y, t, fv, fw):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fv(y1, y2, y3, y4), fw(y1, y2, y3, y4)]
    return dydt


Frames = 500
Interval_Frame = 0
Repeat_Delay_Anim = 0
t = sp.Symbol("t")  # t is a symbol variable
x = sp.Function('x')(t)  # x(t)
fi = sp.Function('fi')(t)  # fi(t)
v = sp.Function('v')(t)  # dx/dt, v(t)
w = sp.Function('w')(t)  # dfi/dt, w(t), omega(t)
a_full = sp.Function('a_full')(t)  # dx/dt, v(t)
e = sp.Function('e')(t)  # dfi/dt, w(t), omega(t)


# Initializing
width = 2  # width of the rectangle
length = 5  # length of the rectangle
a = 5  # distance between spring and rectanlge (DO or OE)
# circle_radius = 0.2  # radius of the circle
# m1 = 1  # mass of the rectangle
# m2 = 1  # mass of the circle
# g = 9.8  # const
# l = 2  # length of stick
# k = 5  # spring stiffness coefficient
# y0 = [0, sp.rad(45), 0, 0]  # x(0), fi(0), v(0), w(0)


# circle_radius = 0.2  # radius of the circle
# m1 = 1  # mass of the rectangle
# m2 = 1  # mass of the circle
# g = 9.8  # const
# l = 2  # length of stick
# k = 51  # spring stiffness coefficient
# y0 = [0, sp.rad(45), 0, 0]  # x(0), fi(0), v(0), w(0)


circle_radius = 0.8
m1 = 2  # mass of the rectangle
m2 = 2  # mass of the circle
g = 9.8  # const
l = 5  # length of stick
k = 100  # spring stiffness coefficient
y0 = [0, sp.rad(180), 0, 0]


# circle_radius = 0.8
# m1 = 0.1  # mass of the rectangle
# m2 = 0.1  # mass of the circle
# g = 9.8  # const
# l = 2  # length of stick
# k = 20  # spring stiffness coefficient
# y0 = [0, sp.rad(180), 0, 0]


# Caluclating Lagrange equations
# Kinetic energy of the rectangle
# Ekin1 = (m1 * v * v) / 2
# # Squared velocity of the circle's center of mass
# Vsquared = v * v + w * w * l * l - 2 * v * w * l * sp.sin(fi)
# # Kinetic energy of the circle
# Ekin2 = m2 * Vsquared / 2
# # Kinetic energy of system
# Ekin = Ekin1 + Ekin2
# # Potential energy
# Spring_delta_x = sp.sqrt(a * a + x * x) - a  # delta_x^2
# # We have two springs so Esprings = 2 * (k*delta_x^2/2) = k*delta_x^2
# Esprings = k * Spring_delta_x * Spring_delta_x
# Epot = - m1 * g * x - m2 * g * (x + l * sp.cos(fi)) + Esprings
# # generalized forces
# Qx = -sp.diff(Epot, x)
# Qfi = -sp.diff(Epot, fi)
#
# # Lagrange function
# Lagr = Ekin - Epot
# ur1 = sp.diff(sp.diff(Lagr, v), t) - sp.diff(Lagr, x)
# ur2 = sp.diff(sp.diff(Lagr, w), t) - sp.diff(Lagr, fi)
# print(ur1)
# print()
# print(ur2 / (l * m2))

a11 = m1 + m2
a12 = -m2 * l * sp.sin(fi)
a21 = sp.sin(fi)
a22 = -l
b1 = -2*k * x * (1 - (a*a / sp.sqrt(x*x + a*a))) + (m1+m2)*g - w*w*sp.cos(fi) * m2 * l
b2 = g * sp.sin(fi)

detA = a11 * a22 - a12 * a21
detA1 = b1 * a22 - b2 * a21
detA2 = a11 * b2 - b1 * a21
dvdt = detA1 / detA
dwdt = detA2 / detA
# constructing the system of differential equations
T = np.linspace(0, 50, Frames)
# lambdify translates function from sympy to numpy and then form arrays faster then by using subs
fv = sp.lambdify([x, fi, v, w], dvdt, "numpy")
fw = sp.lambdify([x, fi, v, w], dwdt, "numpy")
sol = odeint(formY, y0, T, args=(fv, fw))
# sol - our solution
# sol[:,0] - x
# sol[:,1] - fi
# sol[:,2] - v (dx/dt)
# sol[:,3] - w (dfi/dt)

fdv = sp.lambdify([v, w, a_full, e], dvdt, "numpy")
fdw = sp.lambdify([v, w, a_full, e], dwdt, "numpy")
sol_ra = odeint(formY, y0, T, args=(fv, fw))
# sol_ra - our solution
# sol_ra[:,0] - v
# sol_ra[:,1] - w
# sol_ra[:,2] - a_full
# sol_ra[:,3] - e

ra = [0] * len(sol[:, 0])
for i in range(len(ra)):
    ra[i] = m2 * l * (sol_ra[:, 3][i] * np.cos(sol[:, 1][i]) - sol_ra[:, 1][i] ** 2 * np.sin(sol[:, 1][i]))


# point A (center of the rectangle):
ax = sp.lambdify(x, 0)
ay = sp.lambdify(x, x)
AX = ax(sol[:, 0])
AY = -ay(sol[:, 0])

# point B (center of the circle):
bx = sp.lambdify(fi, l * sp.sin(fi))
by = sp.lambdify([x, fi], + l * sp.cos(fi) + x)
BX = bx(sol[:, 1])
BY = -by(sol[:, 0], sol[:, 1])

# start plotting
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax0.axis("equal")
# ax0.set_xlim(-10, 10)
# ax0.set_ylim(-38, 8.7)

# constant arrays
L1X = [-width / 2, -width / 2]
L2X = [width / 2, width / 2]
LY = [min(AY) - length, max(AY) + length]

# plotting environment
ax0.plot(0, 0, marker=".", color="red")  # красная точка
ax0.plot(L1X, LY, color="grey")  # left wall
ax0.plot(L2X, LY, color="grey")  # right wall
sl, = ax0.plot([-a, -length / 2], [0, AY[0] + width / 2],
               color="brown")  # left spring (rope)
sr, = ax0.plot([a, length / 2], [0, AY[0] + width / 2],
               color="brown")  # right spring (rope)
ax0.plot(-a, 0, marker=".", color="black")  # left joint
ax0.plot(a, 0, marker=".", color="black")  # right joint
ax0.axvline(0, linestyle='--', color='k')  # пунктиры
ax0.axhline(0, linestyle='--', color='k')
rect = plt.Rectangle((-width / 2, AY[0]), width,
                     length, color="black")  # rectangle
circ = plt.Circle((BX[0], BY[0]), circle_radius, color="grey")  # circle
# plotting radius vector of B
R_vector, = ax0.plot([0, BX[0]], [0, BY[0]], color="grey")
# adding statistics
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, sol[:, 0])
ax2.set_xlabel('t')
ax2.set_ylabel('x')

ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, sol[:, 1])
ax3.set_xlabel('t')
ax3.set_ylabel('fi')

ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(T, ra)
ax4.set_xlabel('t')
ax4.set_ylabel('R_a')

plt.subplots_adjust(wspace=0.3, hspace=0.7)

# Добавляем подписи к точкам, грузику и ползунку
ax0.text(-a - 1, 0, 'D', ha='right', va='bottom')
ax0.text(a + 1, 0, 'E', ha='left', va='bottom')
ax0.text(0, 0, 'O', ha='right', va='bottom')
text_A = ax0.text(0, AY[0], 'A', ha='right', va='bottom')
text_B = ax0.text(BX[0], BY[0], 'B', ha='right', va='bottom')


# function for initializing the positions
def init():
    rect.set_y(-length / 2)
    ax0.add_patch(rect)
    circ.center = (0, 0)
    ax0.add_patch(circ)
    return rect, circ

def create_zigzag(start, end, num_segments=12, amplitude=0.1):
    x_vals = np.linspace(start[0], end[0], num_segments)
    y_vals = np.linspace(start[1], end[1], num_segments)
    dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    amp_factor = amplitude * (2 + 0.5 * dist)  # Изменение амплитуды пружины в зависимости от расстояния
    for i in range(1, num_segments, 2):
        y_vals[i] += amp_factor
    return x_vals, y_vals

# function for recounting the positions
def anima(i):
    rect.set_y(AY[i] - length / 2)
    spring_left_x, spring_left_y = create_zigzag((-a, 0), (-width / 2, AY[i]), num_segments=12, amplitude=0.1)
    spring_right_x, spring_right_y = create_zigzag((a, 0), (width / 2, AY[i]), num_segments=12, amplitude=0.1)
    sl.set_data(spring_left_x, spring_left_y)
    sr.set_data(spring_right_x, spring_right_y)
    R_vector.set_data([0, BX[i]], [AY[i], BY[i]])
    circ.center = (BX[i], BY[i])

    # Обновляем положения текстовых меток для ползунка и грузика
    text_A.set_position((-1, AY[i]))  # Ползунок A
    text_B.set_position((BX[i], BY[i] + 0.5))  # Грузик B

    # Создание вертикальных стенок
    left_wall_x = [-a-0.4, -a-0.4]
    left_wall_y = [-1, 1]
    right_wall_x = [a+0.5, a+0.5]
    right_wall_y = [-1, 1]

    left_wall, = ax0.plot(left_wall_x, left_wall_y, color='grey')
    right_wall, = ax0.plot(right_wall_x, right_wall_y, color='grey')

    return sl, sr, rect, left_wall, right_wall, R_vector, circ,


# animating function
anim = FuncAnimation(fig, anima, init_func=init, frames=Frames, #interval=40,
                     blit=False, repeat=True, repeat_delay=Repeat_Delay_Anim)
plt.show()
