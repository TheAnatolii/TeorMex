import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Параметры системы
a = 1.0  # расстояние между центрами пружин D и E
l = 2.0  # длина стержня AB
g = 9.81  # ускорение свободного падения
m2 = 1  # масса груза
slider_height = 1.4
slider_y = -0.5  # высота ползуна
point_D = [a, 0]
point_O = [0, 0]
point_E = [-a, 0]
wall_D = [a+0.05, 0]
wall_E = [-a-0.05, 0]
fi0 = np.pi / 4  # начальный угол от вертикали

# Время
t_max = 5.0
dt = 0.08
t_values = np.arange(0, t_max, dt)

# Создаем GridSpec с 4 строками и одним столбцом
fig = plt.figure()
gs = GridSpec(5, 1, figure=fig)

# Создание анимации
ax = fig.add_subplot(gs[:5, :])
line, = ax.plot([], [], 'co-')  # Стержень AB
ball, = ax.plot([], [], 'bo', markersize=10)  # Груз B
slider, = ax.plot([], [], 'go', markersize=15, marker='s')  # Ползун A
point_d, = ax.plot([], [], 'co', markersize=5)  # Точка D
point_o, = ax.plot([], [], 'mo', markersize=5)  # Точка O
point_e, = ax.plot([], [], 'co', markersize=5)  # Точка E
wall_d, = ax.plot([], [], 'bo', markersize=8, marker='s')  # Стенка D
wall_e, = ax.plot([], [], 'bo', markersize=8, marker='s')  # Стенка E
spring_DA, = ax.plot([], [], 'g', lw=2)  # Линия, представляющая пружину DA
spring_EA, = ax.plot([], [], 'g', lw=2)  # Линия, представляющая пружину EA
ax.set_xlim(-10, 10)
ax.set_ylim(-3, 1)
ax.axvline(0, linestyle='--', color='k')  # пунктиры
ax.axhline(point_D[1], linestyle='--', color='k')

# Добавим подписи точек
ax.text(point_D[0], point_D[1], 'D', ha='right', va='bottom')
ax.text(point_O[0], point_O[1], 'O', ha='right', va='bottom')
ax.text(point_E[0], point_E[1], 'E', ha='right', va='bottom')

# Создадим текст для букв A и B
text_A = ax.text(0, slider_y, 'A', ha='right', va='bottom')
text_B = ax.text(l * np.sin(fi0), slider_y - l * np.cos(fi0), 'B', ha='right', va='bottom')

# Создадим зигзагообразные пружины для линий DA и EA
def create_zigzag(start, end, num_segments=15, amplitude=0.1):
    x_vals = np.linspace(start[0], end[0], num_segments)
    y_vals = np.linspace(start[1], end[1]-0.2, num_segments)
    dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    amp_factor = amplitude * (1 + 0.5 * dist)  # Изменение амплитуды пружины в зависимости от расстояния
    for i in range(1, num_segments, 2):
        y_vals[i] += amp_factor
    return x_vals, y_vals

spring_DA_x, spring_DA_y = create_zigzag(point_D, [0, slider_y])
spring_EA_x, spring_EA_y = create_zigzag(point_E, [0, slider_y])

# Создадим линии для зигзагообразных пружин
spring_DA.set_data(spring_DA_x, spring_DA_y)
spring_EA.set_data(spring_EA_x, spring_EA_y)


def animate(i):
    global slider_y
    t = i * dt

    # Изменяем вертикальное положение основания стержня
    slider_y = -l * np.cos(fi0 * np.cos(np.sqrt(g / l) * t)) + slider_height

    # Обновляем угол маятника используя уравнение маятника
    fi = fi0 * np.cos(np.sqrt(g / l) * t)

    x_values = [0, l * np.sin(fi)]
    y_values = [slider_y, slider_y - l * np.cos(fi)]
    line.set_data(x_values, y_values)

    # Обновляем положение груза B
    x_ball = l * np.sin(fi)
    y_ball = slider_y - l * np.cos(fi)
    ball.set_data(x_ball, y_ball)

    # Создадим линии для зигзагообразных пружин
    spring_DA_x, spring_DA_y = create_zigzag(point_D, [0, slider_y])
    spring_EA_x, spring_EA_y = create_zigzag(point_E, [0, slider_y])

    spring_DA.set_data(spring_DA_x, spring_DA_y)
    spring_EA.set_data(spring_EA_x, spring_EA_y)

    # Обновляем положение ползуна A
    slider.set_data(0, slider_y - 0.1)

    # Обновляем положения точек D, O и E
    point_d.set_data(point_D[0], point_D[1])
    point_o.set_data(point_O[0], point_O[1])
    point_e.set_data(point_E[0], point_E[1])

    # Обновляем координаты стенок
    wall_d.set_data(wall_D[0], wall_D[1])
    wall_e.set_data(wall_E[0], wall_E[1])

    # Обновляем координаты букв A и B
    text_A.set_position((0, slider_y))
    text_B.set_position((x_ball, y_ball))

    return line, ball, slider, spring_EA, spring_DA, point_d, point_o, point_e, text_A, text_B, wall_d, wall_e


ani = FuncAnimation(fig, animate, frames=len(t_values), blit=True, interval=100)
plt.show()