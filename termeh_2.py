import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры системы
a = 1.0  # расстояние между центрами пружин D и E
l = 2.0  # длина стержня AB
g = 9.81  # ускорение свободного падения
slider_height = 2.5  # высота ползуна
slider_y = -l + slider_height
point_D = [a, -l + slider_height + 1]
point_O = [0, -l + slider_height + 1]
point_E = [-a, -l + slider_height + 1]
fi0 = np.pi / 4  # начальный угол от вертикали

# Время
t_max = 10.0
dt = 0.01
t_values = np.arange(0, t_max, dt)

# Создание анимации
fig, ax = plt.subplots()
line, = ax.plot([], [], 'co-')  # Стержень AB
ball, = ax.plot([], [], 'bo')  # Груз B
slider, = ax.plot([], [], 'go')  # Ползун A
spring_DA, = ax.plot([], [], 'g', lw=2)  # Линия, представляющая пружину DA
spring_EA, = ax.plot([], [], 'g', lw=2)  # Линия, представляющая пружину EA
point_d, = ax.plot([], [], 'co', markersize=5)  # Точка D
point_o, = ax.plot([], [], 'mo', markersize=5)  # Точка O
point_e, = ax.plot([], [], 'co', markersize=5)  # Точка E
ax.set_xlim(-l, l)
ax.set_ylim(-l, l)
spring_DA_x = [point_D[0], 0]  # Координаты пружины DA
spring_DA_y = [point_D[1], slider_y]
spring_EA_x = [point_E[0], 0]  # Координаты пружины EA
spring_EA_y = [point_E[1], slider_y]
ax.axvline(0, linestyle='--', color='k')  # пунктиры
ax.axhline(point_D[1], linestyle='--', color='k')

# Добавим подписи точек
ax.text(point_D[0], point_D[1], 'D', ha='right', va='bottom')
ax.text(point_O[0], point_O[1], 'O', ha='right', va='bottom')
ax.text(point_E[0], point_E[1], 'E', ha='right', va='bottom')

# Создадим текст для букв A и B
text_A = ax.text(0, slider_y, 'A', ha='right', va='bottom')
text_B = ax.text(l * np.sin(fi0), slider_y - l * np.cos(fi0), 'B', ha='right', va='bottom')

def animate(i):
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

    # Обновляем положение ползуна A
    slider.set_data(0, slider_y)

    # Обновляем положения точек D, O и E
    point_d.set_data(point_D[0], point_D[1])
    point_o.set_data(point_O[0], point_O[1])
    point_e.set_data(point_E[0], point_E[1])

    # Обновляем координаты пружин
    spring_DA_x = [point_D[0], 0]
    spring_DA_y = [point_D[1], slider_y]
    spring_DA.set_data(spring_DA_x, spring_DA_y)

    spring_EA_x = [point_E[0], 0]
    spring_EA_y = [point_E[1], slider_y]
    spring_EA.set_data(spring_EA_x, spring_EA_y)

    # Обновляем координаты букв A и B
    text_A.set_position((0, slider_y))
    text_B.set_position((x_ball, y_ball))

    return line, ball, slider, spring_EA, spring_DA, point_d, point_o, point_e, text_A, text_B


ani = FuncAnimation(fig, animate, frames=len(t_values), blit=True, interval=15)
plt.show()
