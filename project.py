import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ===== Mô hình vật lý =====
def equation(y, t, g, l, k):
    theta, omega = y
    return np.array([omega, - (g / l) * np.sin(theta) - k * omega])

def runge_kutta_4(fun, y0, time, h, g, l, k):
    n = len(time)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        t = time[i]
        k1 = fun(y[i], t, g, l, k)
        k2 = fun(y[i] + 0.5 * h * k1, t + 0.5 * h, g, l, k)
        k3 = fun(y[i] + 0.5 * h * k2, t + 0.5 * h, g, l, k)
        k4 = fun(y[i] + h * k3, t + h, g, l, k)
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

# ===== Cập nhật animation =====
def update(frame):
    if frame >= len(time):
        return line, bob, time_label, point_angle, point_velocity

    theta = sol[frame, 0]
    omega = sol[frame, 1]
    t = time[frame]

    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    line.set_data([0, x], [0, y])
    bob.set_offsets([x, y])
    time_label.set_text(f"Thời gian: {t:.2f} s")

    point_angle.set_offsets([t, theta])
    point_velocity.set_offsets([t, omega])

    epsilon = 0.01
    stop_window = 20
    if frame > stop_window:
        recent = sol[frame-stop_window:frame]
        if np.all(np.abs(recent[:, 0]) < epsilon) and np.all(np.abs(recent[:, 1]) < epsilon):
            ani.event_source.stop()

    return line, bob, time_label, point_angle, point_velocity

# ===== Bắt đầu mô phỏng lại =====
def simulate(event=None):
    global sol, length, time, ani
    global line, bob, time_label, point_angle, point_velocity

    if ani and ani.event_source:
        ani.event_source.stop()
        ani._fig = None
        del ani

    try:
        theta0 = float(textbox_theta.text)
        omega0 = float(textbox_omega.text)
        length = float(textbox_length.text)
        damping = float(textbox_damping.text)
    except ValueError:
        print("Nhập sai định dạng dữ liệu")
        return

    h = 0.025
    T = 60
    time = np.arange(0, T, h)
    y0 = [np.radians(theta0), omega0]
    g = 9.81

    sol = runge_kutta_4(equation, y0, time, h, g, length, damping)

    ax1.clear()
    ax2.clear()
    ax_pendulum.clear()

    ax1.plot(time, sol[:, 0], label="Góc lệch")
    ax1.set_title("Góc lệch theo thời gian")
    ax1.set_xlabel("Thời gian (s)")
    ax1.set_ylabel("Góc (rad)")
    ax1.grid(True)
    ax1.legend(loc='upper right')

    ax2.plot(time, sol[:, 1], label="Vận tốc góc", color="orange")
    ax2.set_title("Vận tốc góc theo thời gian")
    ax2.set_xlabel("Thời gian (s)")
    ax2.set_ylabel("Vận tốc (rad/s)")
    ax2.grid(True)
    ax2.legend(loc='upper right')

    ax_pendulum.set_xlim(-2*length, 2*length)
    ax_pendulum.set_ylim(-2*length, 2*length)
    ax_pendulum.axhline(0, color='green', linestyle='--')
    ax_pendulum.axvline(0, color='green', linestyle='--')

    line, = ax_pendulum.plot([], [], 'k-', lw=2)
    bob = ax_pendulum.scatter([], [], color='red', s=80)
    time_label = ax_pendulum.text(0.05, 1.8, '', transform=ax_pendulum.transData, fontsize=10)

    point_angle = ax1.scatter([], [], color='red', zorder=5)
    point_velocity = ax2.scatter([], [], color='red', zorder=5)

    ani = FuncAnimation(fig, update, frames=range(len(time)), interval=20, blit=False, repeat=False)
    plt.draw()

# ===== Dừng và tiếp tục =====
def pause_animation(event):
    if ani and ani.event_source:
        ani.event_source.stop()

def resume_animation(event):
    if ani and ani.event_source:
        ani.event_source.start()

# ===== Giao diện ban đầu =====
fig = plt.figure("Con lắc đơn - Runge-Kutta 4", figsize=(12, 7))
gs = GridSpec(7, 5, figure=fig, width_ratios=[1.2, 0.1, 0.1, 0.1, 1.2], height_ratios=[0.05, 1, 0.1, 1, 0.2, 0.1, 0.3])

ax1 = fig.add_subplot(gs[1, 0:2])
ax2 = fig.add_subplot(gs[3, 0:2])
ax_pendulum = fig.add_subplot(gs[0:5, 4])

# TextBoxes và Buttons
axbox_theta = plt.axes([0.05, 0.04, 0.08, 0.05])
textbox_theta = TextBox(axbox_theta, '', initial="45")
fig.text(0.05, 0.10, 'Góc (°)', ha='left', va='bottom', fontsize=9)

axbox_omega = plt.axes([0.16, 0.04, 0.08, 0.05])
textbox_omega = TextBox(axbox_omega, '', initial="0")
fig.text(0.16, 0.10, 'Vận tốc', ha='left', va='bottom', fontsize=9)

axbox_length = plt.axes([0.27, 0.04, 0.08, 0.05])
textbox_length = TextBox(axbox_length, '', initial="1")
fig.text(0.27, 0.10, 'Chiều dài', ha='left', va='bottom', fontsize=9)

axbox_damping = plt.axes([0.38, 0.04, 0.08, 0.05])
textbox_damping = TextBox(axbox_damping, '', initial="0.1")
fig.text(0.38, 0.10, 'Ma sát', ha='left', va='bottom', fontsize=9)

ax_button_start = plt.axes([0.52, 0.04, 0.12, 0.05])
button_start = Button(ax_button_start, 'Bắt đầu mô phỏng')
button_start.on_clicked(simulate)

ax_button_pause = plt.axes([0.66, 0.04, 0.08, 0.05])
button_pause = Button(ax_button_pause, 'Dừng')
button_pause.on_clicked(pause_animation)

ax_button_resume = plt.axes([0.75, 0.04, 0.08, 0.05])
button_resume = Button(ax_button_resume, 'Tiếp tục')
button_resume.on_clicked(resume_animation)

line, = ax_pendulum.plot([], [], 'k-', lw=2)
bob = ax_pendulum.scatter([], [], color='red', s=80)
time_label = ax_pendulum.text(0.05, 1.8, '', transform=ax_pendulum.transData, fontsize=10)
point_angle = ax1.scatter([], [], color='red')
point_velocity = ax2.scatter([], [], color='red')
sol = np.zeros((1, 2))
length = 1
time = np.linspace(0, 1, 2)
ani = None

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
