import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox, Button, Slider
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
time_started = None

# ===== Thiết lập môi trường =====
theta0 = 45
omega0 = 0
length = 1
damping = 0.1
speed_factor = 1.0

# ===== Mô hình vật lý =====
def equation(y, t, g, l, k):
    theta, omega = y
    # Phương trình chính xác hơn cho con lắc đơn
    return np.array([omega, - (g / l) * np.sin(theta) - k * omega / l])

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
    global time_started
    import time

    t_real = time.time() - time_started
    t_sim = t_real * speed_factor
    t_sim = min(t_sim, time_array[-1])  # Giới hạn trong dữ liệu

    # Tìm frame tương ứng với thời gian mô phỏng
    frame_sim = np.searchsorted(time_array, t_sim)
    frame_sim = min(frame_sim, len(time_array) - 1)

    theta = sol[frame_sim, 0]
    omega = sol[frame_sim, 1]

    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    line.set_data([0, x], [0, y])
    bob.set_center([x, y])

    time_label.set_text(f"Mô phỏng: {t_sim:.2f}s  |  Thực tế: {t_real:.2f}s")

    point_angle.set_offsets([t_sim, theta])
    point_velocity.set_offsets([t_sim, omega])

    if frame_sim > 0:
        trail_length = min(100, frame_sim)
        trail_x = [length * np.sin(sol[max(0, frame_sim - i), 0]) for i in range(trail_length)]
        trail_y = [-length * np.cos(sol[max(0, frame_sim - i), 0]) for i in range(trail_length)]
        trail.set_data(trail_x, trail_y)

    return line, bob, time_label, point_angle, point_velocity, trail



# ===== Kiểm tra đầu vào là số hợp lệ =====
def validate_numeric_input(text_str):
    """Kiểm tra xem chuỗi có phải là số thực hợp lệ không"""
    try:
        float(text_str)
        return True
    except ValueError:
        return False

# ===== Xử lý sự kiện khi người dùng nhập dữ liệu =====
def on_text_changed(text):
    # Kiểm tra text và thay thế nếu không phải là số
    if not validate_numeric_input(text):
        return False  # Từ chối thay đổi nếu không phải số
    return True

# ===== Bắt đầu mô phỏng lại =====
def simulate(event=None):
    global sol, length, time_array, ani
    global line, bob, time_label, point_angle, point_velocity, trail
    global theta0, omega0, damping, speed_factor
    global time_started  # ⬅️ Dùng để đo thời gian thực tế

    if ani is not None and ani.event_source:
        ani.event_source.stop()
        ani._fig = None
        del ani
        ani = None

    # Lấy giá trị đầu vào
    try:
        theta0 = float(textbox_theta.text)
        omega0 = float(textbox_omega.text)
        length = float(textbox_length.text)
        damping = float(textbox_damping.text)
    except ValueError:
        error_text = fig.text(0.5, 0.15, "Lỗi: đầu vào không hợp lệ", color='red', ha='center', fontsize=12)
        plt.draw()
        plt.pause(2)
        error_text.remove()
        return

    # === Tính toán mô phỏng ===
    h = 0.01
    T = 60
    time_array = np.arange(0, T, h)
    y0 = [np.radians(theta0), omega0]
    g = 9.81
    sol = runge_kutta_4(equation, y0, time_array, h, g, length, damping)

    # Bắt đầu đếm thời gian thực tế
    import time
    time_started = time.time()

    # === Vẽ lại giao diện ===
    ax1.clear()
    ax2.clear()
    ax_pendulum.clear()

    ax1.plot(time_array, sol[:, 0], label="Góc lệch", linewidth=1.5, color='blue')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax1.set_title("Góc lệch theo thời gian")
    ax1.set_xlabel("Thời gian (s)")
    ax1.set_ylabel("Góc (rad)")
    ax1.set_xlim(0, T)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(time_array, sol[:, 1], label="Vận tốc góc", linewidth=1.5, color='orange')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax2.set_title("Vận tốc góc theo thời gian")
    ax2.set_xlabel("Thời gian (s)")
    ax2.set_ylabel("Vận tốc (rad/s)")
    ax2.set_xlim(0, T)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    ax_pendulum.set_xlim(-1.5*length, 1.5*length)
    ax_pendulum.set_ylim(-1.5*length, 0.5*length)
    ax_pendulum.axhline(0, color='green', linestyle='--', alpha=0.5)
    ax_pendulum.axvline(0, color='green', linestyle='--', alpha=0.5)
    ax_pendulum.set_title("Mô phỏng con lắc")
    ax_pendulum.set_aspect('equal')
    ax_pendulum.grid(True, alpha=0.3)
    ax_pendulum.scatter([0], [0], color='black', s=30, zorder=5)

    line, = ax_pendulum.plot([], [], 'k-', lw=2)
    bob = Circle((0, 0), radius=length * 0.1, fc='red', ec='darkred', zorder=4)
    ax_pendulum.add_patch(bob)
    trail, = ax_pendulum.plot([], [], 'r-', lw=1, alpha=0.5)
    time_label = ax_pendulum.text(0.05, 0.1, '', transform=ax_pendulum.transAxes,
                                  fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    point_angle = ax1.scatter([], [], color='red', s=30, zorder=5)
    point_velocity = ax2.scatter([], [], color='red', s=30, zorder=5)

    ani = FuncAnimation(fig, update, frames=range(len(time_array)),
                        interval=10 / speed_factor, blit=True, repeat=False)
    plt.draw()

# ===== Dừng và tiếp tục animation =====
def pause_animation(event):
    if ani and ani.event_source:
        ani.event_source.stop()

def resume_animation(event):
    if ani and ani.event_source:
        ani.event_source.start()

def reset_view(event):
    try:
        l = float(textbox_length.text)
        ax_pendulum.set_xlim(-1.5*l, 1.5*l)
        ax_pendulum.set_ylim(-1.5*l, 0.5*l)
        plt.draw()
    except ValueError:
        pass

# ===== Thay đổi tốc độ mô phỏng =====
def update_speed(val):
    global speed_factor
    speed_factor = val
    simulate()  # ✅ Gọi lại toàn bộ mô phỏng với tốc độ mới

# ===== Giao diện ban đầu =====
plt.rcParams.update({'font.size': 10})  # Kích thước font mặc định

# Tăng chiều cao figure để đồ thị không bị chèn
fig = plt.figure("Con lắc đơn - Runge-Kutta 4", figsize=(14, 9.5))

# Tăng chiều cao hàng 1 và 3 để làm to hai đồ thị góc và vận tốc
# Cập nhật GridSpec phù hợp với bố cục trong ảnh
gs = GridSpec(7, 4, figure=fig,
              width_ratios=[1.8, 0.1, 0.1, 1.1],     # Cột 0 chiếm nhiều hơn cho đồ thị
              height_ratios=[0.05, 1, 0.1, 1, 0.2, 0.2, 0.4])  # Không đổi

ax1 = fig.add_subplot(gs[1, 0:2])  # Góc lệch
ax2 = fig.add_subplot(gs[3, 0:2])  # Vận tốc góc
ax_pendulum = fig.add_subplot(gs[0:5, 3])  # Con lắc ở cột cuối (bên phải)

# Kiểu dáng đồ họa đẹp hơn
plt.style.use('ggplot')

# TextBoxes và Buttons với xác thực đầu vào
axbox_theta = plt.axes([0.05, 0.08, 0.08, 0.05])
textbox_theta = TextBox(axbox_theta, '', initial="45")
textbox_theta.on_submit(lambda text: validate_numeric_input(text))
fig.text(0.05, 0.14, 'Góc ban đầu (°)', ha='left', va='bottom', fontsize=9)

axbox_omega = plt.axes([0.16, 0.08, 0.08, 0.05])
textbox_omega = TextBox(axbox_omega, '', initial="0")
textbox_omega.on_submit(lambda text: validate_numeric_input(text))
fig.text(0.16, 0.14, 'Vận tốc ban đầu', ha='left', va='bottom', fontsize=9)

axbox_length = plt.axes([0.27, 0.08, 0.08, 0.05])
textbox_length = TextBox(axbox_length, '', initial="1")
textbox_length.on_submit(lambda text: validate_numeric_input(text))
fig.text(0.27, 0.14, 'Chiều dài (m)', ha='left', va='bottom', fontsize=9)

axbox_damping = plt.axes([0.38, 0.08, 0.08, 0.05])
textbox_damping = TextBox(axbox_damping, '', initial="0.1")
textbox_damping.on_submit(lambda text: validate_numeric_input(text))
fig.text(0.38, 0.14, 'Hệ số cản', ha='left', va='bottom', fontsize=9)

# Định nghĩa hàm kiểm tra số thực cho tất cả TextBox
def validate_input(text_box):
    def submit_if_valid(text):
        if not text.replace('.', '', 1).replace('-', '', 1).isdigit():
            text_box.set_val(text_box.start_value)
            return False
        return True
    text_box.on_text_change(submit_if_valid)

# Áp dụng xác thực cho tất cả các TextBox
validate_input(textbox_theta)
validate_input(textbox_omega)
validate_input(textbox_length)
validate_input(textbox_damping)

# Thêm thanh trượt điều chỉnh tốc độ
ax_speed = plt.axes([0.05, 0.02, 0.15, 0.03])
speed_slider = Slider(ax_speed, 'Tốc độ', 0.1, 3.0, valinit=1.0)
speed_slider.on_changed(update_speed)

# Các nút điều khiển
ax_button_start = plt.axes([0.52, 0.08, 0.12, 0.05])
button_start = Button(ax_button_start, 'Bắt đầu mô phỏng')
button_start.on_clicked(simulate)

ax_button_pause = plt.axes([0.67, 0.08, 0.08, 0.05])
button_pause = Button(ax_button_pause, 'Dừng')
button_pause.on_clicked(pause_animation)

ax_button_resume = plt.axes([0.78, 0.08, 0.08, 0.05])
button_resume = Button(ax_button_resume, 'Tiếp tục')
button_resume.on_clicked(resume_animation)

# ax_button_reset = plt.axes([0.84, 0.08, 0.08, 0.05])
# button_reset = Button(ax_button_reset, 'Reset khung nhìn')
# button_reset.on_clicked(reset_view)

# Khởi tạo các đối tượng
line, = ax_pendulum.plot([], [], 'k-', lw=2)
bob = Circle((0, 0), radius=0.1, fc='red', ec='darkred', zorder=4)
ax_pendulum.add_patch(bob)
time_label = ax_pendulum.text(0.05, 0.1, '', transform=ax_pendulum.transAxes, fontsize=10)
point_angle = ax1.scatter([], [], color='red')
point_velocity = ax2.scatter([], [], color='red')
trail, = ax_pendulum.plot([], [], 'r-', lw=1, alpha=0.5)
sol = np.zeros((1, 2))
length = 1
time = np.linspace(0, 1, 2)
ani = None

# Thiết lập các đồ thị ban đầu để tránh hiển thị trống
ax1.set_ylim(-0.1, 0.1)
ax1.set_title("Góc lệch theo thời gian")
ax1.set_xlabel("Thời gian (s)")
ax1.set_ylabel("Góc (rad)")
ax1.grid(True, alpha=0.3)

ax2.set_ylim(-0.1, 0.1)
ax2.set_title("Vận tốc góc theo thời gian")
ax2.set_xlabel("Thời gian (s)")
ax2.set_ylabel("Vận tốc (rad/s)")
ax2.grid(True, alpha=0.3)

ax_pendulum.set_xlim(-1.5*length, 1.5*length)
ax_pendulum.set_ylim(-1.5*length, 0.5*length)
ax_pendulum.axhline(0, color='green', linestyle='--', alpha=0.5)
ax_pendulum.axvline(0, color='green', linestyle='--', alpha=0.5)
ax_pendulum.set_title("Mô phỏng con lắc")
ax_pendulum.set_aspect('equal')
ax_pendulum.grid(True, alpha=0.3)

# Thêm tiêu đề và chú thích với khoảng cách phù hợp
fig.suptitle('Mô phỏng Con lắc đơn sử dụng phương pháp Runge-Kutta bậc 4', fontsize=14, y=0.98)

# Phương trình sát với phần điều khiển dưới cùng
eq_box = plt.axes([0.65, 0.35, 0.4, 0.035])
eq_box.axis('off')  # Ẩn viền hộp
eq_box.text(0, 0.5, 'Phương trình: d²θ/dt² = -(g/l)·sin(θ) - (k/l)·dθ/dt', fontsize=10, 
           verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.show()