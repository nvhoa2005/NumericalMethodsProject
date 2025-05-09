# File: pendulum_pygame_full.py
import pygame
import numpy as np
import sys

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

# ===== GUI pygame =====
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Con lắc đơn - Runge-Kutta 4 (Pygame GUI)")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 20)
bigfont = pygame.font.SysFont('Arial', 24, bold=True)

# ===== Colors =====
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 200, 0)
RED = (255, 0, 0)

# ===== Input fields =====
class InputBox:
    def __init__(self, x, y, w, h, label, default):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = BLACK
        self.text = default
        self.txt_surface = font.render(self.text, True, self.color)
        self.label = label
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode
            self.txt_surface = font.render(self.text, True, self.color)

    def draw(self, screen):
        label_surface = font.render(self.label, True, BLACK)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 25))
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))

    def get_value(self):
        try:
            return float(self.text)
        except ValueError:
            return None

# ===== Buttons =====
def draw_button(rect, text, active=True):
    pygame.draw.rect(screen, GRAY if active else (150,150,150), rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    txt = font.render(text, True, BLACK)
    screen.blit(txt, (rect.x + 10, rect.y + 10))

# ===== Inputs & Buttons setup =====
inputs = [
    InputBox(50, 600, 80, 35, 'Góc (deg)', '45'),
    InputBox(160, 600, 80, 35, 'Vận tốc', '0'),
    InputBox(270, 600, 80, 35, 'Chiều dài', '1'),
    InputBox(380, 600, 80, 35, 'Ma sát', '0.1')
]
btn_start = pygame.Rect(500, 600, 140, 40)
btn_pause = pygame.Rect(660, 600, 80, 40)
btn_resume = pygame.Rect(760, 600, 100, 40)

# ===== Simulation State =====
simulating = False
paused = False
frame = 0
sol = []
time = []

# ===== Constants =====
g = 9.81
scale = 250
CENTER = (WIDTH // 2, HEIGHT // 3)

# ===== Main Loop =====
running = True
while running:
    clock.tick(60)
    screen.fill(WHITE)

    # Biểu đồ góc lệch (trên)
    if simulating and frame > 1:
        chart_rect1 = pygame.Rect(50, 50, 400, 150)
        pygame.draw.rect(screen, (240, 240, 240), chart_rect1)
        pygame.draw.rect(screen, BLACK, chart_rect1, 1)
        max_theta = max(np.abs(sol[:,0]))
        for i in range(1, frame):
            x1 = chart_rect1.x + (i-1) * chart_rect1.w / len(time)
            y1 = chart_rect1.y + chart_rect1.h / 2 - sol[i-1,0] / max_theta * (chart_rect1.h / 2)
            x2 = chart_rect1.x + i * chart_rect1.w / len(time)
            y2 = chart_rect1.y + chart_rect1.h / 2 - sol[i,0] / max_theta * (chart_rect1.h / 2)
            pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 1)
        label1 = font.render("Góc (rad) theo thời gian", True, BLACK)
        screen.blit(label1, (chart_rect1.x, chart_rect1.y - 25))

        # Biểu đồ vận tốc góc (dưới)
        chart_rect2 = pygame.Rect(50, 230, 400, 150)
        pygame.draw.rect(screen, (240, 240, 240), chart_rect2)
        pygame.draw.rect(screen, BLACK, chart_rect2, 1)
        max_omega = max(np.abs(sol[:,1]))
        for i in range(1, frame):
            x1 = chart_rect2.x + (i-1) * chart_rect2.w / len(time)
            y1 = chart_rect2.y + chart_rect2.h / 2 - sol[i-1,1] / max_omega * (chart_rect2.h / 2)
            x2 = chart_rect2.x + i * chart_rect2.w / len(time)
            y2 = chart_rect2.y + chart_rect2.h / 2 - sol[i,1] / max_omega * (chart_rect2.h / 2)
            pygame.draw.line(screen, (255, 140, 0), (x1, y1), (x2, y2), 1)
        label2 = font.render("Vận tốc góc (rad/s) theo thời gian", True, BLACK)
        screen.blit(label2, (chart_rect2.x, chart_rect2.y - 25))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for box in inputs:
            box.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if btn_start.collidepoint(event.pos):
                try:
                    theta0 = float(inputs[0].text)
                    omega0 = float(inputs[1].text)
                    length = float(inputs[2].text)
                    damping = float(inputs[3].text)
                    h = 0.025
                    T = 60
                    time = np.arange(0, T, h)
                    y0 = [np.radians(theta0), omega0]
                    sol = runge_kutta_4(equation, y0, time, h, g, length, damping)
                    simulating = True
                    paused = False
                    frame = 0
                except:
                    pass
            elif btn_pause.collidepoint(event.pos):
                paused = True
            elif btn_resume.collidepoint(event.pos):
                paused = False

    for box in inputs:
        box.draw(screen)

    draw_button(btn_start, 'Bắt đầu mô phỏng', active=not simulating or paused)
    draw_button(btn_pause, 'Dừng', active=simulating and not paused)
    draw_button(btn_resume, 'Tiếp tục', active=simulating and paused)

    if simulating and not paused and frame < len(time):
        theta = sol[frame, 0]
        omega = sol[frame, 1]
        ox, oy = CENTER
        x = ox + length * scale * np.sin(theta)
        y = oy + length * scale * np.cos(theta)

        pygame.draw.line(screen, BLACK, (ox, oy), (x, y), 3)
        pygame.draw.circle(screen, RED, (int(x), int(y)), 12)
        pygame.draw.line(screen, GREEN, (ox - 150, oy), (ox + 150, oy), 1)
        pygame.draw.line(screen, GREEN, (ox, oy - 150), (ox, oy + 250), 1)

        status = font.render(f"Thời gian: {time[frame]:.2f}s | Góc: {np.degrees(theta):.1f}° | Vận tốc: {omega:.2f} rad/s", True, BLACK)
        screen.blit(status, (50, 550))
        frame += 1

    pygame.display.flip()

pygame.quit()
sys.exit()
