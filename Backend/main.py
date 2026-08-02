import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import deque
import math
import time
import random
import threading
import base64
import pygame
import ws_server

GRAVITY = 0.7
SLASH_SPEED_THRESHOLD = 650
COMBO_WINDOW = 0.6
DETECT_W, DETECT_H = 320, 240
MAX_FRUITS_ON_SCREEN = 10
MAX_PARTICLES = 150

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


# ---------- Helpers ----------

def load_png_rgba(path, size=180):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Missing image: {path}")
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def overlay_rgba(background, overlay, x, y):
    h, w = overlay.shape[:2]
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = x1 + w, y1 + h
    bh, bw = background.shape[:2]
    ox1, oy1 = max(0, -x1), max(0, -y1)
    ox2, oy2 = w - max(0, x2 - bw), h - max(0, y2 - bh)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(bw, x2), min(bh, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    alpha = overlay_crop[:, :, 3:4] / 255.0
    bg_region = background[y1:y2, x1:x2]
    blended = bg_region * (1 - alpha) + overlay_crop[:, :, :3] * alpha
    background[y1:y2, x1:x2] = blended.astype(np.uint8)


def line_circle_intersect(p1, p2, center, radius):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx * dx + dy * dy
    if a == 0:
        return math.hypot(fx, fy) <= radius
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 9)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 3
        self.color = color
        self.radius = random.randint(3, 7)
        self.life = 1.0

    def update(self):
        self.vy += GRAVITY * 0.5
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.035

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = max(0, self.life)
        r = max(1, int(self.radius * alpha))
        cx, cy = int(self.x), int(self.y)
        h, w = frame.shape[:2]
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r + 1), min(h, cy + r + 1)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.circle(overlay, (cx - x1, cy - y1), r, self.color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    def dead(self):
        return self.life <= 0


def spawn_juice(particles, x, y, color, count=12):
    for _ in range(count):
        particles.append(Particle(x, y, color))


# ---------- Fruit ----------

class Fruit:
    def __init__(self, whole_img, left_img, right_img, x, y, vx, vy, radius=70, is_bomb=False, juice_color=(60, 160, 255)):
        self.whole = whole_img
        self.left = left_img
        self.right = right_img
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.sliced = False
        self.is_bomb = is_bomb
        self.juice_color = juice_color
        self.left_x, self.left_y = x, y
        self.right_x, self.right_y = x, y
        self.left_vx, self.left_vy = 0, 0
        self.right_vx, self.right_vy = 0, 0

    def slice(self, slash_dx, slash_dy):
        self.sliced = True
        perp_x, perp_y = -slash_dy, slash_dx
        norm = math.hypot(perp_x, perp_y) or 1
        perp_x, perp_y = perp_x / norm, perp_y / norm
        self.left_x, self.left_y = self.x, self.y
        self.right_x, self.right_y = self.x, self.y
        self.left_vx, self.left_vy = -perp_x * 5 - 2, -perp_y * 5 - 6
        self.right_vx, self.right_vy = perp_x * 5 + 2, perp_y * 5 - 6

    def update(self):
        if not self.sliced:
            self.vy += GRAVITY
            self.x += self.vx
            self.y += self.vy
        else:
            self.left_vy += GRAVITY
            self.right_vy += GRAVITY
            self.left_x += self.left_vx
            self.left_y += self.left_vy
            self.right_x += self.right_vx
            self.right_y += self.right_vy

    def draw(self, frame):
        if not self.sliced:
            overlay_rgba(frame, self.whole, self.x, self.y)
        else:
            overlay_rgba(frame, self.left, self.left_x, self.left_y)
            overlay_rgba(frame, self.right, self.right_x, self.right_y)

    def off_screen(self, h):
        if not self.sliced:
            return self.y > h + 150
        return self.left_y > h + 150 and self.right_y > h + 150


def make_fruit(fruit_types, w, h, elapsed):
    bomb_chance = min(0.35, 0.08 + elapsed * 0.005)
    names = list(fruit_types.keys())
    fruit_names = [n for n in names if not fruit_types[n]["is_bomb"]]
    bomb_names = [n for n in names if fruit_types[n]["is_bomb"]]

    if bomb_names and random.random() < bomb_chance:
        name = random.choice(bomb_names)
    else:
        name = random.choice(fruit_names)

    data = fruit_types[name]
    x = random.randint(150, w - 150)
    y = h + 50
    power_boost = min(8, elapsed * 0.15)
    vx = random.uniform(-4, 4)
    vy = random.uniform(-30, -24) - power_boost
    return Fruit(data["whole"], data["left"], data["right"], x, y, vx, vy,
                 is_bomb=data["is_bomb"], juice_color=data.get("juice_color", (60, 160, 255)))


def spawn_wave(fruit_types, w, h, elapsed, current_count):
    if current_count >= MAX_FRUITS_ON_SCREEN:
        return []
    if elapsed < 10:
        count = random.choice([1, 1, 2])
    elif elapsed < 25:
        count = random.choice([1, 2, 2, 3])
    else:
        count = random.choice([2, 3, 3, 4])
    count = min(count, MAX_FRUITS_ON_SCREEN - current_count)
    return [make_fruit(fruit_types, w, h, elapsed) for _ in range(count)]


def get_spawn_interval(elapsed):
    difficulty_steps = elapsed // 10
    interval = 1.1 - (difficulty_steps * 0.12)
    return max(0.35, interval)


# ---------- Main ----------

def main():
    cap = ThreadedCamera(0)
    time.sleep(0.5)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("Error: Could not open webcam.")
        return
    first_frame = cv2.flip(first_frame, 1)

    native_h, native_w = first_frame.shape[:2]
    aspect = native_h / native_w
    DISPLAY_W = 1000
    DISPLAY_H = int(DISPLAY_W * aspect)

    base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    pygame.mixer.init()
    slash_sound = pygame.mixer.Sound("assets/sounds/slash.wav")
    bomb_sound = pygame.mixer.Sound("assets/sounds/bomb.wav")
    gameover_sound = pygame.mixer.Sound("assets/sounds/gameover.wav")
    slash_sound.set_volume(0.6)
    bomb_sound.set_volume(0.8)
    gameover_sound.set_volume(0.7)

    pygame.mixer.music.load("assets/sounds/bgm.wav")
    pygame.mixer.music.set_volume(0.15) 
    pygame.mixer.music.play(loops=-1)  

    fruit_types = {
        "avocado": {"whole": load_png_rgba("assets/fruits/avocado_whole.png"),
                    "left": load_png_rgba("assets/fruits/avocado_left.png"),
                    "right": load_png_rgba("assets/fruits/avocado_right.png"),
                    "is_bomb": False, "juice_color": (60, 200, 120)},
        "dragonfruit": {"whole": load_png_rgba("assets/fruits/dragonfruit_whole.png"),
                         "left": load_png_rgba("assets/fruits/dragonfruit_left.png"),
                         "right": load_png_rgba("assets/fruits/dragonfruit_right.png"),
                         "is_bomb": False, "juice_color": (180, 60, 220)},
        "guava": {"whole": load_png_rgba("assets/fruits/guava_whole.png"),
                  "left": load_png_rgba("assets/fruits/guava_left.png"),
                  "right": load_png_rgba("assets/fruits/guava_right.png"),
                  "is_bomb": False, "juice_color": (110, 220, 240)},
        "kiwi": {"whole": load_png_rgba("assets/fruits/kiwi_whole.png"),
                 "left": load_png_rgba("assets/fruits/kiwi_left.png"),
                 "right": load_png_rgba("assets/fruits/kiwi_right.png"),
                 "is_bomb": False, "juice_color": (50, 200, 90)},
        "lemon": {"whole": load_png_rgba("assets/fruits/lemon_whole.png"),
                  "left": load_png_rgba("assets/fruits/lemon_left.png"),
                  "right": load_png_rgba("assets/fruits/lemon_right.png"),
                  "is_bomb": False, "juice_color": (40, 230, 250)},
        "papaya": {"whole": load_png_rgba("assets/fruits/papaya_whole.png"),
                   "left": load_png_rgba("assets/fruits/papaya_left.png"),
                   "right": load_png_rgba("assets/fruits/papaya_right.png"),
                   "is_bomb": False, "juice_color": (40, 150, 250)},
        "passionfruit": {"whole": load_png_rgba("assets/fruits/passionfruit_whole.png"),
                          "left": load_png_rgba("assets/fruits/passionfruit_left.png"),
                          "right": load_png_rgba("assets/fruits/passionfruit_right.png"),
                          "is_bomb": False, "juice_color": (60, 140, 230)},
        "starfruit": {"whole": load_png_rgba("assets/fruits/starfruit_whole.png"),
                      "left": load_png_rgba("assets/fruits/starfruit_left.png"),
                      "right": load_png_rgba("assets/fruits/starfruit_right.png"),
                      "is_bomb": False, "juice_color": (50, 230, 230)},
        "strawberry": {"whole": load_png_rgba("assets/fruits/strawberry_whole.png"),
                       "left": load_png_rgba("assets/fruits/strawberry_left.png"),
                       "right": load_png_rgba("assets/fruits/strawberry_right.png"),
                       "is_bomb": False, "juice_color": (60, 40, 220)},
        "cactus": {"whole": load_png_rgba("assets/cactus/cactus_whole.png"),
                   "left": load_png_rgba("assets/cactus/cactus_left.png"),
                   "right": load_png_rgba("assets/cactus/cactus_right.png"),
                   "is_bomb": True, "juice_color": (0, 0, 0)},
    }

    ws_server.start_server()
    print("WebSocket server running on ws://localhost:8765")
    print("Open the React app in your browser to play.")

    def reset_game():
        return {
            "fruits": [], "particles": [], "spawn_timer": 0, "score": 0, "lives": 3,
            "combo": 0, "last_slice_time": 0, "combo_display_timer": 0,
            "bomb_flash_timer": 0,
            "game_start_time": time.time(), "game_over": False,
        }

    state = reset_game()
    trail_history = deque(maxlen=8)
    last_time = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            continue

        raw_frame = cv2.flip(raw_frame, 1)

        small = cv2.resize(raw_frame, (DETECT_W, DETECT_H))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
        result = landmarker.detect(mp_image)

        frame = cv2.resize(raw_frame, (DISPLAY_W, DISPLAY_H))
        w, h = DISPLAY_W, DISPLAY_H

        if not state["game_over"]:
            now = time.time()
            elapsed = now - state["game_start_time"]

            spawn_interval = get_spawn_interval(elapsed)
            state["spawn_timer"] += 1 / 30
            if state["spawn_timer"] >= spawn_interval:
                state["fruits"].extend(spawn_wave(fruit_types, w, h, elapsed, len(state["fruits"])))
                state["spawn_timer"] = 0

            if result.hand_landmarks:
                index_tip = result.hand_landmarks[0][8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                dt = now - last_time
                last_time = now

                if trail_history:
                    prev = trail_history[-1]
                    dist = math.hypot(x - prev[0], y - prev[1])
                    speed = dist / dt if dt > 0 else 0
                    slash_dx, slash_dy = x - prev[0], y - prev[1]

                    if speed > SLASH_SPEED_THRESHOLD:
                        recent_points = list(trail_history)[-4:] + [(x, y)]
                        segments = list(zip(recent_points[:-1], recent_points[1:]))

                        for fruit in state["fruits"]:
                            if fruit.sliced:
                                continue
                            for seg_p1, seg_p2 in segments:
                                if line_circle_intersect(seg_p1, seg_p2, (fruit.x, fruit.y), fruit.radius):
                                    fruit.slice(slash_dx, slash_dy)
                                    if fruit.is_bomb:
                                        state["lives"] -= 1
                                        state["combo"] = 0
                                        state["bomb_flash_timer"] = 8
                                        spawn_juice(state["particles"], fruit.x, fruit.y, (0, 0, 220), count=22)
                                        bomb_sound.play()
                                    else:
                                        if now - state["last_slice_time"] <= COMBO_WINDOW:
                                            state["combo"] += 1
                                        else:
                                            state["combo"] = 1
                                        state["last_slice_time"] = now
                                        state["combo_display_timer"] = 20
                                        state["score"] += 10 * max(1, state["combo"])
                                        spawn_juice(state["particles"], fruit.x, fruit.y, fruit.juice_color, count=18)
                                        slash_sound.play()
                                    break  
                trail_history.append((x, y))
            else:
                last_time = now

            for fruit in state["fruits"]:
                fruit.update()
                fruit.draw(frame)
            state["fruits"] = [f for f in state["fruits"] if not f.off_screen(h)]

            if len(trail_history) >= 2:
                xs = [p[0] for p in trail_history]
                ys = [p[1] for p in trail_history]
                pad = 40
                bx1, by1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
                bx2, by2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
                if bx2 > bx1 and by2 > by1:
                    roi = frame[by1:by2, bx1:bx2]
                    glow_roi = np.zeros_like(roi)
                    for i in range(1, len(trail_history)):
                        p1 = (trail_history[i-1][0] - bx1, trail_history[i-1][1] - by1)
                        p2 = (trail_history[i][0] - bx1, trail_history[i][1] - by1)
                        thickness = int(i / len(trail_history) * 12) + 2
                        cv2.line(glow_roi, p1, p2, (255, 255, 200), thickness)
                    glow_roi = cv2.GaussianBlur(glow_roi, (13, 13), 0)
                    frame[by1:by2, bx1:bx2] = cv2.add(roi, glow_roi)

            for i in range(1, len(trail_history)):
                thickness = max(1, int(i / len(trail_history) * 6))
                cv2.line(frame, trail_history[i - 1], trail_history[i], (255, 255, 255), thickness)

            if trail_history:
                fx, fy = trail_history[-1]
                for _ in range(6):
                    ox, oy = random.randint(-16, 16), random.randint(-16, 16)
                    cv2.circle(frame, (fx + ox, fy + oy), random.randint(1, 3), (255, 255, 255), -1)
                cv2.circle(frame, (fx, fy), 8, (230, 230, 255), -1)

            if len(state["particles"]) > MAX_PARTICLES:
                state["particles"] = state["particles"][-MAX_PARTICLES:]

            for p in state["particles"]:
                p.update()
                p.draw(frame)
            state["particles"] = [p for p in state["particles"] if not p.dead()]

            if state["bomb_flash_timer"] > 0:
                flash_alpha = state["bomb_flash_timer"] / 8 * 0.4
                red_overlay = np.full_like(frame, (0, 0, 255))
                frame = cv2.addWeighted(red_overlay, flash_alpha, frame, 1 - flash_alpha, 0)
                state["bomb_flash_timer"] -= 1

            if state["lives"] <= 0 and not state["game_over"]:
                state["game_over"] = True
                pygame.mixer.music.stop()
                gameover_sound.play()

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        ws_server.broadcast_frame(frame_b64, state["score"], state["lives"], state["game_over"])

        if ws_server.reset_requested:
            state = reset_game()
            trail_history.clear()
            ws_server.reset_requested = False
            pygame.mixer.music.play(loops=-1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()