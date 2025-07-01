import torch
import pyautogui
import cv2
import numpy as np
import time
from datetime import datetime
from pynput.mouse import Controller, Button

# Задержки по фазам (базовые)
PHASE_DELAYS = {
    "debut": (0.1, 0.6),
    "mittelspiel": (4.0, 10.0),
    "endspiel": (2.8, 10.0),
}

# Динамическая задержка: множитель уменьшения
def get_dynamic_delay_range(phase, move_count):
    base_min, base_max = PHASE_DELAYS[phase]
    reduction_factor = max(0.3, 1.0 - (move_count / 60))  # минимум 30%
    return base_min * reduction_factor, base_max * reduction_factor

move_count = 0

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/chess_arrow_detector_v2/weights/best.pt')
model.conf = 0.4
print("🤖 Бот запущен. Ожидание стрелок на экране...")

mouse = Controller()

def get_phase(move):
    if move <= 10:
        return "debut"
    elif move <= 25:
        return "mittelspiel"
    else:
        return "endspiel"

def get_screen():
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def click(x, y):
    mouse.position = (x, y)
    time.sleep(0.05)
    mouse.click(Button.left, 1)

def detect_and_act():
    global move_count
    frame = get_screen()
    results = model(frame)

    boxes = results.xyxy[0].cpu().numpy()
    detections = {"class1": [], "class2": [], "newgame": []}

    for *xyxy, conf, cls in boxes:
        class_name = model.names[int(cls)]
        x_center = int((xyxy[0] + xyxy[2]) / 2)
        y_center = int((xyxy[1] + xyxy[3]) / 2)
        detections[class_name].append((x_center, y_center))

        cv2.circle(frame, (x_center, y_center), 10, (0, 255, 0), -1)
        cv2.putText(frame, class_name, (x_center + 10, y_center - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("ChessBot View", frame)
    cv2.waitKey(1)

    from_coord, to_coord = None, None

    if detections["class1"] and detections["class2"]:
        from_coord = detections["class1"][0]
        to_coord = detections["class2"][0]
    elif len(detections["class1"]) >= 2 and not detections["class2"]:
        from_coord, to_coord = detections["class1"][:2]
    else:
        from_coord = to_coord = None

    if from_coord and to_coord:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ход #{move_count + 1}: {from_coord} → {to_coord}")
        click(*from_coord)
        time.sleep(0.1)
        click(*to_coord)

        move_count += 1
        phase = get_phase(move_count)
        dyn_min, dyn_max = get_dynamic_delay_range(phase, move_count)
        delay = np.random.uniform(dyn_min, dyn_max)
        print(f"⏳ Фаза: {phase.upper()} | Динамическая задержка: {delay:.2f} сек")
        time.sleep(delay)

    elif detections["newgame"]:
        ng_x, ng_y = detections["newgame"][0]
        print("♻️ Обнаружена кнопка 'newgame'. Перезапуск.")
        click(ng_x, ng_y)
        move_count = 0
        time.sleep(3)

while True:
    try:
        detect_and_act()
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен вручную.")
        break
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        time.sleep(2)
