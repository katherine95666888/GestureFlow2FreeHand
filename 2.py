import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import ttk
import screeninfo
import numpy as np
from PIL import Image, ImageTk

# 初始化PyAutoGUI，设置安全模式
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02  # 每次操作间隔

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 获取屏幕尺寸
def get_screen_size():
    screen = screeninfo.get_monitors()[0]
    return screen.width, screen.height

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()

# 摄像头有效区域（初始为中心1/4区域）
CAM_WIDTH, CAM_HEIGHT = 640, 480  # 摄像头分辨率
ROI_WIDTH, ROI_HEIGHT = CAM_WIDTH // 2, CAM_HEIGHT // 2
ROI_X, ROI_Y = (CAM_WIDTH - ROI_WIDTH) // 2, (CAM_HEIGHT - ROI_HEIGHT) // 2

# 手势类定义
class Gesture:
    def __init__(self):
        self.gestures = {
            "NUMBER_1": self.detect_number_1,
            "NUMBER_2": self.detect_number_2,
            "NUMBER_3": self.detect_number_3,
            "NUMBER_4": self.detect_number_4,
            "NUMBER_5": self.detect_number_5,
            "NUMBER_6": self.detect_number_6,
            "OK": self.detect_ok,
            "ORCHID": self.detect_orchid,
            "RAP": self.detect_rap,
            "FIST": self.detect_fist
        }

    def detect_number_1(self, landmarks):
        # 食指伸直，其他手指弯曲
        return (landmarks[8].y < landmarks[6].y and  # 食指指尖高于关节
                all(landmarks[i].y > landmarks[i-2].y for i in [12, 16, 20]))  # 其他手指弯曲

    def detect_number_2(self, landmarks):
        # 食指和中指伸直，其他手指弯曲
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                all(landmarks[i].y > landmarks[i-2].y for i in [16, 20]))

    def detect_number_3(self, landmarks):
        # 食指、中指、无名指伸直，其他手指弯曲
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                landmarks[16].y < landmarks[14].y and
                landmarks[20].y > landmarks[18].y)

    def detect_number_4(self, landmarks):
        # 四指伸直，拇指弯曲
        return (all(landmarks[i].y < landmarks[i-2].y for i in [8, 12, 16, 20]) and
                landmarks[4].x > landmarks[3].x)

    def detect_number_5(self, landmarks):
        # 五指伸直
        return all(landmarks[i].y < landmarks[i-2].y for i in [8, 12, 16, 20]) and landmarks[4].x < landmarks[3].x

    def detect_number_6(self, landmarks):
        # 拇指和食指伸直，其他手指弯曲
        return (landmarks[4].x < landmarks[3].x and
                landmarks[8].y < landmarks[6].y and
                all(landmarks[i].y > landmarks[i-2].y for i in [12, 16, 20]))

    def detect_ok(self, landmarks):
        # 拇指和无名指指尖捏合
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        distance = ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5
        return distance < 0.05

    def detect_orchid(self, landmarks):
        # 拇指和中指指尖捏合
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
        return distance < 0.05

    def detect_rap(self, landmarks):
        # 拇指和食指指尖捏合
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        return distance < 0.05

    def detect_fist(self, landmarks):
        # 所有手指弯曲
        return all(landmarks[i].y > landmarks[i-2].y for i in [8, 12, 16, 20])

    def recognize(self, landmarks):
        for gesture_name, detect_func in self.gestures.items():
            if detect_func(landmarks):
                return gesture_name
        return None

# 鼠标控制类
class MouseControl:
    def __init__(self, sensitivity=1.0):
        self.sensitivity = sensitivity

    def move(self, x, y):
        # 将摄像头坐标映射到屏幕坐标
        screen_x = int(x * SCREEN_WIDTH * self.sensitivity)
        screen_y = int(y * SCREEN_HEIGHT * self.sensitivity)
        pyautogui.moveTo(screen_x, screen_y)

    def left_click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def double_click(self):
        pyautogui.doubleClick()

    def scroll_up(self):
        pyautogui.scroll(100)

    def scroll_down(self):
        pyautogui.scroll(-100)

# 触摸屏类（三指操作模拟）
class TouchScreen:
    def three_finger_swipe(self, direction="up"):
        if direction == "up":
            pyautogui.hotkey('ctrl', 'up')
        elif direction == "down":
            pyautogui.hotkey('ctrl', 'down')

# 前端界面
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Control")
        self.is_running = False
        self.hand = "right"  # 默认右手
        self.sensitivity = tk.DoubleVar(value=1.0)
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
        self.gesture = Gesture()
        self.mouse = MouseControl(sensitivity=self.sensitivity.get())
        self.touch = TouchScreen()

        # 绑定配置
        self.bindings = {
            "NUMBER_5": self.mouse.move,
            "OK": self.mouse.left_click,
            "ORCHID": self.mouse.right_click,
            "RAP": self.mouse.double_click,
            "FIST": self.mouse.scroll_up,
            "NUMBER_6": self.mouse.scroll_down
        }

        # 主界面
        self.setup_main_ui()
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def setup_main_ui(self):
        # 摄像头画面
        self.canvas = tk.Canvas(self.root, width=CAM_WIDTH, height=CAM_HEIGHT)
        self.canvas.pack()

        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        # 启动/暂停按钮
        ttk.Button(control_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Pause", command=self.pause).pack(side=tk.LEFT, padx=5)

        # 灵敏度调节
        ttk.Label(control_frame, text="Sensitivity:").pack(side=tk.LEFT, padx=5)
        ttk.Scale(control_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, variable=self.sensitivity,
                  command=self.update_sensitivity).pack(side=tk.LEFT, padx=5)

        # 左右手选择
        ttk.Label(control_frame, text="Hand:").pack(side=tk.LEFT, padx=5)
        hand_combo = ttk.Combobox(control_frame, values=["right", "left"], state="readonly")
        hand_combo.set(self.hand)
        hand_combo.bind("<<ComboboxSelected>>", self.set_hand)
        hand_combo.pack(side=tk.LEFT, padx=5)

        # ROI设置
        ttk.Button(control_frame, text="Set ROI", command=self.open_roi_settings).pack(side=tk.LEFT, padx=5)

        # 备注
        ttk.Label(control_frame, text="Notes:").pack(side=tk.LEFT, padx=5)
        self.notes = ttk.Entry(control_frame)
        self.notes.pack(side=tk.LEFT, padx=5)

        # 手势提示
        ttk.Label(self.root, text="Gestures: 5=Move, OK=Left Click, Orchid=Right Click, Rap=Double Click, Fist=Scroll Up, 6=Scroll Down").pack()

    def update_sensitivity(self, val):
        self.mouse.sensitivity = self.sensitivity.get()

    def set_hand(self, event):
        self.hand = event.widget.get()

    def start(self):
        self.is_running = True

    def pause(self):
        self.is_running = False

    def open_roi_settings(self):
        roi_window = tk.Toplevel(self.root)
        roi_window.title("Set ROI")
        ttk.Label(roi_window, text="ROI X:").pack()
        x_entry = ttk.Entry(roi_window)
        x_entry.insert(0, self.roi_x)
        x_entry.pack()
        ttk.Label(roi_window, text="ROI Y:").pack()
        y_entry = ttk.Entry(roi_window)
        y_entry.insert(0, self.roi_y)
        y_entry.pack()
        ttk.Label(roi_window, text="Width:").pack()
        w_entry = ttk.Entry(roi_window)
        w_entry.insert(0, self.roi_w)
        w_entry.pack()
        ttk.Label(roi_window, text="Height:").pack()
        h_entry = ttk.Entry(roi_window)
        h_entry.insert(0, self.roi_h)
        h_entry.pack()
        ttk.Button(roi_window, text="Apply", command=lambda: self.apply_roi(
            int(x_entry.get()), int(y_entry.get()), int(w_entry.get()), int(h_entry.get())
        )).pack()

    def apply_roi(self, x, y, w, h):
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = x, y, w, h

    def update_camera(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # 镜像翻转
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # 绘制ROI
                cv2.rectangle(frame, (self.roi_x, self.roi_y), 
                            (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (0, 255, 0), 2)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # 确保检测到的是用户选择的手
                        if handedness.classification[0].label.lower() == self.hand:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            gesture = self.gesture.recognize(hand_landmarks.landmark)
                            if gesture:
                                # 获取掌心位置（用于鼠标移动）
                                palm_x = hand_landmarks.landmark[0].x
                                palm_y = hand_landmarks.landmark[0].y
                                # 仅在ROI内处理
                                if (self.roi_x / CAM_WIDTH <= palm_x <= (self.roi_x + self.roi_w) / CAM_WIDTH and
                                    self.roi_y / CAM_HEIGHT <= palm_y <= (self.roi_y + self.roi_h) / CAM_HEIGHT):
                                    if gesture == "NUMBER_5":
                                        # 归一化坐标
                                        norm_x = (palm_x * CAM_WIDTH - self.roi_x) / self.roi_w
                                        norm_y = (palm_y * CAM_HEIGHT - self.roi_y) / self.roi_h
                                        self.bindings[gesture](norm_x, norm_y)
                                    elif gesture in self.bindings:
                                        self.bindings[gesture]()

                # 显示摄像头画面
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk

        self.root.after(10, self.update_camera)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
