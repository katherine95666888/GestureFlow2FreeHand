import cv2
import mediapipe as mp
import json
import time
import pyautogui
import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import threading
import numpy as np

# ======== 抽象输入设备类 ========
class InputDevice(ABC):
    def __init__(self, name: str = "UnknownDevice"):
        self.name = name

    def move(self, x: int, y: int):
        print(f"[{self.name}] 移动到位置 ({x}, {y})")

    def click(self, x: int = None, y: int = None):
        pos = f"位置 ({x}, {y})" if x is not None and y is not None else "当前位置"
        print(f"[{self.name}] 点击 {pos}")

    def double_click(self, x: int = None, y: int = None):
        pos = f"位置 ({x}, {y})" if x is not None and y is not None else "当前位置"
        print(f"[{self.name}] 双击 {pos}")

    @abstractmethod
    def unique_feature(self):
        pass


# ======== 鼠标类 ========
class Mouse(InputDevice):
    def __init__(self, name: str = "Mouse"):
        super().__init__(name)
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.5

    def move(self, x: int, y: int):
        # 平滑鼠标移动
        smooth_x = self.prev_x + (x - self.prev_x) * self.smoothing_factor
        smooth_y = self.prev_y + (y - self.prev_y) * self.smoothing_factor
        
        # 将坐标映射到屏幕尺寸
        screen_x = int(smooth_x * self.screen_width)
        screen_y = int(smooth_y * self.screen_height)
        
        pyautogui.moveTo(screen_x, screen_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y

    def click(self, x: int = None, y: int = None):
        if x is not None and y is not None:
            pyautogui.click(x, y)
        else:
            pyautogui.click()

    def double_click(self, x: int = None, y: int = None):
        if x is not None and y is not None:
            pyautogui.doubleClick(x, y)
        else:
            pyautogui.doubleClick()

    def scroll_up(self, amount: int = 3):
        pyautogui.scroll(amount)
        print(f"[{self.name}] 页面向上滚动 {amount} 格")

    def scroll_down(self, amount: int = 3):
        pyautogui.scroll(-amount)
        print(f"[{self.name}] 页面向下滚动 {amount} 格")

    def right_click(self, x: int = None, y: int = None):
        if x is not None and y is not None:
            pyautogui.rightClick(x, y)
        else:
            pyautogui.rightClick()
        print(f"[{self.name}] 右键点击 位置 ({x}, {y})" if x is not None and y is not None else f"[{self.name}] 右键点击 当前位置")

    def unique_feature(self):
        print(f"[{self.name}] 独有功能：可编程宏按键，DPI切换等")


# ======== 触摸屏类 ========
class TouchScreen(InputDevice):
    def __init__(self, name: str = "TouchScreen"):
        super().__init__(name)

    def three_finger_gesture(self, gesture_type: str):
        """三指手势操作"""
        if gesture_type == "swipe_up":
            # 三指上滑
            pyautogui.hotkey('win', 'tab')
            print(f"[{self.name}] 三指上滑 - 切换应用程序")
        elif gesture_type == "swipe_down":
            # 三指下滑
            pyautogui.hotkey('win', 'd')
            print(f"[{self.name}] 三指下滑 - 显示桌面")
        elif gesture_type == "swipe_left":
            # 三指左滑
            pyautogui.hotkey('alt', 'tab')
            print(f"[{self.name}] 三指左滑 - 切换窗口")
        elif gesture_type == "swipe_right":
            # 三指右滑
            pyautogui.hotkey('alt', 'shift', 'tab')
            print(f"[{self.name}] 三指右滑 - 反向切换窗口")

    def pinch_in(self):
        # 模拟双指捏合缩小
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print(f"[{self.name}] 双指捏合缩小")

    def pinch_out(self):
        # 模拟双指张开放大
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print(f"[{self.name}] 双指张开放大")

    def unique_feature(self):
        print(f"[{self.name}] 独有功能：多点触控，多指手势识别等")


# ======== 手势识别类 ========
class Gesture:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.click_threshold = 0.05
        self.scroll_threshold = 0.03
        # 初始化摄像头操作区域为视场的1/4，居中
        self.roi_x, self.roi_y = 0.25, 0.25
        self.roi_width, self.roi_height = 0.5, 0.5

    def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5
    
    def is_finger_folded(self, finger_tip, finger_dip, palm):
        """判断手指是否弯曲"""
        tip_distance = self.calculate_distance(finger_tip, palm)
        dip_distance = self.calculate_distance(finger_dip, palm)
        return tip_distance < dip_distance

    def is_in_roi(self, x: float, y: float) -> bool:
        """检查点是否在ROI区域内"""
        return (self.roi_x <= x <= self.roi_x + self.roi_width and 
                self.roi_y <= y <= self.roi_y + self.roi_height)

    def recognize(self, frame) -> Tuple[str, Dict[str, Any]]:
        """
        识别手势，返回手势名称和相关参数
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return "no_hand", {}

        # 处理多手情况，这里我们主要关注一只手
        hand_landmarks = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # 获取关键点
        landmarks = hand_landmarks.landmark
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_dip = landmarks[7]
        middle_dip = landmarks[11]
        ring_dip = landmarks[15]
        pinky_dip = landmarks[19]
        
        wrist = landmarks[0]
        
        # 计算捏合距离
        index_pinch = self.calculate_distance(thumb_tip, index_tip)
        middle_pinch = self.calculate_distance(thumb_tip, middle_tip)
        ring_pinch = self.calculate_distance(thumb_tip, ring_tip)
        pinky_pinch = self.calculate_distance(thumb_tip, pinky_tip)
        
        # 获取手部中心点（用于移动控制）
        palm_x = (landmarks[0].x + landmarks[9].x) / 2
        palm_y = (landmarks[0].y + landmarks[9].y) / 2
        
        # 检查是否在ROI区域内
        if not self.is_in_roi(palm_x, palm_y):
            return "out_of_roi", {}
        
        # 检测数字手势（1-6）
        # 数字1：仅食指伸出
        if (not self.is_finger_folded(index_tip, index_dip, wrist) and
            self.is_finger_folded(middle_tip, middle_dip, wrist) and
            self.is_finger_folded(ring_tip, ring_dip, wrist) and
            self.is_finger_folded(pinky_tip, pinky_dip, wrist)):
            return "gesture_1", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        
        # 数字2：食指和中指伸出
        elif (not self.is_finger_folded(index_tip, index_dip, wrist) and
              not self.is_finger_folded(middle_tip, middle_dip, wrist) and
              self.is_finger_folded(ring_tip, ring_dip, wrist) and
              self.is_finger_folded(pinky_tip, pinky_dip, wrist)):
            return "gesture_2", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        
        # 数字3：食指、中指和无名指伸出
        elif (not self.is_finger_folded(index_tip, index_dip, wrist) and
              not self.is_finger_folded(middle_tip, middle_dip, wrist) and
              not self.is_finger_folded(ring_tip, ring_dip, wrist) and
              self.is_finger_folded(pinky_tip, pinky_dip, wrist)):
            return "gesture_3", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        
        # 数字4：除拇指外所有手指伸出
        elif (not self.is_finger_folded(index_tip, index_dip, wrist) and
              not self.is_finger_folded(middle_tip, middle_dip, wrist) and
              not self.is_finger_folded(ring_tip, ring_dip, wrist) and
              not self.is_finger_folded(pinky_tip, pinky_dip, wrist)):
            return "gesture_4", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        
        # 在Gesture类的recognize方法中，找到数字5手势识别部分，替换为：
        # 数字5：所有手指都伸出（默认手掌状态）
        # 用作移动手势
        elif (not self.is_finger_folded(index_tip, index_dip, wrist) and
              not self.is_finger_folded(middle_tip, middle_dip, wrist) and
              not self.is_finger_folded(ring_tip, ring_dip, wrist) and
              not self.is_finger_folded(pinky_tip, pinky_dip, wrist)):
            return "gesture_5_move", {"x": palm_x, "y": palm_y}

# 在Mouse类中，替换整个move方法以正确处理坐标：
    def move(self, x: int, y: int):
        # x和y是归一化坐标(0-1之间)，需要转换为屏幕坐标
        if not (0 <= x <= 1 and 0 <= y <= 1):
            # 如果坐标不在0-1范围内，可能是像素坐标，需要归一化
            # 这种情况可能出现在手势识别返回了像素坐标
            print(f"警告：收到非归一化坐标 x={x}, y={y}")
            return
        
        # 将归一化坐标转换为屏幕坐标
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)
        
        # 应用平滑处理
        smooth_x = self.prev_x + (screen_x - self.prev_x) * self.smoothing_factor
        smooth_y = self.prev_y + (screen_y - self.prev_y) * self.smoothing_factor
        
        pyautogui.moveTo(int(smooth_x), int(smooth_y))
        self.prev_x, self.prev_y = smooth_x, smooth_y

# 在GestureBindingManager类的setup_default_bindings方法中，添加move手势的绑定：
    def setup_default_bindings(self, device_pool: Dict[str, InputDevice]):
        """设置默认绑定"""
        mouse = device_pool.get("鼠标")
        touch = device_pool.get("触摸屏")
        
        if mouse:
            self.bind("gesture_5_move", mouse, "move")
            self.bind("ok_gesture", mouse, "click")
            self.bind("orchid_finger_gesture", mouse, "right_click")
            self.bind("rap_gesture", mouse, "double_click")
            self.bind("fist", mouse, "scroll_up")
            self.bind("gesture_6", mouse, "scroll_down")
            # 添加默认的move绑定，处理默认的"move"手势
            self.bind("move", mouse, "move")
        
        if touch:
            self.bind("three_finger_swipe_up", touch, "three_finger_gesture")
            self.bind("three_finger_swipe_down", touch, "three_finger_gesture")
            self.bind("three_finger_swipe_left", touch, "three_finger_gesture")
            self.bind("three_finger_swipe_right", touch, "three_finger_gesture")


# ======== 前端界面类 ========
class GestureControlUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("手势控制系统")
        self.root.geometry("600x500")
        
        # 系统状态
        self.is_running = False
        self.gesture_system = None
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 创建标签页
        tab_control = ttk.Notebook(self.root)
        
        # 主界面
        self.main_tab = ttk.Frame(tab_control)
        tab_control.add(self.main_tab, text='主界面')
        
        # 操作绑定界面
        self.binding_tab = ttk.Frame(tab_control)
        tab_control.add(self.binding_tab, text='操作绑定')
        
        # 有效区域设置界面
        self.roi_tab = ttk.Frame(tab_control)
        tab_control.add(self.roi_tab, text='有效区域设置')
        
        tab_control.pack(expand=1, fill="both")
        
        # 主界面内容
        self.create_main_tab()
        
        # 操作绑定界面内容
        self.create_binding_tab()
        
        # 有效区域设置界面内容
        self.create_roi_tab()
        
    def create_main_tab(self):
        # 启动/暂停按钮
        self.start_button = tk.Button(self.main_tab, text="启动", command=self.toggle_system)
        self.start_button.pack(pady=10)
        
        # 系统状态显示
        self.status_label = tk.Label(self.main_tab, text="系统状态: 已停止", fg="red")
        self.status_label.pack(pady=5)
        
        # 摄像头有效区域显示（简化）
        roi_frame = tk.LabelFrame(self.main_tab, text="摄像头有效区域")
        roi_frame.pack(pady=10, padx=10, fill="x")
        
        self.roi_canvas = tk.Canvas(roi_frame, width=200, height=150, bg="black")
        self.roi_canvas.pack(pady=5)
        
        # 绘制ROI区域
        self.draw_roi()
        
        # 简单手势使用提示
        tips_frame = tk.LabelFrame(self.main_tab, text="手势使用提示")
        tips_frame.pack(pady=10, padx=10, fill="x")
        
        tips = [
            "数字5的移动 - 鼠标移动",
            "OK手势 - 左单击",
            "兰花指手势 - 右单击",
            "说唱手势 - 左双击",
            "握拳 - 页面上滑",
            "数字6 - 页面下滑"
        ]
        
        for tip in tips:
            tk.Label(tips_frame, text=tip, anchor="w").pack(fill="x", padx=5, pady=2)
        
        # 用户自定义备注
        notes_frame = tk.LabelFrame(self.main_tab, text="用户备注")
        notes_frame.pack(pady=10, padx=10, fill="x")
        
        self.notes_text = tk.Text(notes_frame, height=4)
        self.notes_text.pack(pady=5, padx=5, fill="x")
        
    def create_binding_tab(self):
        # 绑定设置
        binding_frame = tk.LabelFrame(self.binding_tab, text="手势绑定设置")
        binding_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # 创建一个简单的表格来显示绑定关系
        tk.Label(binding_frame, text="手势", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(binding_frame, text="绑定操作", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        
        gestures = ["数字5移动", "OK手势", "兰花指手势", "说唱手势", "握拳", "数字6"]
        actions = ["鼠标移动", "左单击", "右单击", "左双击", "页面上滑", "页面下滑"]
        
        for i, (gesture, action) in enumerate(zip(gestures, actions), start=1):
            tk.Label(binding_frame, text=gesture).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            tk.Label(binding_frame, text=action).grid(row=i, column=1, padx=5, pady=5, sticky="w")
        
        # 保存按钮
        save_button = tk.Button(self.binding_tab, text="保存绑定设置", command=self.save_bindings)
        save_button.pack(pady=10)
        
    def create_roi_tab(self):
        # ROI设置
        roi_settings_frame = tk.LabelFrame(self.roi_tab, text="摄像头操作区域设置")
        roi_settings_frame.pack(pady=10, padx=10, fill="x")
        
        # ROI参数
        tk.Label(roi_settings_frame, text="X坐标 (0-1): ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.roi_x_var = tk.DoubleVar(value=0.25)
        tk.Entry(roi_settings_frame, textvariable=self.roi_x_var).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(roi_settings_frame, text="Y坐标 (0-1): ").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.roi_y_var = tk.DoubleVar(value=0.25)
        tk.Entry(roi_settings_frame, textvariable=self.roi_y_var).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(roi_settings_frame, text="宽度 (0-1): ").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.roi_width_var = tk.DoubleVar(value=0.5)
        tk.Entry(roi_settings_frame, textvariable=self.roi_width_var).grid(row=2, column=1, padx=5, pady=5)
        
        tk.Label(roi_settings_frame, text="高度 (0-1): ").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.roi_height_var = tk.DoubleVar(value=0.5)
        tk.Entry(roi_settings_frame, textvariable=self.roi_height_var).grid(row=3, column=1, padx=5, pady=5)
        
        # 更新按钮
        update_button = tk.Button(self.roi_tab, text="更新操作区域", command=self.update_roi)
        update_button.pack(pady=10)
        
        # ROI可视化
        roi_vis_frame = tk.LabelFrame(self.roi_tab, text="区域预览")
        roi_vis_frame.pack(pady=10, padx=10, fill="x")
        
        self.roi_vis_canvas = tk.Canvas(roi_vis_frame, width=200, height=150, bg="black")
        self.roi_vis_canvas.pack(pady=5)
        
        # 绘制初始ROI
        self.draw_roi_preview()
        
    def draw_roi(self):
        """绘制主界面的ROI区域"""
        self.roi_canvas.delete("all")
        # 绘制整个区域
        self.roi_canvas.create_rectangle(0, 0, 200, 150, outline="white")
        # 绘制ROI区域
        x, y, w, h = 50, 37.5, 100, 75  # 1/4区域，居中
        self.roi_canvas.create_rectangle(x, y, x+w, y+h, outline="green", width=2)
        self.roi_canvas.create_text(100, 140, text="有效操作区域", fill="green", font=("Arial", 8))
        
    def draw_roi_preview(self):
        """绘制ROI设置界面的预览"""
        self.roi_vis_canvas.delete("all")
        # 绘制整个区域
        self.roi_vis_canvas.create_rectangle(0, 0, 200, 150, outline="white")
        # 绘制ROI区域
        x = self.roi_x_var.get() * 200
        y = self.roi_y_var.get() * 150
        w = self.roi_width_var.get() * 200
        h = self.roi_height_var.get() * 150
        self.roi_vis_canvas.create_rectangle(x, y, x+w, y+h, outline="green", width=2)
        
    def toggle_system(self):
        """切换系统启动/停止状态"""
        if not self.is_running:
            self.start_system()
        else:
            self.stop_system()
            
    def start_system(self):
        """启动手势识别系统"""
        self.is_running = True
        self.start_button.config(text="停止")
        self.status_label.config(text="系统状态: 运行中", fg="green")
        
        # 在新线程中运行手势识别系统
        self.system_thread = threading.Thread(target=self.run_gesture_system)
        self.system_thread.daemon = True
        self.system_thread.start()
        
    def stop_system(self):
        """停止手势识别系统"""
        self.is_running = False
        self.start_button.config(text="启动")
        self.status_label.config(text="系统状态: 已停止", fg="red")
        
        if self.gesture_system:
            self.gesture_system.stop()
            
    def run_gesture_system(self):
        """运行手势识别系统"""
        # 初始化设备
        mouse = Mouse("鼠标")
        touch = TouchScreen("触摸屏")
        
        device_pool = {
            mouse.name: mouse,
            touch.name: touch,
        }
        
        # 初始化手势识别器
        self.gesture_system = GestureControlSystem(device_pool)
        self.gesture_system.run()
        
    def update_roi(self):
        """更新ROI区域设置"""
        if self.gesture_system and self.gesture_system.gesture_detector:
            x = self.roi_x_var.get()
            y = self.roi_y_var.get()
            w = self.roi_width_var.get()
            h = self.roi_height_var.get()
            
            # 限制参数范围
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            self.gesture_system.gesture_detector.set_roi(x, y, w, h)
            print(f"ROI已更新: x={x}, y={y}, width={w}, height={h}")
            
            # 更新预览
            self.draw_roi_preview()
            
    def save_bindings(self):
        """保存绑定设置"""
        print("绑定设置已保存")
        
    def run(self):
        """运行UI"""
        self.root.mainloop()


# ======== 手势控制系统主类 ========
class GestureControlSystem:
    def __init__(self, device_pool: Dict[str, InputDevice]):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        
        # 初始化手势识别器
        self.gesture_detector = Gesture()
        
        # 初始化设备池
        self.device_pool = device_pool
        
        # 初始化手势绑定管理器
        self.binding_manager = GestureBindingManager()
        self.binding_manager.load_from_file(device_pool)
        
        # 系统状态
        self.running = False
        
    def run(self):
        """运行手势控制系统"""
        self.running = True
        
        print("手势控制系统已启动")
        print("支持的手势：")
        print("- 数字1-6手势")
        print("- OK手势（拇指和食指捏合）")
        print("- 兰花指手势（拇指和中指捏合）")
        print("- 说唱手势（拇指和无名指捏合）")
        print("- 握拳手势")
        print("- 数字5的移动控制鼠标移动")
        print("按 'q' 键退出程序")
        
        last_gesture_time = time.time()
        gesture_cooldown = 0.3  # 手势冷却时间，防止误触
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 识别手势
            gesture_name, params = self.gesture_detector.recognize(frame)
            
            # 执行对应操作（带冷却时间限制）
            current_time = time.time()
            if gesture_name not in ["no_hand", "out_of_roi"] and current_time - last_gesture_time > gesture_cooldown:
                print(f"识别手势：{gesture_name}，参数：{params}")
                self.binding_manager.execute(gesture_name, **params)
                last_gesture_time = current_time
            
            # 显示ROI区域
            h, w = frame.shape[:2]
            x1 = int(self.gesture_detector.roi_x * w)
            y1 = int(self.gesture_detector.roi_y * h)
            x2 = int((self.gesture_detector.roi_x + self.gesture_detector.roi_width) * w)
            y2 = int((self.gesture_detector.roi_y + self.gesture_detector.roi_height) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow("Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # 按'q'键退出
            if key == ord('q'):
                break
        
        self.stop()
        
    def stop(self):
        """停止手势控制系统"""
        self.running = False
        if self.gesture_detector:
            self.gesture_detector.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    # 创建并运行前端界面
    ui = GestureControlUI()
    ui.run()


if __name__ == "__main__":
    main()