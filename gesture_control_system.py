import cv2
import mediapipe as mp
import json
import time
import pyautogui
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

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

    def scroll_up(self, amount: int = 1):
        pyautogui.scroll(amount)
        print(f"[{self.name}] 滚轮向上滚动 {amount} 格")

    def scroll_down(self, amount: int = 1):
        pyautogui.scroll(-amount)
        print(f"[{self.name}] 滚轮向下滚动 {amount} 格")

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

    def rotate(self, angle: int):
        print(f"[{self.name}] 双指旋转 {angle} 度")

    def swipe_from_edge(self, direction: str):
        print(f"[{self.name}] 从屏幕{direction}边缘滑动")

    def unique_feature(self):
        print(f"[{self.name}] 独有功能：多点触控，多指手势识别等")


# ======== 手势识别类 ========
class Gesture:
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.click_threshold = 0.05
        self.scroll_threshold = 0.03

    def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5
    
    def is_finger_folded(self, finger_tip, finger_dip, palm):
        """判断手指是否弯曲"""
        tip_distance = self.calculate_distance(finger_tip, palm)
        dip_distance = self.calculate_distance(finger_dip, palm)
        return tip_distance < dip_distance

    def recognize(self, frame) -> Tuple[str, Dict[str, Any]]:
        """
        识别手势，返回手势名称和相关参数
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return "no_hand", {}

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
        
        # 检测手势
        if index_pinch < self.click_threshold:
            return "left_click", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        elif middle_pinch < self.click_threshold:
            return "scroll_down", {}
        elif ring_pinch < self.click_threshold:
            return "scroll_up", {}
        elif pinky_pinch < self.click_threshold:
            return "double_click", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        # 检测右键点击（拇指与无名指捏合且中指伸直）
        elif (ring_pinch < self.click_threshold and 
              not self.is_finger_folded(middle_tip, middle_dip, wrist)):
            return "right_click", {"x": int(palm_x * frame.shape[1]), "y": int(palm_y * frame.shape[0])}
        # 检测速度滚动（仅伸出食指）
        elif (not self.is_finger_folded(index_tip, index_dip, wrist) and
              self.is_finger_folded(middle_tip, middle_dip, wrist) and
              self.is_finger_folded(ring_tip, ring_dip, wrist) and
              self.is_finger_folded(pinky_tip, pinky_dip, wrist)):
            return "speed_scroll", {"x": palm_x, "y": palm_y}
        else:
            # 默认返回移动手势
            return "move", {"x": palm_x, "y": palm_y}

    def close(self):
        self.hands.close()


# ======== 手势绑定管理类 ========
class GestureBindingManager:
    def __init__(self):
        self.bindings = {}  # 形如 { "手势名称": (设备对象, "设备方法名") }
        self.config_file = "gesture_config.json"

    def bind(self, gesture_name: str, device: InputDevice, method_name: str):
        self.bindings[gesture_name] = (device, method_name)

    def unbind(self, gesture_name: str):
        if gesture_name in self.bindings:
            del self.bindings[gesture_name]

    def execute(self, gesture_name: str, **kwargs):
        if gesture_name in self.bindings:
            device, method_name = self.bindings[gesture_name]
            method = getattr(device, method_name, None)
            if method:
                # 根据方法签名过滤参数
                import inspect
                sig = inspect.signature(method)
                method_params = sig.parameters
                
                # 只传递方法需要的参数
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in method_params}
                method(**filtered_kwargs)
            else:
                print(f"设备[{device.name}]没有方法[{method_name}]")
        else:
            print(f"未绑定手势[{gesture_name}]")

    def save_to_file(self):
        # 保存绑定关系，只保存设备名和方法名，设备对象需重建后绑定
        simple_bindings = {k: (v[0].name, v[1]) for k, v in self.bindings.items()}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(simple_bindings, f, indent=2, ensure_ascii=False)
        print(f"手势绑定已保存到 {self.config_file}")

    def load_from_file(self, device_pool: Dict[str, InputDevice]):
        # device_pool: dict 设备名->设备对象，用于恢复绑定
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                simple_bindings = json.load(f)
            self.bindings = {}
            for gesture_name, (device_name, method_name) in simple_bindings.items():
                device = device_pool.get(device_name)
                if device:
                    self.bindings[gesture_name] = (device, method_name)
                else:
                    print(f"设备池中无设备名: {device_name}")
            print(f"手势绑定已从 {self.config_file} 加载")
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在，使用默认绑定")
            self.setup_default_bindings(device_pool)

    def setup_default_bindings(self, device_pool: Dict[str, InputDevice]):
        """设置默认绑定"""
        mouse = device_pool.get("罗技鼠标")
        if mouse:
            self.bind("move", mouse, "move")
            self.bind("left_click", mouse, "click")
            self.bind("double_click", mouse, "double_click")
            self.bind("right_click", mouse, "right_click")
            self.bind("scroll_up", mouse, "scroll_up")
            self.bind("scroll_down", mouse, "scroll_down")
            self.bind("speed_scroll", mouse, "move")  # 简化处理，实际应有专门的速度滚动处理


def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 初始化手势识别器
    gesture_detector = Gesture()
    
    # 初始化设备
    mouse = Mouse("罗技鼠标")
    touch = TouchScreen("iPad Pro")
    
    device_pool = {
        mouse.name: mouse,
        touch.name: touch,
    }
    
    # 初始化手势绑定管理器
    binding_manager = GestureBindingManager()
    binding_manager.load_from_file(device_pool)
    
    print("手势控制系统已启动")
    print("支持的手势：")
    print("- 手掌移动：控制鼠标移动")
    print("- 食指和拇指捏合：鼠标左键点击")
    print("- 中指和拇指捏合：鼠标滚轮向下")
    print("- 无名指和拇指捏合：鼠标滚轮向上")
    print("- 小指和拇指捏合：鼠标左键双击")
    print("- 无名指和拇指捏合且中指伸直：鼠标右键点击")
    print("- 仅伸出食指（其他手指弯曲）：速度滚动")
    print("按 'q' 键退出程序，按 's' 键保存配置")
    
    last_gesture_time = time.time()
    gesture_cooldown = 0.3  # 手势冷却时间，防止误触
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 识别手势
        gesture_name, params = gesture_detector.recognize(frame)
        
        # 执行对应操作（带冷却时间限制）
        current_time = time.time()
        if gesture_name != "no_hand" and current_time - last_gesture_time > gesture_cooldown:
            print(f"识别手势：{gesture_name}，参数：{params}")
            binding_manager.execute(gesture_name, **params)
            last_gesture_time = current_time
        
        cv2.imshow("Gesture Control", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # 按'q'键退出
        if key == ord('q'):
            break
        # 按's'键保存配置
        elif key == ord('s'):
            binding_manager.save_to_file()
    
    # 清理资源
    gesture_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()