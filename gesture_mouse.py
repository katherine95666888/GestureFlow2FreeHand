import cv2
import mediapipe as mp
import pyautogui
import math
import time

class GestureMouse:
    def __init__(self):
        # 初始化MediaPipe手部检测模块
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        
        # 初始化变量
        self.prev_x, self.prev_y = 0, 0
        self.click_threshold = 0.05  # 捏合阈值
        self.smoothing_factor = 0.5  # 平滑因子（提高灵敏度）
        self.last_click_time = 0
        self.click_delay = 0.3  # 点击延迟
        # 增加滚轮速度控制参数
        self.scroll_speed = 3  # 滚轮速度倍数
        self.scroll_threshold = 0.03  # 滚轮手势阈值
        
    def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_folded(self, finger_tip, finger_dip, palm):
        """判断手指是否弯曲"""
        tip_distance = self.calculate_distance(finger_tip, palm)
        dip_distance = self.calculate_distance(finger_dip, palm)
        return tip_distance < dip_distance
    
    def detect_gesture(self, hand_landmarks):
        """检测手势"""
        landmarks = hand_landmarks.landmark
        
        # 获取关键点
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
        
        # 检测手势
        if index_pinch < self.click_threshold:
            return "left_click"
        elif middle_pinch < self.click_threshold:
            return "scroll_down"
        elif ring_pinch < self.click_threshold:
            return "scroll_up"
        elif pinky_pinch < self.click_threshold:
            return "double_click"
        # 检测右键点击（拇指与无名指捏合且中指伸直）
        elif (ring_pinch < self.click_threshold and 
              not self.is_finger_folded(middle_tip, middle_dip, wrist)):
            return "right_click"
        else:
            return "move"
    
    def execute_action(self, gesture, x, y):
        """执行对应的操作"""
        current_time = time.time()
        
        if gesture == "move":
            # 平滑鼠标移动
            smooth_x = self.prev_x + (x - self.prev_x) * self.smoothing_factor
            smooth_y = self.prev_y + (y - self.prev_y) * self.smoothing_factor
            
            # 将坐标映射到屏幕尺寸
            screen_x = int(smooth_x * self.screen_width)
            screen_y = int(smooth_y * self.screen_height)
            
            pyautogui.moveTo(screen_x, screen_y)
            self.prev_x, self.prev_y = smooth_x, smooth_y
            
        elif gesture == "left_click" and current_time - self.last_click_time > self.click_delay:
            pyautogui.click()
            self.last_click_time = current_time
            
        elif gesture == "double_click" and current_time - self.last_click_time > self.click_delay:
            pyautogui.doubleClick()
            self.last_click_time = current_time
            
        elif gesture == "scroll_down":
            pyautogui.scroll(-self.scroll_speed)
            
        elif gesture == "scroll_up":
            pyautogui.scroll(self.scroll_speed)
            
        elif gesture == "right_click" and current_time - self.last_click_time > self.click_delay:
            pyautogui.rightClick()
            self.last_click_time = current_time
    
    def run(self):
        """运行主循环"""
        print("手势鼠标控制已启动")
        print("手势说明：")
        print("- 食指和拇指捏合：鼠标左键点击")
        print("- 中指和拇指捏合：鼠标滚轮向下")
        print("- 无名指和拇指捏合：鼠标滚轮向上")
        print("- 小指和拇指捏合：鼠标左键双击")
        print("- 无名指和拇指捏合且中指伸直：鼠标右键点击")
        print("- 仅伸出食指（其他手指弯曲）：速度滚动（上下移动控制滚轮）")
        print("- 手掌移动：控制鼠标移动")
        print("按 'q' 键退出程序")
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("无法读取摄像头画面")
                continue
            
            # 转换图像颜色空间
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 处理图像并获取结果
            results = self.hands.process(image)
            
            # 绘制手部关键点
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 检测手势
                    gesture = self.detect_gesture(hand_landmarks)
                    
                    # 获取手部中心点（使用手掌中心而不是手腕）
                    palm_x = (hand_landmarks.landmark[0].x + hand_landmarks.landmark[9].x) / 2
                    palm_y = (hand_landmarks.landmark[0].y + hand_landmarks.landmark[9].y) / 2
                    
                    # 更新位置信息
                    self.prev_palm_x, self.prev_palm_y = palm_x, palm_y
                    self.prev_time = time.time()
                    
                    # 执行对应操作
                    self.execute_action(gesture, palm_x, palm_y)
            
            # 显示图像
            cv2.imshow('MediaPipe手势鼠标', image)
            
            # 按'q'键退出
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    gesture_mouse = GestureMouse()
    gesture_mouse.run()