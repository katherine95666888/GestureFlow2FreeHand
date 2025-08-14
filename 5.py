import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import ttk, messagebox
import screeninfo
import numpy as np
from PIL import Image, ImageTk

# Initialize PyAutoGUI with safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
def get_screen_size():
    try:
        screen = screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except Exception as e:
        print(f"Error getting screen size: {e}")
        return 1920, 1080  # Fallback resolution

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()

# Camera settings (higher resolution)
CAM_WIDTH, CAM_HEIGHT = 1280, 720
ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 0, 0, CAM_WIDTH, CAM_HEIGHT

# Gesture class for hand gesture recognition
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
        return (landmarks[8].y < landmarks[6].y and
                all(landmarks[i].y > landmarks[i-2].y for i in [12, 16, 20]))

    def detect_number_2(self, landmarks):
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                all(landmarks[i].y > landmarks[i-2].y for i in [16, 20]))

    def detect_number_3(self, landmarks):
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                landmarks[16].y < landmarks[14].y and
                landmarks[20].y > landmarks[18].y)

    def detect_number_4(self, landmarks):
        return (all(landmarks[i].y < landmarks[i-2].y for i in [8, 12, 16, 20]) and
                landmarks[4].x > landmarks[3].x)

    def detect_number_5(self, landmarks):
        return (all(landmarks[i].y < landmarks[i-2].y for i in [8, 12, 16, 20]) and
                landmarks[4].x < landmarks[3].x)

    def detect_number_6(self, landmarks):
        return (landmarks[4].x < landmarks[3].x and
                landmarks[8].y < landmarks[6].y and
                all(landmarks[i].y > landmarks[i-2].y for i in [12, 16, 20]))

    def detect_ok(self, landmarks):
        thumb_tip, ring_tip = landmarks[4], landmarks[16]
        distance = ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5
        index_distance = ((thumb_tip.x - landmarks[8].x) ** 2 + (thumb_tip.y - landmarks[8].y) ** 2) ** 0.5
        return distance < 0.05 and index_distance > 0.1

    def detect_orchid(self, landmarks):
        thumb_tip, middle_tip = landmarks[4], landmarks[12]
        distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
        index_distance = ((thumb_tip.x - landmarks[8].x) ** 2 + (thumb_tip.y - landmarks[8].y) ** 2) ** 0.5
        ring_distance = ((thumb_tip.x - landmarks[16].x) ** 2 + (thumb_tip.y - landmarks[16].y) ** 2) ** 0.5
        return distance < 0.05 and index_distance > 0.1 and ring_distance > 0.1

    def detect_rap(self, landmarks):
        thumb_tip, index_tip = landmarks[4], landmarks[8]
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        middle_distance = ((thumb_tip.x - landmarks[12].x) ** 2 + (thumb_tip.y - landmarks[12].y) ** 2) ** 0.5
        return distance < 0.05 and middle_distance > 0.1

    def detect_fist(self, landmarks):
        return all(landmarks[i].y > landmarks[i-2].y for i in [8, 12, 16, 20])

    def recognize(self, landmarks):
        for gesture_name, detect_func in self.gestures.items():
            if detect_func(landmarks):
                return gesture_name
        return None

# Mouse control class
class MouseControl:
    def __init__(self, sensitivity=1.0):
        self.sensitivity = sensitivity

    def move(self, x, y):
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

# Touchscreen class
class TouchScreen:
    def three_finger_swipe(self, direction="up"):
        if direction == "up":
            pyautogui.hotkey('ctrl', 'up')
        elif direction == "down":
            pyautogui.hotkey('ctrl', 'down')

# Main application with Tkinter frontend
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Control")
        self.is_running = False
        self.camera_on = False
        self.hand = "right"
        self.sensitivity = tk.DoubleVar(value=1.0)
        self.camera_index = tk.StringVar(value="0")
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
        self.gesture = Gesture()
        self.mouse = MouseControl(sensitivity=self.sensitivity.get())
        self.touch = TouchScreen()
        self.current_gesture = tk.StringVar(value="No gesture detected")

        # Default gesture bindings
        self.bindings = {
            "NUMBER_5": (self.mouse.move, "Mouse Move"),
            "OK": (self.mouse.left_click, "Left Click"),
            "ORCHID": (self.mouse.right_click, "Right Click"),
            "RAP": (self.mouse.double_click, "Double Click"),
            "FIST": (self.mouse.scroll_up, "Scroll Up"),
            "NUMBER_6": (self.mouse.scroll_down, "Scroll Down")
        }
        self.available_actions = {
            "Mouse Move": self.mouse.move,
            "Left Click": self.mouse.left_click,
            "Right Click": self.mouse.right_click,
            "Double Click": self.mouse.double_click,
            "Scroll Up": self.mouse.scroll_up,
            "Scroll Down": self.mouse.scroll_down,
            "Three Finger Swipe Up": lambda: self.touch.three_finger_swipe("up"),
            "Three Finger Swipe Down": lambda: self.touch.three_finger_swipe("down")
        }

        self.cap = None
        self.setup_main_ui()
        self.toggle_camera()
        self.update_camera()

    def setup_main_ui(self):
        # Camera display
        self.canvas = tk.Canvas(self.root, width=CAM_WIDTH, height=CAM_HEIGHT, bg="black")
        self.canvas.pack()

        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        # Start/Pause buttons
        ttk.Button(control_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Pause", command=self.pause).pack(side=tk.LEFT, padx=5)

        # Camera toggle
        ttk.Button(control_frame, text="Toggle Camera", command=self.toggle_camera).pack(side=tk.LEFT, padx=5)

        # Camera index selection
        ttk.Label(control_frame, text="Camera Index:").pack(side=tk.LEFT, padx=5)
        camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_index, values=["0", "1", "2"], state="readonly")
        camera_combo.pack(side=tk.LEFT, padx=5)

        # Sensitivity slider
        ttk.Label(control_frame, text="Sensitivity:").pack(side=tk.LEFT, padx=5)
        ttk.Scale(control_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, variable=self.sensitivity,
                  command=self.update_sensitivity).pack(side=tk.LEFT, padx=5)

        # Hand selection
        ttk.Label(control_frame, text="Hand:").pack(side=tk.LEFT, padx=5)
        hand_combo = ttk.Combobox(control_frame, values=["right", "left"], state="readonly")
        hand_combo.set(self.hand)
        hand_combo.bind("<<ComboboxSelected>>", self.set_hand)
        hand_combo.pack(side=tk.LEFT, padx=5)

        # ROI and binding settings
        ttk.Button(control_frame, text="Set ROI", command=self.open_roi_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Set Bindings", command=self.open_binding_settings).pack(side=tk.LEFT, padx=5)

        # Notes
        ttk.Label(control_frame, text="Notes:").pack(side=tk.LEFT, padx=5)
        self.notes = ttk.Entry(control_frame)
        self.notes.pack(side=tk.LEFT, padx=5)

        # Gesture hints and current gesture
        ttk.Label(self.root, text="Gestures: 5=Move, OK=Left Click, Orchid=Right Click, Rap=Double Click, Fist=Scroll Up, 6=Scroll Down").pack()
        ttk.Label(self.root, text="Current Gesture:").pack()
        ttk.Label(self.root, textvariable=self.current_gesture, font=("Arial", 12, "bold")).pack()

    def update_sensitivity(self, val):
        self.mouse.sensitivity = self.sensitivity.get()

    def set_hand(self, event):
        self.hand = event.widget.get()

    def start(self):
        self.is_running = True
        if not self.camera_on:
            self.toggle_camera()

    def pause(self):
        self.is_running = False

    def toggle_camera(self):
        if self.camera_on:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.camera_on = False
            self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Off", fill="white", font=("Arial", 16))
            self.current_gesture.set("No gesture detected")
        else:
            index = int(self.camera_index.get())
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                self.camera_on = True
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if actual_width != CAM_WIDTH or actual_height != CAM_HEIGHT:
                    print(f"Warning: Set {CAM_WIDTH}x{CAM_HEIGHT}, got {actual_width}x{actual_height}")
            else:
                messagebox.showerror("Error", f"Cannot open camera at index {index}. Try another index.")
                self.cap = None

    def open_binding_settings(self):
        binding_window = tk.Toplevel(self.root)
        binding_window.title("Gesture Bindings")
        for gesture, (action, action_name) in self.bindings.items():
            frame = ttk.Frame(binding_window)
            frame.pack(pady=5, fill=tk.X)
            ttk.Label(frame, text=f"{gesture}:").pack(side=tk.LEFT, padx=5)
            combo = ttk.Combobox(frame, values=list(self.available_actions.keys()), state="readonly")
            combo.set(action_name)
            combo.bind("<<ComboboxSelected>>", lambda e, g=gesture: self.update_binding(g, e.widget.get()))
            combo.pack(side=tk.LEFT, padx=5)

    def update_binding(self, gesture, action_name):
        self.bindings[gesture] = (self.available_actions[action_name], action_name)
        hints = ", ".join([f"{g}={a[1]}" for g, a in self.bindings.items()])
        self.root.winfo_children()[-2].config(text=f"Gestures: {hints}")

    def open_roi_settings(self):
        roi_window = tk.Toplevel(self.root)
        roi_window.title("Set ROI")
        self.roi_canvas = tk.Canvas(roi_window, width=CAM_WIDTH, height=CAM_HEIGHT, bg="black")
        self.roi_canvas.pack()

        # Numerical inputs
        control_frame = ttk.Frame(roi_window)
        control_frame.pack(pady=10)
        ttk.Label(control_frame, text="X:").pack(side=tk.LEFT)
        x_entry = ttk.Entry(control_frame, width=5)
        x_entry.insert(0, self.roi_x)
        x_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Y:").pack(side=tk.LEFT)
        y_entry = ttk.Entry(control_frame, width=5)
        y_entry.insert(0, self.roi_y)
        y_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Width:").pack(side=tk.LEFT)
        w_entry = ttk.Entry(control_frame, width=5)
        w_entry.insert(0, self.roi_w)
        w_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Height:").pack(side=tk.LEFT)
        h_entry = ttk.Entry(control_frame, width=5)
        h_entry.insert(0, self.roi_h)
        h_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Apply", command=lambda: self.apply_roi(
            int(x_entry.get()), int(y_entry.get()), int(w_entry.get()), int(h_entry.get())
        )).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset ROI", command=lambda: self.apply_roi(0, 0, CAM_WIDTH, CAM_HEIGHT)).pack(side=tk.LEFT, padx=5)

        # Draggable and resizable rectangle
        self.roi_rect = self.roi_canvas.create_rectangle(
            self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h,
            outline="green", width=2
        )
        self.roi_canvas.bind("<Button-1>", self.start_drag)
        self.roi_canvas.bind("<B1-Motion>", self.drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.stop_drag)
        self.roi_canvas.bind("<Button-3>", self.start_resize)
        self.roi_canvas.bind("<B3-Motion>", self.resize)
        self.roi_canvas.bind("<ButtonRelease-3>", self.stop_resize)
        self.drag_data = {"x": 0, "y": 0}
        self.resize_data = {"x": 0, "y": 0}
        self.update_roi_canvas()

    def start_drag(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def drag(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.roi_x += dx
        self.roi_y += dy
        self.roi_x = max(0, min(self.roi_x, CAM_WIDTH - self.roi_w))
        self.roi_y = max(0, min(self.roi_y, CAM_HEIGHT - self.roi_h))
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.roi_canvas.coords(self.roi_rect, self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h)
        self.update_roi_canvas()

    def start_resize(self, event):
        self.resize_data["x"] = event.x
        self.resize_data["y"] = event.y

    def resize(self, event):
        self.roi_w = max(50, event.x - self.roi_x)
        self.roi_h = max(50, event.y - self.roi_y)
        self.roi_w = min(self.roi_w, CAM_WIDTH - self.roi_x)
        self.roi_h = min(self.roi_h, CAM_HEIGHT - self.roi_y)
        self.roi_canvas.coords(self.roi_rect, self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h)
        self.update_roi_canvas()

    def stop_drag(self, event):
        self.drag_data = {"x": 0, "y": 0}

    def stop_resize(self, event):
        self.resize_data = {"x": 0, "y": 0}

    def apply_roi(self, x, y, w, h):
        if 0 <= x <= CAM_WIDTH - w and 0 <= y <= CAM_HEIGHT - h and w >= 50 and h >= 50:
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = x, y, w, h
            self.roi_canvas.coords(self.roi_rect, self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h)
        else:
            messagebox.showerror("Error", "Invalid ROI dimensions")

    def update_roi_canvas(self):
        if self.camera_on and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.roi_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.roi_canvas.imgtk = imgtk
                self.roi_canvas.coords(self.roi_rect, self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h)
            else:
                self.roi_canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Error", fill="red", font=("Arial", 16))
        else:
            self.roi_canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Off", fill="white", font=("Arial", 16))

    def update_camera(self):
        if self.is_running and self.camera_on and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                print(f"Frame size: {frame.shape}")  # Debug frame size
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                cv2.rectangle(frame, (self.roi_x, self.roi_y),
                             (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (0, 255, 0), 2)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        if handedness.classification[0].label.lower() == self.hand:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            gesture = self.gesture.recognize(hand_landmarks.landmark)
                            if gesture:
                                self.current_gesture.set(f"{gesture}: {self.bindings.get(gesture, (None, 'Unbound'))[1]}")
                                palm_x = hand_landmarks.landmark[0].x
                                palm_y = hand_landmarks.landmark[0].y
                                if (self.roi_x / CAM_WIDTH <= palm_x <= (self.roi_x + self.roi_w) / CAM_WIDTH and
                                    self.roi_y / CAM_HEIGHT <= palm_y <= (self.roi_y + self.roi_h) / CAM_HEIGHT):
                                    if gesture == "NUMBER_5":
                                        norm_x = (palm_x * CAM_WIDTH - self.roi_x) / self.roi_w if self.roi_w > 0 else 0
                                        norm_y = (palm_y * CAM_HEIGHT - self.roi_y) / self.roi_h if self.roi_h > 0 else 0
                                        self.bindings[gesture][0](norm_x, norm_y)
                                    elif gesture in self.bindings:
                                        self.bindings[gesture][0]()
                            else:
                                self.current_gesture.set("No gesture detected")
                else:
                    self.current_gesture.set("No gesture detected")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk
            else:
                self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Error", fill="red", font=("Arial", 16))
                self.current_gesture.set("No gesture detected")
        elif not self.camera_on:
            self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Off", fill="white", font=("Arial", 16))
            self.current_gesture.set("No gesture detected")

        self.root.after(10, self.update_camera)

    def __del__(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
