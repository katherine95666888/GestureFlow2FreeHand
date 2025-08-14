import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import ttk, messagebox
import screeninfo
import numpy as np
from PIL import Image, ImageTk
import logging

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gesture_control.log'),
        logging.StreamHandler()
    ]
)

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
        logging.info(f"Screen size detected: {screen.width}x{screen.height}")
        return screen.width, screen.height
    except Exception as e:
        logging.error(f"Error getting screen size: {e}")
        return 1920, 1080  # Fallback resolution

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()
SCREEN_ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT

# Camera settings (default resolution, will be updated)
CAM_WIDTH, CAM_HEIGHT = 1280, 720
ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 0, 0, CAM_WIDTH, CAM_HEIGHT

# Function to get supported camera resolutions
def get_supported_resolutions(camera_index=0):
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logging.error(f"Camera at index {camera_index} could not be opened")
            return [(1280, 720)]  # Fallback resolution
        resolutions = []
        for width in [1920, 1280, 640]:
            for height in [1080, 720, 480]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (actual_width, actual_height) not in resolutions and actual_width > 0 and actual_height > 0:
                    resolutions.append((actual_width, actual_height))
        cap.release()
        logging.info(f"Supported resolutions for camera {camera_index}: {resolutions}")
        return resolutions if resolutions else [(1280, 720)]
    except Exception as e:
        logging.error(f"Error getting supported resolutions: {e}")
        return [(1280, 720)]

# Gesture class for hand gesture recognition
class Gesture:
    def __init__(self, hand="right"):
        self.hand = hand
        self.gestures = {
            "NUMBER_1": self.detect_number_1,
            "NUMBER_2": self.detect_number_2,
            "NUMBER_3": self.detect_number_3,
            "NUMBER_4": self.detect_number_4,
            "NUMBER_5": self.detect_number_5,
            "OK": self.detect_ok,
            "ORCHID": self.detect_orchid,
            "RAP": self.detect_rap,
            "FIST": self.detect_fist
        }

    def _is_finger_extended(self, tip, pip, mcp, threshold=0.05):
        tip_to_mcp = ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5
        pip_to_mcp = ((pip.x - mcp.x) ** 2 + (pip.y - mcp.y) ** 2) ** 0.5
        return tip_to_mcp > pip_to_mcp + threshold

    def _is_thumb_extended(self, thumb_tip, thumb_ip, wrist):
        if self.hand == "right":
            return thumb_tip.x < thumb_ip.x
        else:
            return thumb_tip.x > thumb_ip.x

    def detect_number_1(self, landmarks):
        thumb_extended = self._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0])
        return (not thumb_extended and
                self._is_finger_extended(landmarks[8], landmarks[6], landmarks[5]) and
                all(not self._is_finger_extended(landmarks[i], landmarks[i-2], landmarks[i-3]) for i in [12, 16, 20]))

    def detect_number_2(self, landmarks):
        thumb_extended = self._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0])
        return (not thumb_extended and
                self._is_finger_extended(landmarks[8], landmarks[6], landmarks[5]) and
                self._is_finger_extended(landmarks[12], landmarks[10], landmarks[9]) and
                all(not self._is_finger_extended(landmarks[i], landmarks[i-2], landmarks[i-3]) for i in [16, 20]))

    def detect_number_3(self, landmarks):
        thumb_extended = self._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0])
        return (not thumb_extended and
                self._is_finger_extended(landmarks[8], landmarks[6], landmarks[5]) and
                self._is_finger_extended(landmarks[12], landmarks[10], landmarks[9]) and
                self._is_finger_extended(landmarks[16], landmarks[14], landmarks[13]) and
                not self._is_finger_extended(landmarks[20], landmarks[18], landmarks[17]))

    def detect_number_4(self, landmarks):
        return (all(self._is_finger_extended(landmarks[i], landmarks[i-2], landmarks[i-3]) for i in [8, 12, 16, 20]) and
                not self._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0]))

    def detect_number_5(self, landmarks):
        return (all(self._is_finger_extended(landmarks[i], landmarks[i-2], landmarks[i-3]) for i in [8, 12, 16, 20]) and
                self._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0]))

    def detect_ok(self, landmarks):
        thumb_tip, index_tip = landmarks[4], landmarks[8]
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        middle_distance = ((thumb_tip.x - landmarks[12].x) ** 2 + (thumb_tip.y - landmarks[12].y) ** 2) ** 0.5
        return distance < 0.05 and middle_distance > 0.1

    def detect_orchid(self, landmarks):
        thumb_tip, middle_tip = landmarks[4], landmarks[12]
        distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
        index_distance = ((thumb_tip.x - landmarks[8].x) ** 2 + (thumb_tip.y - landmarks[8].y) ** 2) ** 0.5
        ring_distance = ((thumb_tip.x - landmarks[16].x) ** 2 + (thumb_tip.y - landmarks[16].y) ** 2) ** 0.5
        return distance < 0.05 and index_distance > 0.1 and ring_distance > 0.1

    def detect_rap(self, landmarks):
        thumb_tip, ring_tip = landmarks[4], landmarks[16]
        distance = ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5
        index_distance = ((thumb_tip.x - landmarks[8].x) ** 2 + (thumb_tip.y - landmarks[8].y) ** 2) ** 0.5
        return distance < 0.05 and index_distance > 0.1

    def detect_fist(self, landmarks):
        thumb_extended = self._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0])
        return not thumb_extended and all(not self._is_finger_extended(landmarks[i], landmarks[i-2], landmarks[i-3]) for i in [8, 12, 16, 20])

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
    def four_finger_swipe(self, direction="up"):
        # Note: PyAutoGUI's hotkey may not work for all systems
        if direction == "up":
            pyautogui.hotkey('ctrl', 'up')
        elif direction == "down":
            pyautogui.hotkey('ctrl', 'down')
        logging.info(f"Four-finger swipe {direction} executed")

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
        self.resolution = tk.StringVar(value="1280x720")
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
        self.gesture = Gesture(hand=self.hand)
        self.mouse = MouseControl(sensitivity=self.sensitivity.get())
        self.touch = TouchScreen()
        self.current_gesture = tk.StringVar(value="No gesture detected")
        self.prev_gesture = None
        self.actual_cam_width = CAM_WIDTH
        self.actual_cam_height = CAM_HEIGHT
        self.cap = None

        # Initial gesture bindings
        self.bindings = {
            "NUMBER_1": (self.mouse.move, "Mouse Move"),
            "NUMBER_5": (self.mouse.move, "Mouse Move"),
            "OK": (self.mouse.left_click, "Left Click"),
            "ORCHID": (self.mouse.right_click, "Right Click"),
            "RAP": (self.mouse.double_click, "Double Click"),
            "FIST": (self.mouse.scroll_up, "Scroll Up"),
            "NUMBER_3": (self.mouse.scroll_down, "Scroll Down")
        }
        self.available_actions = {
            "Mouse Move": self.mouse.move,
            "Left Click": self.mouse.left_click,
            "Right Click": self.mouse.right_click,
            "Double Click": self.mouse.double_click,
            "Scroll Up": self.mouse.scroll_up,
            "Scroll Down": self.mouse.scroll_down,
            "Four Finger Swipe Up": lambda: self.touch.four_finger_swipe("up"),
            "Four Finger Swipe Down": lambda: self.touch.four_finger_swipe("down")
        }

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
        camera_combo.bind("<<ComboboxSelected>>", self.update_resolution_list)

        # Resolution selection
        ttk.Label(control_frame, text="Resolution:").pack(side=tk.LEFT, padx=5)
        self.resolution_combo = ttk.Combobox(control_frame, textvariable=self.resolution, state="readonly")
        self.update_resolution_list(None)
        self.resolution_combo.pack(side=tk.LEFT, padx=5)

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
        self.hints_label = ttk.Label(self.root, text="")
        self.hints_label.pack()
        self.update_hints()
        ttk.Label(self.root, text="Current Gesture:").pack()
        ttk.Label(self.root, textvariable=self.current_gesture, font=("Arial", 12, "bold")).pack()

    def update_resolution_list(self, event):
        try:
            resolutions = get_supported_resolutions(int(self.camera_index.get()))
            resolution_strings = [f"{w}x{h}" for w, h in resolutions]
            self.resolution_combo['values'] = resolution_strings
            if resolution_strings:
                self.resolution.set(resolution_strings[0])
            else:
                self.resolution.set("1280x720")
            logging.info(f"Resolution list updated: {resolution_strings}")
        except Exception as e:
            logging.error(f"Error updating resolution list: {e}")
            messagebox.showerror("Error", f"Failed to update resolution list: {e}")

    def update_hints(self):
        hints = ", ".join([f"{g}={a[1]}" for g, a in sorted(self.bindings.items())])
        self.hints_label.config(text=f"Gestures: {hints}")

    def update_sensitivity(self, val):
        self.mouse.sensitivity = self.sensitivity.get()
        logging.info(f"Sensitivity updated to: {self.sensitivity.get()}")

    def set_hand(self, event):
        self.hand = event.widget.get()
        self.gesture.hand = self.hand
        logging.info(f"Hand preference set to: {self.hand}")

    def start(self):
        self.is_running = True
        if not self.camera_on:
            self.toggle_camera()
        logging.info("Gesture control started")

    def pause(self):
        self.is_running = False
        logging.info("Gesture control paused")

    def toggle_camera(self):
        global CAM_WIDTH, CAM_HEIGHT, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
        if self.camera_on:
            if self.cap:
                try:
                    self.cap.release()
                    logging.info("Camera released successfully")
                except Exception as e:
                    logging.error(f"Error releasing camera: {e}")
                self.cap = None
            self.camera_on = False
            self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Off", fill="white", font=("Arial", 16))
            self.current_gesture.set("No gesture detected")
            logging.info("Camera turned off")
        else:
            try:
                index = int(self.camera_index.get())
                self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.camera_on = True
                    # Set resolution
                    res = self.resolution.get().split('x')
                    CAM_WIDTH, CAM_HEIGHT = int(res[0]), int(res[1])
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                    self.actual_cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.actual_cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if self.actual_cam_width != CAM_WIDTH or self.actual_cam_height != CAM_HEIGHT:
                        messagebox.showwarning("Resolution Mismatch",
                            f"Requested {CAM_WIDTH}x{CAM_HEIGHT}, but camera set to {self.actual_cam_width}x{self.actual_cam_height}")
                        CAM_WIDTH, CAM_HEIGHT = self.actual_cam_width, self.actual_cam_height
                    # Reset ROI to full camera frame
                    ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 0, 0, CAM_WIDTH, CAM_HEIGHT
                    self.roi_x, self.roi_y, self.roi_w, self.roi_h = ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
                    # Resize canvas
                    self.canvas.config(width=CAM_WIDTH, height=CAM_HEIGHT)
                    logging.info(f"Camera opened at index {index} with resolution {CAM_WIDTH}x{CAM_HEIGHT}")
                else:
                    self.cap.release()
                    self.cap = None
                    messagebox.showerror("Error", f"Cannot open camera at index {index}. Try another index or check if the camera is in use.")
                    logging.error(f"Failed to open camera at index {index}")
            except Exception as e:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                messagebox.showerror("Error", f"Failed to initialize camera: {e}")
                logging.error(f"Camera initialization error: {e}")

    def open_binding_settings(self):
        binding_window = tk.Toplevel(self.root)
        binding_window.title("Gesture Bindings")
        gesture_list = list(self.gesture.gestures.keys())
        values = ["None"] + gesture_list
        for action_name in self.available_actions:
            frame = ttk.Frame(binding_window)
            frame.pack(pady=5, fill=tk.X)
            ttk.Label(frame, text=f"{action_name}:").pack(side=tk.LEFT, padx=5)
            combo = ttk.Combobox(frame, values=values, state="readonly")
            old_gestures = [g for g, (_, an) in self.bindings.items() if an == action_name]
            combo.set(', '.join(sorted(old_gestures)) if old_gestures else "None")
            combo.bind("<<ComboboxSelected>>", lambda e, a=action_name: self.update_binding(a, e.widget.get()))
            combo.pack(side=tk.LEFT, padx=5)

    def update_binding(self, action_name, gesture):
        old_gestures = [g for g, (_, an) in self.bindings.items() if an == action_name]
        for og in old_gestures:
            del self.bindings[og]
        if gesture != "None":
            if gesture in self.bindings:
                del self.bindings[gesture]
            self.bindings[gesture] = (self.available_actions[action_name], action_name)
        self.update_hints()
        logging.info(f"Binding updated: {gesture} -> {action_name}")

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
        proposed_w = max(50, event.x - self.roi_x)
        proposed_h = max(50, event.y - self.roi_y)
        self.roi_w = min(proposed_w, CAM_WIDTH - self.roi_x)
        self.roi_h = min(proposed_h, CAM_HEIGHT - self.roi_y)
        # Enforce aspect ratio
        if self.roi_w / self.roi_h > SCREEN_ASPECT:
            self.roi_w = int(self.roi_h * SCREEN_ASPECT)
        else:
            self.roi_h = int(self.roi_w / SCREEN_ASPECT)
        self.roi_canvas.coords(self.roi_rect, self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h)
        self.update_roi_canvas()

    def stop_drag(self, event):
        self.drag_data = {"x": 0, "y": 0}

    def stop_resize(self, event):
        self.resize_data = {"x": 0, "y": 0}

    def apply_roi(self, x, y, w, h):
        try:
            if 0 <= x <= CAM_WIDTH - w and 0 <= y <= CAM_HEIGHT - h and w >= 50 and h >= 50:
                self.roi_x, self.roi_y, self.roi_w, self.roi_h = x, y, w, h
                # Enforce aspect ratio
                if self.roi_w / self.roi_h > SCREEN_ASPECT:
                    self.roi_w = int(self.roi_h * SCREEN_ASPECT)
                else:
                    self.roi_h = int(self.roi_w / SCREEN_ASPECT)
                self.roi_canvas.coords(self.roi_rect, self.roi_x, self.roi_y, self.roi_x + self.roi_w, self.roi_y + self.roi_h)
                logging.info(f"ROI updated: x={x}, y={y}, w={w}, h={h}")
            else:
                messagebox.showerror("Error", "Invalid ROI dimensions")
                logging.error(f"Invalid ROI dimensions: x={x}, y={y}, w={w}, h={h}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply ROI: {e}")
            logging.error(f"Error applying ROI: {e}")

    def update_roi_canvas(self):
        if self.camera_on and self.cap and self.cap.isOpened():
            try:
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
                    logging.error("Failed to read camera frame in ROI canvas")
            except Exception as e:
                self.roi_canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Error", fill="red", font=("Arial", 16))
                logging.error(f"Error updating ROI canvas: {e}")
        else:
            self.roi_canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Off", fill="white", font=("Arial", 16))

    def update_camera(self):
        if self.is_running and self.camera_on and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Optional: Resize frame for performance
                    # frame = cv2.resize(frame, (640, 480))
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    cv2.rectangle(frame, (self.roi_x, self.roi_y),
                                  (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (0, 255, 0), 2)

                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            if handedness.classification[0].label.lower() == self.hand:
                                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                                landmarks = hand_landmarks.landmark
                                gesture = self.gesture.recognize(landmarks)
                                # Visual feedback for finger status
                                thumb_extended = self.gesture._is_thumb_extended(landmarks[4], landmarks[3], landmarks[0])
                                index_extended = self.gesture._is_finger_extended(landmarks[8], landmarks[6], landmarks[5])
                                middle_extended = self.gesture._is_finger_extended(landmarks[12], landmarks[10], landmarks[9])
                                ring_extended = self.gesture._is_finger_extended(landmarks[16], landmarks[14], landmarks[13])
                                pinky_extended = self.gesture._is_finger_extended(landmarks[20], landmarks[18], landmarks[17])

                                cv2.circle(frame, (int(landmarks[4].x * CAM_WIDTH), int(landmarks[4].y * CAM_HEIGHT)), 10, (0, 255, 0) if thumb_extended else (0, 0, 255), -1)
                                cv2.circle(frame, (int(landmarks[8].x * CAM_WIDTH), int(landmarks[8].y * CAM_HEIGHT)), 10, (0, 255, 0) if index_extended else (0, 0, 255), -1)
                                cv2.circle(frame, (int(landmarks[12].x * CAM_WIDTH), int(landmarks[12].y * CAM_HEIGHT)), 10, (0, 255, 0) if middle_extended else (0, 0, 255), -1)
                                cv2.circle(frame, (int(landmarks[16].x * CAM_WIDTH), int(landmarks[16].y * CAM_HEIGHT)), 10, (0, 255, 0) if ring_extended else (0, 0, 255), -1)
                                cv2.circle(frame, (int(landmarks[20].x * CAM_WIDTH), int(landmarks[20].y * CAM_HEIGHT)), 10, (0, 255, 0) if pinky_extended else (0, 0, 255), -1)

                                if gesture:
                                    palm_x = landmarks[0].x
                                    palm_y = landmarks[0].y
                                    in_roi = (self.roi_x / CAM_WIDTH <= palm_x <= (self.roi_x + self.roi_w) / CAM_WIDTH and
                                              self.roi_y / CAM_HEIGHT <= palm_y <= (self.roi_y + self.roi_h) / CAM_HEIGHT)
                                    if in_roi and gesture in self.bindings:
                                        action_func, action_name = self.bindings[gesture]
                                        self.current_gesture.set(f"{gesture}: {action_name}")
                                        if action_name in ["Mouse Move", "Scroll Up", "Scroll Down"]:
                                            if action_name == "Mouse Move":
                                                norm_x = (palm_x * CAM_WIDTH - self.roi_x) / self.roi_w if self.roi_w > 0 else 0
                                                norm_y = (palm_y * CAM_HEIGHT - self.roi_y) / self.roi_h if self.roi_h > 0 else 0
                                                action_func(norm_x, norm_y)
                                            else:
                                                action_func()
                                        else:
                                            if gesture != self.prev_gesture:
                                                action_func()
                                                logging.info(f"Action executed: {action_name}")
                                else:
                                    self.current_gesture.set("No gesture detected")
                    else:
                        self.current_gesture.set("No gesture detected")

                    if gesture:
                        self.prev_gesture = gesture
                    else:
                        self.prev_gesture = None

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.canvas.imgtk = imgtk
                else:
                    self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Error", fill="red", font=("Arial", 16))
                    self.current_gesture.set("No gesture detected")
                    logging.error("Failed to read camera frame")
                    self.toggle_camera()  # Reinitialize camera on failure
            except Exception as e:
                self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Error", fill="red", font=("Arial", 16))
                self.current_gesture.set("No gesture detected")
                logging.error(f"Error in update_camera: {e}")
                self.toggle_camera()  # Reinitialize camera on error
        elif not self.camera_on:
            self.canvas.create_text(CAM_WIDTH // 2, CAM_HEIGHT // 2, text="Camera Off", fill="white", font=("Arial", 16))
            self.current_gesture.set("No gesture detected")

        self.root.after(10, self.update_camera)

    def __del__(self):
        if self.cap:
            try:
                self.cap.release()
                logging.info("Camera released on application exit")
            except Exception as e:
                logging.error(f"Error releasing camera on exit: {e}")
        cv2.destroyAllWindows()
        logging.info("Application closed, resources released")

# Run the application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GestureApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")
