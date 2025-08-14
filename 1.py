import cv2
import mediapipe as mp
import pyautogui

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()

# 摄像头分辨率设定（尽量与你摄像头实际分辨率一致）
cam_width, cam_height = 640, 480

# 操作区域（摄像头画面中的框）
box_width, box_height = 400, 300
box_x = (cam_width - box_width) // 2
box_y = (cam_height - box_height) // 2

# 初始化摄像头和MediaPipe
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 翻转镜像，方便操作
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 画操作框
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x = int(hand_landmarks.landmark[8].x * w)  # 食指尖
        y = int(hand_landmarks.landmark[8].y * h)

        # 画点
        cv2.circle(frame, (x, y), 10, (0, 0, 255), cv2.FILLED)

        # 判断是否在操作框内
        if box_x <= x <= box_x + box_width and box_y <= y <= box_y + box_height:
            # 映射到屏幕坐标
            relative_x = x - box_x
            relative_y = y - box_y
            screen_x = int(relative_x / box_width * screen_width)
            screen_y = int(relative_y / box_height * screen_height)

            # 控制鼠标移动
            pyautogui.moveTo(screen_x, screen_y)
            cv2.putText(frame, f"Mouse: ({screen_x},{screen_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
