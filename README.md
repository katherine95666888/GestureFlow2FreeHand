# MediaPipe4Win

这是一个使用MediaPipe和OpenCV在Windows系统上实现手势控制鼠标的项目。

## 功能
- 通过手势识别控制鼠标移动
- 通过手势识别实现鼠标点击
- 通过手势识别实现鼠标滚轮操作

## 安装依赖
```bash
pip install opencv-python mediapipe pyautogui
```

## 使用方法
```bash
python gesture_mouse.py
```

## 手势说明
- 食指和拇指捏合：鼠标左键点击
- 中指和拇指捏合：鼠标滚轮向下
- 无名指和拇指捏合：鼠标滚轮向上
- 小指和拇指捏合：鼠标左键双击
- 手掌移动：控制鼠标移动