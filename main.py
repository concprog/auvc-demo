import cv2
import numpy as np
from pynput.mouse import Controller, Button
from pynput.keyboard import Controller as KeyboardController

import detectors
from tracker import ObjectTracker as tracker
from nets import GestureRecognizer

# ---------------------------------------------------------------------#
# Set up the key and mouse controllers and define the gesture actions. #
# ---------------------------------------------------------------------#
gestures = {}
mouse = Controller()
kb = KeyboardController()

def press_keys(keyboard: KeyboardController, keycodes: list):
    for key in keycodes:
        keyboard.press(key)
        keyboard.release(key)

gestures['okay'] = lambda: mouse.click(Button.left, 1)
gestures['fox'] = lambda: mouse.press(Button.left)
gestures['open fox'] = lambda: mouse.release(Button.left)
gestures['w'] = lambda: press_keys(kb, ['w'])
gestures['a'] = lambda: press_keys(kb, ['a'])
gestures['d'] = lambda: press_keys(kb, ['d'])
gestures['l'] = lambda: press_keys(kb, ['l'])
gestures['u'] = lambda: press_keys(kb, ['u'])
gestures['perspective_view'] = lambda: press_keys(kb, ['Shift', '7'])
gestures['neutral'] = lambda: print('Neutral')
gestures['sketch'] = lambda: press_keys(kb, ['Shift', 's'])
gestures['extrude'] = lambda: press_keys(kb, ['Shift', 'e'])

# ----------------------------------------------------------------#
# Initialize the hand detector, tracker, and gesture recognizer.  #
# ----------------------------------------------------------------#
hand_detector = detectors.MediaPipeHandDetector()
hand_tracker = tracker()

gesture_recognizer = GestureRecognizer(
    model_path="best_gesture_model_fp16.pth", 
    class_names=list(gestures.keys()),
    device="cpu",
    use_fp16=True
)

# -------------------------------------------------#
# Open the video capture (e.g. from the webcam)    #
# -------------------------------------------------#
def box_to_point(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


delta = 20  
last_bbox = None  


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    
    # ---------------------------------------------------------#
    # Run detector on each frame.                              #
    # ---------------------------------------------------------#
    detections = hand_detector.detect(frame)
    if detections is not None:
        for label, bbox in detections.items():
            bbox = list(map(int, bbox))
            
            # Calculate motion displacement
            if last_bbox is not None:
                p1 = box_to_point(last_bbox)
                p2 = box_to_point(bbox)
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist > delta:
                    distVec = map((p2[0] - p1[0], p2[1] - p1[1]), lambda x: x * 4)
                    mouse.move(*distVec)

            last_bbox = bbox
            
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or x < 0 or y < 0 or (x+w) > frame.shape[1] or (y+h) > frame.shape[0]:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            

            roi = frame[y:y+h, x:x+w]
            if roi.size != 0:
                gesture_name, probs = gesture_recognizer.predict(roi)
                cv2.putText(frame, f"Gesture: {gesture_name}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            break 
    else:
        last_bbox = None

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
