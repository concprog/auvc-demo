import cv2
from pynput import keyboard, mouse
from pynput.mouse import Controller, Button
from pynput.keyboard import Controller as KeyboardController

import detectors
from tracker import ObjectTracker as tracker
from nets import GestureRecognizer

# ----------------------------------------------------------------
# Set up the key and mouse controllers and define the gesture actions.
# ----------------------------------------------------------------
gestures = {}
mouse_controller = mouse.Controller()
kb = KeyboardController()

def press_keys(keyboard: KeyboardController, keycodes: list):
    for key in keycodes:
        keyboard.press(key)
        keyboard.release(key)

gestures['okay'] = lambda: mouse_controller.click(Button.left, 1)
gestures['fox'] = lambda: mouse_controller.press(Button.left)
gestures['open fox'] = lambda: mouse_controller.release(Button.left)
gestures['w'] = lambda: press_keys(kb, ['w'])
gestures['a'] = lambda: press_keys(kb, ['a'])
gestures['d'] = lambda: press_keys(kb, ['d'])
gestures['l'] = lambda: press_keys(kb, ['l'])
gestures['u'] = lambda: press_keys(kb, ['u'])
gestures['perspective_view'] = lambda: press_keys(kb, ['Shift', '7'])
gestures['top_view'] = lambda: press_keys(kb, ['Shift', '5'])
gestures['sketch'] = lambda: press_keys(kb, ['Shift', 's'])
gestures['extrude'] = lambda: press_keys(kb, ['Shift', 'e'])

# ----------------------------------------------------------------#
# Initialize the hand detector, tracker, and gesture recognizer.  #
# ----------------------------------------------------------------#
hand_detector = detectors.MediaPipeHandDetector()
hand_tracker = tracker()

gesture_recognizer = GestureRecognizer(
    model_path="gesture_model_final.pth", 
    class_names=list(gestures.keys()),
    device="cpu"
)

# -------------------------------------------------#
# Open the video capture (e.g. from the webcam)    #
# -------------------------------------------------#
cap = cv2.VideoCapture(0)
tracking_bbox = None

while True:
    ret, frame = cap.read()
    cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    if not ret:
        break


    # ---------------------------------------------------------
    # (1) On the first frame—or if the tracker loses the hand—
    # run the detector to find a hand.
    # --------------------------------------------------------
    if tracking_bbox is None:
        detections = hand_detector.detect(frame)
        if detections is not None:
            # For simplicity, take the first detected hand.
            for _, bbox in detections.items():
                tracking_bbox = bbox 
                hand_tracker.init(frame, tracking_bbox)
                break
        continue
    else:
        # -----------------------------------------------------
        # (2) Use the tracker to update the bounding box.
        # If tracking fails, tracking boox is None (remember I did that in ObjectTracker)
        # -----------------------------------------------------
        if frame.shape != (480, 640, 3):
            # Re-detect if frame size has changed
            tracking_bbox = None
            print(frame.shape)
        else:
            new_bbox = hand_tracker.track(frame)
            if new_bbox is None:
                tracking_bbox = None
            else:
                tracking_bbox = new_bbox

    # --------------------------------------------------------
    # (3) If we have a valid bounding box, crop the ROI and run gesture recognition.
    # --------------------------------------------------------
    if tracking_bbox is not None:
        x, y, w, h = map(int, tracking_bbox)
        # Draw a green rectangle around the detected/tracked hand. (this is necessary as I forgot to remove it in the training data)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue
        roi = frame[y:y+h, x:x+w]
        if roi.size:
            gesture_name, probs = gesture_recognizer.predict(roi)
            cv2.putText(frame, f"Gesture: {gesture_name}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            # if gesture_name in gestures:
            #     gestures[gesture_name]()
        cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
