import os
import uuid
import cv2
import detectors

def save_gesture(name: str, frame):
    """Saves a cropped hand image to data/gestures/{name}/frame_{uuid}.png"""
    dir_path = f"data/gestures/{name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = f"{dir_path}/frame_{uuid.uuid4().hex}.png"
    cv2.imwrite(filename, frame)

# Gesture actions dictionary for reference (here used only for naming)
gestures = {
    'okay': lambda: print("Gesture: okay"),
    'fox': lambda: print("Gesture: fox"),
    'open fox': lambda: print("Gesture: open fox"),
    'w': lambda: print("Gesture: w"),
    'a': lambda: print("Gesture: a"),
    'd': lambda: print("Gesture: d"),
    'l': lambda: print("Gesture: l"),
    'u': lambda: print("Gesture: u"),
    'perspective_view': lambda: print("Gesture: perspective_view"),
    'neutral': lambda: print("Gesture: neutral"),
    'sketch': lambda: print("Gesture: sketch"),
    'extrude': lambda: print("Gesture: extrude")
}

# List of gesture names to cycle through
gesture_names = list(gestures.keys())
current_gesture_index = 0

if __name__ == "__main__":
    detector = detectors.MediaPipeHandDetector()
    cap = cv2.VideoCapture(0)
    cropped_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        cropped_frame = None  # Reset for each new frame

        hands = detector.detect(frame)
        if hands:
            for label in hands:
                x, y, w, h = map(int, hands[label])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                hand_crop = frame[y:y+h, x:x+w]
                if hand_crop.size:
                    cropped_frame = hand_crop
                    cv2.imshow(f'hand_{label}', hand_crop)
                    break
                
        current_gesture_name = gesture_names[current_gesture_index]
        cv2.putText(frame, f"Gesture: {current_gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)

        # Key event handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if cropped_frame is not None:
                save_gesture(current_gesture_name, cropped_frame)
                print(f"Saved gesture '{current_gesture_name}'")
            else:
                print("No hand detected to record.")
        elif key == ord('t'):
            current_gesture_index = (current_gesture_index + 1) % len(gesture_names)
            print(f"Switched to gesture: {gesture_names[current_gesture_index]}")

    cap.release()
    cv2.destroyAllWindows()
