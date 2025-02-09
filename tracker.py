from typing import Callable
import cv2
import numpy as np


def box_to_point(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


class ObjectTracker:
    def __init__(self, deltaX=20) -> None:
        self.tracker = cv2.TrackerKCF_create()
        self.last_bbox = None
        self.deltaX = deltaX
        self.distVec = 0, 0
        self.velocity = 0
        self.motion_callback = None

    def on_motion(self, cb: Callable):
        self.motion_callback = cb

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return None
        p1 = box_to_point(self.last_bbox)
        p2 = box_to_point(bbox)
        dist = np.linalg.norm(np.array(p1) - np.array(p2))

        if dist > self.deltaX:
            self.distVec = p2[0] - p1[0], p2[1] - p1[1]
            if self.motion_callback is not None:
                self.motion_callback(*self.distVec)

        self.last_bbox = bbox
        return bbox


if __name__ == "__main__":
    from pynput.mouse import Controller

    tracker = ObjectTracker(10)
    con = Controller()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Ask user to select ROI
    bbox = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")

    tracker.on_motion(con.move)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = tracker.track(frame)
        if bbox is not None:
            p1 = box_to_point(tracker.last_bbox)
            p2 = box_to_point(bbox)
            cv2.arrowedLine(frame, p1, p2, (0, 255, 0), 2)
            print(tracker.distVec)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
