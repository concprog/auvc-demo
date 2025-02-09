# haar cascade detector
import stat
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import numpy as np
import mediapipe as mp

#Load image:
class HaarHandDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier('data/hand.xml')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(hands) > 0:
            return {i: hand for i, hand in enumerate(hands)}

class MediaPipeHandDetector:
    def __init__(self):
        self.detector = mp.solutions.hands.Hands(static_image_mode=True)

    def detect(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        results = self.detector.process(image)
        if results.multi_hand_landmarks:
            output = {}
            for i, hand_landmark in enumerate(results.multi_hand_landmarks):
                x_coords = [landmark.x * image_width for landmark in hand_landmark.landmark]
                y_coords = [landmark.y * image_height for landmark in hand_landmark.landmark]

                xmin = int(min(x_coords))
                ymin = int(min(y_coords))
                xmax = int(max(x_coords))
                ymax = int(max(y_coords))

                box_width = xmax - xmin
                box_height = ymax - ymin

                output[i] = (xmin-20, ymin-20, box_width+30, box_height+30)
            return output

    def landmarks(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
class FlorenceDetector:
    model = None
    processor = None

    @staticmethod
    def load():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if FlorenceDetector.model is None or FlorenceDetector.processor is None:
            model_id = 'microsoft/Florence-2-base-ft'
            FlorenceDetector.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).to(device)
            FlorenceDetector.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return FlorenceDetector.model, FlorenceDetector.processor

    def __init__(self):
        self.model, self.processor = FlorenceDetector.load()
    
    def _run_model(self, image, task_prompt, text_input=''):

        prompt = task_prompt + text_input
        image_height, image_width, _ = image.shape

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            early_stopping=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_height, image_width),
        )
        return parsed_answer

    def _get_polygons(self,prediction):

        output = {}

        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            for polygon in polygons:
                polygon = np.array(polygon).flatten().tolist()
                if len(polygon) < 6:  # Less than 3 points
                    continue
                output[label] = polygon
        return output
  

    def detect(self, frame, objects = ["hands"]):
        input_prompt = ". ".join(objects) if len(objects) > 1 else objects[0]
        results = self._run_model(frame, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>', text_input=input_prompt)
        bboxes, labels = results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'], results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
        output = {}
        for bbox, label in zip(bboxes, labels):
            if label in objects:
                output[label] = bbox
        return output

    def detect_all(self, frame):
        results = self._run_model(frame, task_prompt='<OD>')
        bboxes, labels = results['<OD>']['bboxes'], results['<OD>']['labels']
        output = {}
        for bbox, label in zip(bboxes, labels):
            output[label] = bbox
        return output
    def segment(self, frame, objects = ["hands"]):
        input_prompt = ". ".join(objects) if len(objects) > 1 else objects[0]
        results = self._run_model(frame, task_prompt='<REFERRING_EXPRESSION_SEGMENTATION>',  text_input=input_prompt)
        return results['<REFERRING_EXPRESSION_SEGMENTATION>']

if __name__ == "__main__":
    detector = FlorenceDetector()
    object_to_detect = 'human hand'
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        
        hands = detector.detect(frame, objects=[object_to_detect])
        if hands is not None:
            for label in hands:
                x, y, w, h = hands[label]
                x, y, w, h = int(x), int(y), int(w), int(h)          
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # segments = detector.segment(frame, objects=["human face"])
        # if "human face" in segments:
        #     # cv2.polylines(frame, [segments["face"]], True, (0, 255, 0), 2)
        #     print(segments["human face"])

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if  key == ord('s'):
            object_to_detect = input("Object: ")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


