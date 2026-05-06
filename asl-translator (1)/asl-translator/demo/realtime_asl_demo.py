#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

# ASL Alphabet classes (29 total) - must match training order
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]
NUM_CLASSES = len(ASL_CLASSES)


class ASLRecognizer:
    
    def __init__(self, model_path=None, device='cuda', use_mediapipe=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_mediapipe = use_mediapipe
        
        # Load model
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        # MediaPipe for hand detection (optional but recommended)
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7
                )
                self.mp_draw = mp.solutions.drawing_utils
                print("MediaPipe hands initialized")
            except ImportError:
                print("MediaPipe not available, using ROI-based detection")
                self.use_mediapipe = False
    
    def _load_model(self, model_path):
       
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        
        if model_path and Path(model_path).exists():
            print(f"✓ Loading trained weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print("⚠ WARNING: No trained model found. Using ImageNet pretrained weights.")
            print("  For best results, train the model first using notebooks/stage1_training.ipynb")
            print("  Then run: python realtime_asl_demo.py --model ../models/asl_mobilenetv2.pth")
            # Load ImageNet weights as fallback (won't classify ASL correctly but demo will run)
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform and predict
        input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = probs.max(1)
        
        prediction = ASL_CLASSES[pred_idx.item()]
        return prediction, confidence.item(), probs[0].cpu().numpy()
    
    def detect_hand_region(self, frame):
        if not self.use_mediapipe:
            return None, None, None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            # Get bounding box from landmarks
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            padding = 40
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            hand_roi = frame[y_min:y_max, x_min:x_max]
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            return hand_roi, bbox, hand_landmarks
        
        return None, None, None


class DemoUI:
   
    
    def __init__(self, window_name="ASL Recognition Demo"):
        self.window_name = window_name
        self.sentence = ""
        self.last_prediction = None
        self.last_add_time = 0
        self.stable_prediction = None
        self.stable_count = 0
        self.fps_history = []
        
    def draw(self, frame, prediction, confidence, fps, hand_bbox=None, hand_landmarks=None):
        
        h, w = frame.shape[:2]
        
       
        if hand_bbox:
            x, y, bw, bh = hand_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
       
        roi_size = 300
        roi_x = w - roi_size - 20
        roi_y = 20
        
        if not hand_bbox:
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), 
                          (0, 255, 0), 3)
            cv2.putText(frame, "Place hand here", (roi_x + 60, roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
       
        cv2.rectangle(frame, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (380, 160), (0, 255, 0), 2)
        
        
        if prediction == 'space':
            display_pred = "[SPACE]"
        elif prediction == 'del':
            display_pred = "[DELETE]"
        elif prediction == 'nothing':
            display_pred = "..."
        else:
            display_pred = prediction
        
       
        if confidence > 0.9:
            color = (0, 255, 0)
        elif confidence > 0.7:
            color = (0, 255, 255) 
        else:
            color = (0, 0, 255)  
        
        cv2.putText(frame, f"Prediction: {display_pred}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
       
        cv2.rectangle(frame, (10, h - 90), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 90), (w - 10, h - 10), (255, 255, 0), 2)
        
        display_sentence = self.sentence if self.sentence else "(spell something!)"
        cv2.putText(frame, f"Text: {display_sentence}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
       
        cv2.putText(frame, "Q=quit | C=clear | SPACE=add space | B=backspace | ENTER=confirm letter",
                    (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame, (roi_x, roi_y, roi_size)
    
    def update_sentence(self, prediction, confidence, threshold=0.85, hold_frames=10):
        if prediction == self.stable_prediction:
            self.stable_count += 1
        else:
            self.stable_prediction = prediction
            self.stable_count = 1
        
       
        current_time = time.time()
        if (self.stable_count >= hold_frames and 
            confidence > threshold and
            prediction not in ['nothing'] and
            current_time - self.last_add_time > 1.0):
            
            if prediction == 'space':
                self.sentence += ' '
            elif prediction == 'del':
                self.sentence = self.sentence[:-1] if self.sentence else ""
            else:
                self.sentence += prediction
            
            self.last_add_time = current_time
            self.stable_count = 0
            return True
        
        return False
    
    def handle_key(self, key):
       
        if key == ord('q') or key == 27: 
            return False
        elif key == ord('c'):
            self.sentence = ""
        elif key == ord(' '):
            self.sentence += ' '
        elif key == ord('b') or key == 8: 
            self.sentence = self.sentence[:-1] if self.sentence else ""
        elif key == 13: 
            self.last_add_time = 0 
        return True


def main():
    parser = argparse.ArgumentParser(description='Real-Time ASL Recognition Demo')
    parser.add_argument('--model', type=str, default='../models/asl_mobilenetv2.pth',
                        help='Path to trained model weights')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU, use CPU only')
    parser.add_argument('--use-mediapipe', action='store_true',
                        help='Use MediaPipe for hand detection')
    parser.add_argument('--threshold', type=float, default=0.80,
                        help='Confidence threshold for auto-adding letters (default: 0.80)')
    args = parser.parse_args()
    
   
    device = 'cpu' if args.no_gpu else 'cuda'
    
   
    print("\n" + "="*60)
    print("  ASL ALPHABET RECOGNITION DEMO")
    print("  CSE 474 - Machine Learning, Spring 2026")
    print("  Team: Nitin, Yash, Vanshaj")
    print("="*60)
    
    recognizer = ASLRecognizer(
        model_path=args.model,
        device=device,
        use_mediapipe=args.use_mediapipe
    )
    
   
    ui = DemoUI()
    
   
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Try: python realtime_asl_demo.py --camera 1")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n✓ Camera opened successfully!")
    print("\nControls:")
    print("  - Place your hand in the green box")
    print("  - Make ASL alphabet signs")
    print("  - Letters auto-add when held steady with high confidence")
    print("  - Press Q to quit")
    print("  - Press C to clear text")
    print("  - Press SPACE to add space")
    print("  - Press B for backspace")
    print("-"*60 + "\n")
    
    fps_history = []
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)  
            
            
            hand_roi, hand_bbox, hand_landmarks = recognizer.detect_hand_region(frame)
            
            
            if hand_roi is None:
                h, w = frame.shape[:2]
                roi_size = 300
                roi_x = w - roi_size - 20
                roi_y = 20
                hand_roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
          
            prediction = "nothing"
            confidence = 0.0
            
            if hand_roi is not None and hand_roi.size > 0:
                prediction, confidence, _ = recognizer.predict(hand_roi)
                
               
                ui.update_sentence(prediction, confidence, threshold=args.threshold)
            
           
            fps = 1.0 / (time.time() - start_time + 1e-6)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
          
            frame, _ = ui.draw(frame, prediction, confidence, avg_fps, hand_bbox, hand_landmarks)
            
          
            cv2.imshow(ui.window_name, frame)
            
           
            key = cv2.waitKey(1) & 0xFF
            if not ui.handle_key(key):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{'='*60}")
        print(f"Final text: {ui.sentence}")
        print(f"{'='*60}")
        print("Demo ended. Thank you!")


if __name__ == "__main__":
    main()
