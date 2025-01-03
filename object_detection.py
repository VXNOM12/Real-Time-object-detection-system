import torch
import cv2
import numpy as np
from PIL import Image
import time
import datetime
from pathlib import Path

class ObjectDetector:
    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.25):
        """
        Initialize the object detector
        Args:
            model_path: Path to the YOLOv5 model
            conf_threshold: Confidence threshold for detections
        """
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf_threshold
        
        # Initialize video writer
        self.out = None
        self.recording = False
        
    def process_frame(self, frame):
        """
        Process a single frame and return the frame with detections
        Args:
            frame: Input frame from webcam
        Returns:
            Frame with detection boxes and labels
        """
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detection
        results = self.model(frame_rgb)
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()
        
        # Draw boxes and labels
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name
            class_name = self.model.names[int(cls)]
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return frame
    
    def start_recording(self, output_path='output.mp4', fps=20.0, frame_size=(640, 480)):
        """
        Start recording video with detections
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.recording = True
        
    def stop_recording(self):
        """
        Stop recording video
        """
        if self.out is not None:
            self.out.release()
            self.recording = False
    
    def run_detection(self):
        """
        Run real-time object detection using webcam
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Set frame size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting detection... Press 'q' to quit, 'r' to start/stop recording")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Record if enabled
            if self.recording and self.out is not None:
                self.out.write(processed_frame)
            
            # Display recording indicator
            if self.recording:
                cv2.putText(processed_frame, "Recording", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Object Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.recording:
                    self.start_recording()
                else:
                    self.stop_recording()
        
        # Clean up
        cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run the object detector
    """
    # Initialize and run detector
    detector = ObjectDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()