
---

# Real-Time Object Detection with YOLOv5

This project demonstrates a real-time object detection system using YOLOv5 and OpenCV. The script captures live video from a webcam, processes each frame to detect objects, and displays the results with bounding boxes and class labels. It also supports video recording with object detection overlays.

## Features

- **Real-Time Object Detection**: Detect objects in live video using YOLOv5.
- **Bounding Box Visualization**: Draw bounding boxes and display class labels for detected objects.
- **Video Recording**: Record real-time video with detection overlays.
- **Timestamp Overlay**: Display the current timestamp on each frame.
- **Keyboard Controls**:
  - Press **`q`** to quit the program.
  - Press **`r`** to toggle video recording.

## Prerequisites

Before running the script, ensure the following dependencies are installed:

- Python 3.9+
- OpenCV
- PyTorch
- YOLOv5 model (default: `yolov5s.pt`)

Install the required Python libraries:
```bash
pip install torch torchvision opencv-python numpy pillow
```

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VXNOM12/object-detection.git
   cd object-detection
   ```

2. **Download YOLOv5 Model**:
   - Download the YOLOv5 model file (`yolov5s.pt`) from the [official repository](https://github.com/ultralytics/yolov5) or use a custom-trained model.

3. **Run the Script**:
   ```bash
   python object_detection.py
   ```

4. **Controls**:
   - Press **`r`** to start/stop recording.
   - Press **`q`** to quit the application.

## File Structure

```
.
├── object_detection.py  # Main script for object detection
├── yolov5s.pt           # Pre-trained YOLOv5 model file
└── README.md            # Project documentation
```

## How It Works

1. **Model Initialization**:
   - The YOLOv5 model is loaded using PyTorch Hub.

2. **Frame Processing**:
   - Each frame is captured from the webcam and converted to RGB.
   - The YOLOv5 model detects objects and returns bounding boxes, labels, and confidence scores.
   - Bounding boxes and labels are drawn on the frame.

3. **Video Recording**:
   - When recording is toggled, processed frames are saved to an output video file (`output.mp4`).

## Customization

- **Confidence Threshold**:
  - Adjust the confidence threshold for detections in the `ObjectDetector` class:
    ```python
    detector = ObjectDetector(conf_threshold=0.5)
    ```

- **Frame Size**:
  - Modify the frame size for the webcam in `cap.set()` calls.

- **Output Video Settings**:
  - Change the output file name, FPS, or resolution in `start_recording()`.

## Example Output

![Object Detection Example](https://github.com/user-attachments/assets/8b48c2af-65ff-4cab-b387-f37e78bc330a)

## Known Issues

- **Webcam Access**: Ensure your webcam is accessible. Use `cap = cv2.VideoCapture(<index>)` to switch devices.
- **Performance**: Detection speed depends on hardware capabilities (e.g., GPU acceleration).

## Future Improvements

- Add support for multiple camera inputs.
- Integrate advanced models for better accuracy.
- Provide pre-trained weights for specific datasets.

