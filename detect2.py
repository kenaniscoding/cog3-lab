
# Copyright 2021 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0

import argparse
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # Red

##################################################
def visualize(image, detection_result):
    """Draws bounding boxes on the input image and returns it."""
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (_MARGIN + bbox.origin_x, _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    return image

##################################################
def run(model: str, camera_id: int, width: int, height: int,
        num_threads: int, enable_edgetpu: bool, target_object: str) -> None:
    """Continuously run inference on images from the camera and detect only the specified object."""

    counter, fps = 0, 0
    start_time = time.time()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    row_size = 20
    left_margin = 24
    text_color = (0, 0, 255)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model with allowlist
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5,
        category_allowlist=[target_object]  # Detect only specified object
    )
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam.')

        counter += 1
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(input_tensor)

        # Draw detection results
        image = visualize(image, detection_result)

        # Get frame midpoint (vertical line)
        frame_mid_x = width // 2
        cv2.line(image, (frame_mid_x, 0), (frame_mid_x, height), (0, 255, 0), 2)

        # Analyze object position
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x_min, x_max = bbox.origin_x, bbox.origin_x + bbox.width
            y_min, y_max = bbox.origin_y, bbox.origin_y + bbox.height

            # Compute overlap with left and right halves
            left_overlap = max(0, min(frame_mid_x, x_max) - x_min)
            right_overlap = max(0, x_max - max(frame_mid_x, x_min))

            left_percentage = (left_overlap / bbox.width) * 100 if bbox.width > 0 else 0
            right_percentage = (right_overlap / bbox.width) * 100 if bbox.width > 0 else 0

            # Determine side
            if left_percentage > 70:
                position = "LEFT"
            elif right_percentage > 70:
                position = "RIGHT"
            else:
                position = "CENTER"

            print(f"Detected {target_object}: Left={left_percentage:.1f}%, Right={right_percentage:.1f}% → {position}")

            # Display side info on frame
            cv2.putText(image, f"{position}",
                        (bbox.origin_x, bbox.origin_y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

        # FPS calculation
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        fps_text = f"FPS = {fps:.1f}"
        cv2.putText(image, fps_text, (left_margin, row_size),
                    cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        cv2.imshow('Object Detector (Filtered)', image)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

##################################################
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', help='Camera ID', type=int, default=0)
    parser.add_argument('--frameWidth', help='Frame width', type=int, default=640)
    parser.add_argument('--frameHeight', help='Frame height', type=int, default=480)
    parser.add_argument('--numThreads', help='CPU threads', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', help='Run on EdgeTPU', action='store_true', default=False)
    parser.add_argument('--target', help='Object category to detect (e.g., person, bottle, cup)', type=str, required=True)
    args = parser.parse_args()

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight,
        args.numThreads, args.enableEdgeTPU, args.target)

##################################################
if __name__ == '__main__':
    main()
