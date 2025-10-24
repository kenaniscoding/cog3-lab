
import cv2
import numpy as np
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# --- Setup push button ---
BUTTON_PIN = 17  # GPIO pin for button
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- Setup Mediapipe segmentation ---
mp_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_segmentation.SelfieSegmentation(model_selection=1)

# --- Load new background image ---
background_img = cv2.imread("new_background.jpg")
background_img = cv2.resize(background_img, (640, 480))

# --- Video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

use_new_bg = False
last_button_state = GPIO.input(BUTTON_PIN)

print("Press the button to toggle background...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not available.")
            break

        # Resize for consistency
        frame = cv2.resize(frame, (640, 480))

        # Segment the person from background
        results = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask

        # Binary mask for person (1=person, 0=background)
        condition = mask > 0.5

        # --- Check button press to toggle ---
        button_state = GPIO.input(BUTTON_PIN)
        if button_state == GPIO.LOW and last_button_state == GPIO.HIGH:
            use_new_bg = not use_new_bg
            print("Background toggled:", "New" if use_new_bg else "Original")
            time.sleep(0.3)  # debounce delay

        last_button_state = button_state

        # --- Choose background ---
        if use_new_bg:
            output_image = np.where(condition[..., None], frame, background_img)
        else:
            output_image = frame

        cv2.imshow("Selfie Segmentation", output_image)

        # Exit with 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
