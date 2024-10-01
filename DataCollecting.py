import cv2
import numpy as np
import os
import mss
from pynput import keyboard
from datetime import datetime

# Directory to store data
data_dir = "trackmania_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create directories for labels
for label in ['0', '1', '2', '3']:  # 0: straight (w), 1: left (a), 2: right (d), 3: backward
    label_dir = os.path.join(data_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Variable to store the pressed key
key_pressed = None

# Function to handle key press events
def on_press(key):
    global key_pressed
    try:
        if key.char == 'w':
            key_pressed = '0'  # Straight
        elif key.char == 'a':
            key_pressed = '1'  # Left
        elif key.char == 'd':
            key_pressed = '2'  # Right
    except AttributeError:
        pass

def on_release(key):
    global key_pressed
    key_pressed = None

# Keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Function to capture and save screenshots
def collect_data():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Screen to capture
        
        while True:
            if key_pressed is not None:
                # Take a screenshot
                screenshot = np.array(sct.grab(monitor))
                
                # Convert the image from BGRA to BGR (for OpenCV)
                img_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                
                # Resize the image (optional, you can adjust this)
                img_bgr = cv2.resize(img_bgr, (100, 100))
                
                # Create a unique filename based on the timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                img_filename = f"{timestamp}.jpg"
                
                # Check if key_pressed is not None
                if key_pressed is not None:
                    # Path to the directory with the correct label
                    label_dir = os.path.join(data_dir, key_pressed)
                    img_path = os.path.join(label_dir, img_filename)
                    
                    # Save the original screenshot
                    cv2.imwrite(img_path, img_bgr)
                    print(f"Screenshot saved at {img_path}")

                    # If the key is 'left' (1) or 'right' (2), flip the image and save with the opposite label
                    if key_pressed == '1':  # Left
                        flipped_img = cv2.flip(img_bgr, 1)  # Flip horizontally
                        flipped_label_dir = os.path.join(data_dir, '2')  # Flip to right
                    elif key_pressed == '2':  # Right
                        flipped_img = cv2.flip(img_bgr, 1)  # Flip horizontally
                        flipped_label_dir = os.path.join(data_dir, '1')  # Flip to left
                    
                    if key_pressed in ['1', '2']:  # Save the flipped image
                        flipped_img_filename = f"{timestamp}_flipped.jpg"
                        flipped_img_path = os.path.join(flipped_label_dir, flipped_img_filename)
                        cv2.imwrite(flipped_img_path, flipped_img)
                        print(f"Flipped screenshot saved at {flipped_img_path}")

# Start data collection
collect_data()