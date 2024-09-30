import cv2
import numpy as np
import os
import mss
from pynput import keyboard
from datetime import datetime

# Directory om data op te slaan
data_dir = "trackmania_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Maak mappen aan voor labels
for label in ['0', '1', '2']:  # 0: rechtdoor (z), 1: links (q), 2: rechts (d)
    label_dir = os.path.join(data_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Variabelen om toetsaanslagen op te slaan
key_pressed = None

# Functie om toetsaanslagen vast te leggen
def on_press(key):
    global key_pressed
    try:
        if key.char == 'z':
            key_pressed = '0'  # Rechtdoor
        elif key.char == 'q':
            key_pressed = '1'  # Links
        elif key.char == 'd':
            key_pressed = '2'  # Rechts
    except AttributeError:
        pass

def on_release(key):
    global key_pressed
    key_pressed = None

# Listener voor toetsenbord
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Functie om screenshots te maken en op te slaan
def collect_data():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Scherm dat je wilt capturen (je kunt dit aanpassen)
        
        while True:
            if key_pressed is not None:
                # Screenshot nemen
                screenshot = np.array(sct.grab(monitor))
                
                # Converteer de afbeelding van BGRA naar BGR (voor OpenCV)
                img_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                
                # Verklein de afbeelding
                

                
                # Maak een uniek bestandsnaam op basis van de tijd
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                img_filename = f"{timestamp}.jpg"
                
                # Controleer of key_pressed niet None is
                if key_pressed is not None:
                    # Pad naar de map met het juiste label
                    label_dir = os.path.join(data_dir, key_pressed)
                    img_path = os.path.join(label_dir, img_filename)
                    
                    # Sla het screenshot op
                    cv2.imwrite(img_path, img_bgr)
                    
                    print(f"Screenshot saved at {img_path}")

# Start data verzamelen
collect_data()
