import cv2
import numpy as np
import torch
from pynput.keyboard import Controller
from CNNModel import CNNModel  # Verwijs naar je modelbestand of plaats dit in hetzelfde script
import mss
import time

# Model laden
model = CNNModel()
model.load_state_dict(torch.load('trackmania_model.pth'))
model.eval()  # Zet model in evaluatiemodus

keyboard = Controller()

IMG_HEIGHT, IMG_WIDTH = 100, 100

# Functie om een toets in te drukken
def press_key(key):
    keyboard.press(key)
    keyboard.release(key)

# Functie om de auto te besturen
def drive():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Scherm dat je wilt capturen (je kunt dit aanpassen)

        while True:
            # Neem een screenshot
            screenshot = np.array(sct.grab(monitor))

            # Converteer BGRA naar BGR
            frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            frame_resized = torch.tensor(frame_bgr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # PyTorch input

            # Voorspelling maken
            with torch.no_grad():
                output = model(frame_resized)
                predicted_class = torch.argmax(output, dim=1).item()

            # Bestuur de auto
            if predicted_class == 0:
                press_key('z')  # Rechtdoor
            elif predicted_class == 1:
                press_key('q')  # Links
            elif predicted_class == 2:
                press_key('d')  # Rechts

            # Optioneel: Toon het huidige frame
            cv2.imshow('Trackmania', frame_bgr)

            # Wacht 0.3 seconde voordat je de volgende screenshot neemt
            time.sleep(0.6)

            # Stoppen als 'q' wordt ingedrukt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

drive()
