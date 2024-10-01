import cv2
import numpy as np
import torch
from CNNModel import CNNModel  # Verwijs naar je modelbestand of plaats dit in hetzelfde script
import mss
import time
import keyboard



# Model laden
model = CNNModel()
model.load_state_dict(torch.load('trackmania_model.pth'))
model.eval()  # Zet model in evaluatiemodus

IMG_HEIGHT, IMG_WIDTH = 100, 100


# Functie om screenshots te nemen en te gebruiken voor het model
def drive():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Scherm dat je wilt capturen (je kunt dit aanpassen)

        while True:
            # Screenshot nemen
            screenshot = np.array(sct.grab(monitor))

            # Converteer de afbeelding van BGRA naar BGR (voor OpenCV)
            img_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            # Verklein de afbeelding naar 100x100 pixels (zelfde als tijdens training)
            img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))

            # Converteer de afbeelding naar een tensor voor het model
            img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # PyTorch input

            # Maak een voorspelling
            with torch.no_grad():
                output = model(img_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            # Print de voorspelling naar de console
            print(f"Predicted class: {predicted_class}")

            # Voeg de voorspelling toe aan het frame
            cv2.putText(img_bgr, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Bestuur de auto op basis van voorspelling
            if predicted_class == 0:
                keyboard.press("z")
                keyboard.release("q")
                keyboard.release("d")
                keyboard.release("s")
            elif predicted_class == 1:
                keyboard.press("q")
                keyboard.release("z")
                keyboard.release("d")
                keyboard.release("s")
            elif predicted_class == 2:
                keyboard.press("d")
                keyboard.release("q")
                keyboard.release("z")
                keyboard.release("s")
            elif predicted_class == 3:
                keyboard.press("s")
                keyboard.release("q")
                keyboard.release("z")
                keyboard.release("d")
            time.sleep(0.2)  # Vertraging om de toetsaanslag te registreren

            

            

    cv2.destroyAllWindows()

# Start de drive functie
drive()
