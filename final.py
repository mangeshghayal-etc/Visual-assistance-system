from ultralytics import YOLO
import pyttsx3
import threading
import cv2
import time

# --- Initialize model ---
model = YOLO("yolo11n.pt")   # Use your YOLO model

# --- Text-to-speech function ---
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Open camera ---
cap = cv2.VideoCapture(0)

last_spoken = ""      # To avoid repeating same object too fast
cooldown = 3          # seconds between repeats
last_time = time.time()

# --- Create a window and set fullscreen mode ---
cv2.namedWindow("YOLO Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    names = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            names.append(name)
            print(1,names)
    if names:
        most_common = max(set(names), key=names.count)

        if most_common != last_spoken or (time.time() - last_time) > cooldown:
            print(f"Detected: {most_common}")
            t = threading.Thread(target=speak, args=(most_common,))
            t.start()
            t.join()
            last_spoken = most_common
            last_time = time.time()

    # --- Display detection results in fullscreen ---
    cv2.imshow("YOLO Detection", results[0].plot())

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
