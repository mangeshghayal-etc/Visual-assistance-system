import customtkinter as ctk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image
import threading
import cv2
import time
import pyttsx3
from ultralytics import YOLO

# ------------------ CONFIG ------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

model = YOLO("yolo11n.pt")  # YOLO11n model

# ------------------ TTS FUNCTION ------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ------------------ DETECTION FUNCTION ------------------
def run_detection(source):
    """Run YOLO detection (image or webcam) with normal window + TTS."""
    last_spoken = ""
    cooldown = 3
    last_time = time.time()

    # Webcam detection
    if source == 0:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)  # normal size

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

            if names:
                most_common = max(set(names), key=names.count)
                if most_common != last_spoken or (time.time() - last_time) > cooldown:
                    print(f"Detected: {most_common}")
                    t = threading.Thread(target=speak, args=(most_common,))
                    t.start()
                    t.join()
                    last_spoken = most_common
                    last_time = time.time()

            cv2.imshow("YOLO Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()

    # Image detection
    else:
        results = model(source)
        names = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                names.append(name)

        if names:
            most_common = max(set(names), key=names.count)
            print(f"Detected in image: {most_common}")
            t = threading.Thread(target=speak, args=(most_common,))
            t.start()
            t.join()

        results[0].show()

# ------------------ GUI SETUP ------------------
app = TkinterDnD.Tk()
app.title("YOLO Detection with Voice")
app.geometry("200x300")

bg_image = Image.open("background.png").resize((500, 500))
bg_ctk = ctk.CTkImage(light_image=bg_image, dark_image=bg_image, size=(500, 500))
bg_label = ctk.CTkLabel(app, image=bg_ctk, text="")
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

selected_file_path = None

# ------------------ FILE HANDLERS ------------------
def select_file():
    global selected_file_path
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg"),)
    )
    if file_path:
        selected_file_path = file_path
        threading.Thread(target=run_detection, args=(file_path,), daemon=True).start()

def drop(event):
    global selected_file_path
    file_path = event.data.strip("{}")
    try:
        img = Image.open(file_path)
        img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(380, 100))
        drag_label.configure(image=img_ctk, text="")
        drag_label.image = img_ctk
        selected_file_path = file_path
        threading.Thread(target=run_detection, args=(file_path,), daemon=True).start()
    except Exception:
        drag_label.configure(text="Unsupported file type", image="")

def submit_file():
    if selected_file_path:
        threading.Thread(target=run_detection, args=(selected_file_path,), daemon=True).start()
    else:
        print("No file selected")

def start_camera():
    threading.Thread(target=run_detection, args=(0,), daemon=True).start()

def destroy_frame():
    app.destroy()

# ------------------ GUI ELEMENTS ------------------
modal_frame = ctk.CTkFrame(app, fg_color="#2A2A2A", corner_radius=10, width=420, height=420)
modal_frame.place(x=40, y=40)

title_label = ctk.CTkLabel(modal_frame, text="Object Detection", font=("Arial", 18, "bold"))
title_label.place(x=140, y=15)

title_desc_label = ctk.CTkLabel(modal_frame, text="Upload and detect objects from Image", font=("Arial", 12))
title_desc_label.place(x=110, y=45)

upload_label = ctk.CTkLabel(modal_frame, text="Upload files", font=("Arial", 18, "bold"))
upload_label.place(x=20, y=70)

desc_label = ctk.CTkLabel(modal_frame,
    text="Attachments that have been uploaded as part of this project.",
    font=("Arial", 12))
desc_label.place(x=20, y=100)

drag_frame = ctk.CTkFrame(modal_frame, fg_color="#1E1E1E", corner_radius=5, width=390, height=110)
drag_frame.place(x=20, y=130)

drag_label = ctk.CTkLabel(
    drag_frame,
    text="Drag & drop your files here or choose files\n500 MB max file size.",
    font=("Arial", 12))
drag_label.place(relx=0.5, rely=0.5, anchor="center")

drag_frame.drop_target_register(DND_FILES)
drag_frame.dnd_bind("<<Drop>>", drop)

select_label = ctk.CTkLabel(modal_frame, text="Selected files", font=("Arial", 18, "bold"))
select_label.place(x=20, y=250)

file_entry = ctk.CTkEntry(modal_frame, placeholder_text="No file selected", width=245)
file_entry.place(x=20, y=280)
file_entry.configure(state="readonly")

select_button = ctk.CTkButton(modal_frame, text="Select Files", width=100, command=select_file)
select_button.place(x=270, y=280)

camera_btn = ctk.CTkButton(modal_frame, text="Start Webcam", width=150, command=start_camera)
camera_btn.place(x=130, y=330)

cancel_btn = ctk.CTkButton(modal_frame, text="Close", command=destroy_frame, width=100)
cancel_btn.place(x=20, y=380)

attach_btn = ctk.CTkButton(modal_frame, text="Submit file", command=submit_file,
                            width=150, fg_color="#3B82F6", hover_color="#2563EB")
attach_btn.place(x=250, y=380)

app.mainloop()
