

from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO
from IPython.display import Image

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="IfYBRUsYCLgPIZOzBGzL")
project = rf.workspace("east-west-uniersity").project("violance-nonviolance")
version = project.version(7)
dataset = version.download("yolov11")

#to print dataset images
import os
import matplotlib.pyplot as plt
import cv2
dataset_path = 'upload dataset path'
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
for i in range(5):
    img_path = os.path.join(dataset_path, image_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'Image {i+1}: {image_files[i]}')
    plt.axis('off')
    plt.show()

!yolo task=detect mode=predict model= "" conf=0.9 source={dataset.location}/test/images save=True

Image("", width=600)

Image("", width=600)

Image("", width=600)

Image("", width=600)

Image("/content/drive/MyDrive/Final_trained_epochs/runs/detect/train/results.png", width=600)

Image("", width=600)

Image("", width=600)

Image("", width=600)

csv_path=''
import pandas as pd
df = pd.read_csv(csv_path)
print(df)
df.tail()

!pip install ultralytics
!pip install opencv-python-headless
!pip install yagmail

!pip install gradio

"""e-mail alone

"""

import gradio as gr
import cv2
import os
import yagmail
from datetime import datetime
from ultralytics import YOLO

# === Configuration ===
model_path = ""#Replace with your best.pt
threshold = 0.85
snapshot_path = "first_violence_frame.jpg"
camera_label = ""
your_email =""  # Replace with your email
app_password = ""     # Replace with your app password
receiver_email = ""  # Replace with receiver email

# === Load Model ===
model = YOLO(model_path)

def process_video_gradio(video_file):
    violence_count = 0
    non_violence_count = 0
    first_violence_frame_saved = False
    timestamp_of_first_violence = ""

    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= threshold:
                if cls_id == 1:
                    violence_count += 1

                    if not first_violence_frame_saved:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        label = f"Violence {conf:.2f}"
                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        timestamp_of_first_violence = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.imwrite(snapshot_path, frame)
                        first_violence_frame_saved = True

                elif cls_id == 0:
                    non_violence_count += 1

    cap.release()

    # === Send Email ===
    if violence_count > non_violence_count and os.path.exists(snapshot_path):
        yag = yagmail.SMTP(user=your_email, password=app_password)
        email_body = (
            f"\U0001F6A8 ALERT : Violence Detected\n"
            f"Camera Location: {camera_label}\n"
            f"Timestamp : {timestamp_of_first_violence}\n"
            f"The attached image shows the detected activity."
        )

        yag.send(
            to=receiver_email,
            subject=f"\U0001F6A8 Suspicious Activity Alert - {camera_label}",
            contents=[email_body],
            attachments=snapshot_path
        )
        result_text = "Violence Detected. Alert email sent."
        detected_img = snapshot_path
        timestamp = timestamp_of_first_violence
        location = camera_label
    else:
        result_text = "No violence detected."
        detected_img = None
        timestamp = ""
        location = ""

    return result_text, detected_img, timestamp, location

# === Gradio Interface ===
interface = gr.Interface(
    fn=process_video_gradio,
    inputs=gr.Video(label="Upload CCTV Video"),
    outputs=[
        gr.Text(label="Detection Result"),
        gr.Image(label="Detected Snapshot"),
        gr.Text(label="Timestamp"),
        gr.Text(label="Camera Location")
    ],
    title="Violence Detection System with YOLOv11",
    description="Upload a CCTV video. If suspicious activity is detected, the system will display the snapshot, timestamp, location and send an alert email."
)

interface.launch(debug=True)

"""email and call alert

"""

!pip install ultralytics yagmail twilio gradio opencv-python-headless

import gradio as gr
import cv2
import os
import yagmail
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client

# === Configuration ===
model_path = "" # replace with your path
threshold = 0.85
snapshot_path = "first_violence_frame.jpg"
camera_label = "Metro Station-Platform 3 (CAM_04)"

# Email Config (replace with your credentials)
your_email = ""
app_password = ""
receiver_email = ""

# Twilio Config (replace with your Twilio details)
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""   # Your Twilio number
SECURITY_PHONE_NUMBER = ""  # Security personnel number (must be verified in trial)

# Load YOLO Model
model = YOLO(model_path)

# === Helper Functions ===
def send_email_alert(snapshot_path, timestamp):
    """Send email when violence is detected"""
    yag = yagmail.SMTP(user=your_email, password=app_password)
    email_body = (
        f"\U0001F6A8 ALERT : Violence Detected\n"
        f"Camera Location: {camera_label}\n"
        f"Timestamp : {timestamp}\n"
        f"The attached image shows the detected activity."
    )

    yag.send(
        to=receiver_email,
        subject=f"\U0001F6A8 Suspicious Activity Alert - {camera_label}",
        contents=[email_body],
        attachments=snapshot_path
    )

def make_call_alert(timestamp):
    """Make a phone call when violence is detected (repeats until hangup)"""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call_message = (
        f"Alert! Violence detected at {camera_label}. "
        f"Time: {timestamp}. Immediate action required."
    )

    call = client.calls.create(
        to=SECURITY_PHONE_NUMBER,
        from_=TWILIO_PHONE_NUMBER,
        twiml=f'<Response><Say loop="0">{call_message}</Say></Response>'
    )
    return call.sid

def process_video_gradio(video_file):
    violence_count = 0
    non_violence_count = 0
    first_violence_frame_saved = False
    timestamp_of_first_violence = ""

    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    consecutive_violence_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)[0]

        violence_in_frame = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= threshold:
                if cls_id == 1:
                    violence_in_frame = True
                    violence_count += 1

                    if not first_violence_frame_saved:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        label = f"Violence {conf:.2f}"
                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        timestamp_of_first_violence = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.imwrite(snapshot_path, frame)
                        first_violence_frame_saved = True

                elif cls_id == 0:
                    non_violence_count += 1

        if violence_in_frame:
            consecutive_violence_frames += 1
        else:
            consecutive_violence_frames = 0

        if consecutive_violence_frames >= 5:
            break

    cap.release()

    # === Send Email & Call ===
    if consecutive_violence_frames >= 5 and os.path.exists(snapshot_path):
        send_email_alert(snapshot_path, timestamp_of_first_violence)
        make_call_alert(timestamp_of_first_violence)
        result_text = "🚨 Violence Detected. Email + Phone Call Alert Sent."
        detected_img = snapshot_path
        timestamp = timestamp_of_first_violence
        location = camera_label
    else:
        result_text = "No violence detected."
        detected_img = None
        timestamp = ""
        location = ""

    return result_text, detected_img, timestamp, location


interface = gr.Interface(
    fn=process_video_gradio,
    inputs=gr.Video(label="Upload CCTV Video"),
    outputs=[
        gr.Text(label="Detection Result"),
        gr.Image(label="Detected Snapshot"),
        gr.Text(label="Timestamp"),
        gr.Text(label="Camera Location")
    ],
    title="Violence Detection System with YOLOv11",
    description="Upload a CCTV video. If suspicious activity is detected , the system will send an Email and keep calling security until they hang up."
)

interface.launch(debug=True)



# === Install dependencies (Run only once) ===
!pip install ultralytics yagmail twilio gradio opencv-python-headless

# === Import Libraries ===
import gradio as gr
import cv2
import os
import yagmail
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client

# === Configuration ===
model_path = ""
threshold = 0.70  # relaxed threshold for better recall
snapshot_path = "first_violence_frame.jpg"
camera_label = ""

# === Email Config ===
your_email = ""
app_password = ""
receiver_email = ""

# === Twilio Config ===
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""   # Twilio verified number
SECURITY_PHONE_NUMBER = ""  # Security personnel (verified number)

# === Load YOLOv11 Model ===
model = YOLO(model_path)


# === Helper Function: Send Email ===
def send_email_alert(snapshot_path, timestamp):
    """Send an email when violence is detected."""
    yag = yagmail.SMTP(user=your_email, password=app_password)
    email_body = (
        f"🚨 ALERT : Violence Detected\n\n"
        f"📍 Camera Location: {camera_label}\n"
        f"🕒 Timestamp : {timestamp}\n\n"
        f"The attached image shows the detected activity."
    )

    yag.send(
        to=receiver_email,
        subject=f"🚨 Suspicious Activity Alert - {camera_label}",
        contents=[email_body],
        attachments=snapshot_path
    )


# === Helper Function: Make Call ===
def make_call_alert(timestamp):
    """Make a phone call when violence is detected."""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call_message = (
        f"Alert! Violence detected at {camera_label}. "
        f"Time: {timestamp}. Immediate action required."
    )

    call = client.calls.create(
        to=SECURITY_PHONE_NUMBER,
        from_=TWILIO_PHONE_NUMBER,
        twiml=f'<Response><Say loop="2">{call_message}</Say></Response>'
    )
    return call.sid


# === Main Processing Function ===
def process_video_gradio(video_file):
    violence_count = 0
    non_violence_count = 0
    first_violence_frame_saved = False
    timestamp_of_first_violence = ""
    consecutive_violence_frames = 0

    print("🔹 Starting video processing...")
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⛔ End of video or unable to read frame.")
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)[0]

        violence_in_frame = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            print(f"Frame {frame_count}: Detected class={cls_id}, conf={conf:.2f}")

            if conf >= threshold:
                if cls_id == 1:  # Violence
                    violence_in_frame = True
                    violence_count += 1

                    if not first_violence_frame_saved:
                        # Save the first violence frame WITHOUT drawing bounding boxes
                        timestamp_of_first_violence = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.imwrite(snapshot_path, frame)
                        first_violence_frame_saved = True
                        print("📸 First violence frame saved (no bounding box).")

                elif cls_id == 0:
                    non_violence_count += 1

        # Check for consecutive frames
        if violence_in_frame:
            consecutive_violence_frames += 1
            print(f"⚠️ Violence frame count: {consecutive_violence_frames}")
        else:
            consecutive_violence_frames = 0

        if consecutive_violence_frames >= 5:
            print("🚨 Violence confirmed after 5 consecutive frames!")
            break

    cap.release()

    # === Sequential Alert Section ===
    if consecutive_violence_frames >= 5 and os.path.exists(snapshot_path):
        try:
            print("📧 Sending email alert...")
            send_email_alert(snapshot_path, timestamp_of_first_violence)
            print("✅ Email sent successfully.")

            print("📞 Initiating call alert...")
            make_call_alert(timestamp_of_first_violence)
            print("✅ Call initiated successfully.")

            result_text = "🚨 Violence Detected. Email sent, then Phone Call Alert initiated."
            detected_img = snapshot_path
            timestamp = timestamp_of_first_violence
            location = camera_label

        except Exception as e:
            print(f"❌ Error during alerts: {e}")
            result_text = f"Violence Detected, but alert failed: {e}"
            detected_img = snapshot_path
            timestamp = timestamp_of_first_violence
            location = camera_label

    else:
        print("✅ No violence detected in this video.")
        result_text = "No violence detected."
        detected_img = None
        timestamp = ""
        location = ""

    print("🔹 Processing complete.")
    return result_text, detected_img, timestamp, location


# === Gradio Interface ===
interface = gr.Interface(
    fn=process_video_gradio,
    inputs=gr.Video(label="Upload CCTV Video"),
    outputs=[
        gr.Text(label="Detection Result"),
        gr.Image(label="Detected Snapshot"),
        gr.Text(label="Timestamp"),
        gr.Text(label="Camera Location")
    ],
    title="Violence Detection System with YOLOv11",
    description="Upload a CCTV video. If violence is detected, the system will first send an Email alert, then make a phone call to security personnel."
)

# === Launch Gradio App ===
interface.launch(debug=True)
