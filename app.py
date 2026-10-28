import json
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

import torch
from gpiozero import OutputDevice

# ---------------- Configuration ----------------
RELAY_PIN = 17            # BCM pin number
RELAY_ON_SECONDS = 3
CAMERA_INDEX = 0
MODEL_PATH = os.path.join("models", "model.pt")
LABELS_PATH = os.path.join("models", "labels.json")

app = Flask(__name__)

# ---------------- Relay (active low) ----------------
# active_high=False -> GPIO LOW switches the relay ON, HIGH switches it OFF
relay = OutputDevice(RELAY_PIN, active_high=False, initial_value=False)
relay_lock = threading.Lock()


def pulse_relay():
    if not relay_lock.acquire(blocking=False):
        return  # already running
    try:
        relay.on()
        time.sleep(RELAY_ON_SECONDS)
    finally:
        relay.off()
        relay_lock.release()


# ---------------- Model ----------------
torch.set_num_threads(4)
model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
IMG_SIZE = 224

with open(LABELS_PATH) as f:
    LABELS = json.load(f)  # e.g. ["defected", "normal"]


def predict(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)   # 0-255, normalised inside the model
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        probs = model(x)[0]
    idx = int(torch.argmax(probs))
    return LABELS[idx], float(probs[idx])


# ---------------- Camera ----------------
class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.5)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


camera = Camera(CAMERA_INDEX)


def mjpeg():
    while True:
        frame = camera.get()
        if frame is not None:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(0.04)


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/analyze", methods=["POST"])
def analyze():
    frame = camera.get()
    if frame is None:
        return jsonify(error="Camera not available"), 503

    label, confidence = predict(frame)
    infected = label.lower() == "defected"
    if infected:
        threading.Thread(target=pulse_relay, daemon=True).start()

    return jsonify(
        label=label,
        confidence=round(confidence * 100, 1),
        infected=infected,
        relay_triggered=infected,
        relay_seconds=RELAY_ON_SECONDS if infected else 0,
    )


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        relay.off()
