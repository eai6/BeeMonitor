from RPi import GPIO
import time
import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import os

# Set Height & Width & fps of the frame
frameWidth = 1920
frameHeight = 1080
fps = 25

# Set recording duration (in seconds)
record_duration = 3600 # 1 hour = 3600 sec

# Initialize camera
picam2 = Picamera2()
picam2.video_configuration.controls.FrameRate = fps
picam2.video_configuration.size = (frameWidth, frameHeight)

# Timestamp configuration
colour = (255, 255, 255)
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2

def apply_timestamp(request):
    timestamp = time.strftime("%Y-%m-%d %X")
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)

# picam2.pre_callback = apply_timestamp

def continuous_recording():
    while True:
        # Start recording
        timestamp = time.strftime("%Y-%m-%dT%H%M")
        video_name = f"{timestamp}.mp4"
        directory = "/media/insect1/Elements/1-hour-continue/"
        os.makedirs(directory, exist_ok=True)  # This line will create the directory if it doesn't exist
        video_path = f"{directory}{video_name}"
        encoder = H264Encoder(bitrate=50000000) # maximum so far should be 17000000, can use qp=0 to replace
        out = FfmpegOutput(video_path)
        picam2.start_recording(encoder, out)
        print("Recording started.")

        # Record for 1 hour
        time.sleep(record_duration)

        # Stop recording
        picam2.stop_recording()
        print("Recording stopped.")

if __name__ == '__main__':
    try:
        continuous_recording()
    except KeyboardInterrupt:
        print("Recording interrupted.")
