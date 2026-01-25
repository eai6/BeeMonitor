from picamera2 import Picamera2, Preview
import time
import libcamera
cam = Picamera2()

preview_config = cam.create_preview_configuration(transform=libcamera.Transform(vflip=1))

cam.configure(preview_config)

cam.start_preview(Preview.QT)

cam.start()

time.sleep(2)

cam.start_and_record_video("/home/apis/Desktop/cameraOutput/cameraTesting/test.mp4", duration = 20)

cam.stop()

cam.stop_preview

