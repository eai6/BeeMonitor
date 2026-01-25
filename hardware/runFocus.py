from picamera2 import Picamera2, Preview
import time
import libcamera


cam = Picamera2()

preview_config = cam.create_preview_configuration(transform=libcamera.Transform(vflip=1))

cam.configure(preview_config)

cam.start_preview(True)

cam.start()

time.sleep(1)

cam.start_and_record_video("/home/apis/Desktop/cameraOutput/cameraTesting/focus.mp4", duration = 40)

cam.stop()

cam.stop_preview


    

