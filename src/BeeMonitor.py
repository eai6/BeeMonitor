# A class that takes a path to a video file and return a csv file for the tracking data
import pandas as pd
from ultralytics import YOLO
import nest_processor
import motion_tracking
import process_tracking
import synthesize_video
import synthesize_csv
import Ultralytics_Tracker_wrapper as yolo_tracker

class VideoAnalyzer:
    def __init__(self, nest_model_path, tracking_model_path, res_height, res_width):
        self.nest_model = YOLO(nest_model_path) # load nest detection model
        self.tracking_model = YOLO(tracking_model_path) # load tracking model
        self.res_height = res_height
        self.res_width = res_width
    
    def getNestDetection(self, video_path):
        return nest_processor.getNestDetection(video_path, self.nest_model, self.res_height, self.res_width)
    
    def processNestDetection(self, nest_detection):
        return nest_processor.processNestDetection(nest_detection, self.res_height, self.res_width)
    
    def getMotionTracking(self, video_path, hotel_ROI, output_folder,visualize=False):
        return motion_tracking.detectMotionAndObjects(video_path, hotel_ROI, self.tracking_model, self.res_height, self.res_width, visualize, output_folder)
    
    def processMotionTracking(self, motion, nests):
        return process_tracking.processTracking(motion, nests)
    
    def synthesizeVideo(self, video_path, events, motion, output_folder, nests):
        return synthesize_video.synthesize(video_path, events, motion, nests, output_folder, self.res_height, self.res_width)
    
    def synthesizeCSV(self, events, filename):
        return synthesize_csv.processCSV(events, filename)

    def getYoloTracks(self, video_path, tracker_path):
        return yolo_tracker.getTracks(video_path, self.tracking_model, tracker_path)
    
    def processYoloTracks(self, tracks, nest):
        return process_tracking.processYoloTracks(tracks, nest)