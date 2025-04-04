import cv2
from ultralytics import YOLO
import pandas as pd
import os
from  BeeTrack import Tracker
import traceback


def runInfereceOnFrame(current_frame, current_frame_roi, model, site_roi, visualize=False, output_folder=None, frame_num=0):

    results = model(current_frame, verbose=False, cls=3, iou=0.5)

    #annotated_frame = results[0].plot()

    #cv2.imwrite(f"{output_folder}/{frame_num}_annotated_frame.png", annotated_frame)

    # get detection
    boxes = results[0].boxes.xywh.tolist()

    # get class ids
    labels = results[0].boxes.cls.tolist()

    normalized_boxes = []

    for x, y, w, h in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)

        aspect_ratio = w/h  

        if aspect_ratio > 0.5 and aspect_ratio < 2:

            #normalized_boxes.append((x, y, x + w, y + h))
            normalized_boxes.append((x - int(w/2) , y -int(h/2), x + w - int(w/2), y + h -int(h/2)))

            if visualize:
                #cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(current_frame, (x - (int(w/2)), y -(int(h/2))), (x + w - (int(w/2)), y + h -(int(h/2))), (0, 0, 255), 2)

    #cv2.imwrite(f"{output_folder}/{frame_num}_annotate_current_frame.png", current_frame)
                
    return normalized_boxes, labels, current_frame


def detectMotionOnFrame(current_frame, prev_frame_gray, current_frame_gray, site_roi, threshold=5, visualize=False, output_folder=None, frame_num=0):

    # save previous frame
    #cv2.imwrite(f"{output_folder}/{frame_num}_prev_frame.png", prev_frame_gray)
    
    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)

    # save frame
    #cv2.imwrite(f"{output_folder}/{frame_num}_frame_diff.png", frame_diff)
    
    # Apply thresholding to the difference frame
    _, thresholded_frame = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # save frame
    #cv2.imwrite(f"{output_folder}/{frame_num}_thresholded_frame.png", thresholded_frame)
    
    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around moving objects
    frame_contours = []


    x1, y1 , x2, y2 = site_roi
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # You can adjust this threshold based on your specific case
            (x, y, w, h) = cv2.boundingRect(contour)

            aspect_ratio = w/h  

            if aspect_ratio > 0.5 and aspect_ratio < 2:

                # plot on diff and thresholded frame
                cv2.rectangle(thresholded_frame, (x, y), (x + w, y + h), (255,0,0), 2)
                cv2.rectangle(frame_diff, (x, y), (x + w, y + h), (255,0,0), 2)
                
                #frame_contours.append((x, y, w, h))
                frame_contours.append((x+x1 , y+y1 , x + w+x1, y + h+y1))

                if visualize:
                    cv2.rectangle(current_frame, (x+x1 , y+y1 ), (x + w+x1, y + h+y1), (255,0,0), 2)


    return frame_contours, current_frame


def motionDetection(cap, frame_num, res_height, res_width, site_roi, visualize=False, model = None, video_output = None, output_folder=None):
    # Read the next frame

    try:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            print('Motion No frame')
            return None

        # Resize the frame
        frame = cv2.resize(frame, (res_width, res_height))

        # focus on roi
        x1, y1, x2, y2 = site_roi
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        frame_roi = frame[y1:y2, x1:x2]

        prev_frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        test_output  = output_folder
        
        while True:

            # Read the next frame
            frame_num += 1

            # Set the frame position
            #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # Read the frame
            ret, frame = cap.read()

            if not ret:
                break

            # Resize the frame
            frame = cv2.resize(frame, (res_width, res_height))

            # focus on roi
            frame_roi = frame[y1:y2, x1:x2]

            # Get the ROI in grayscale
            frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

            # detect motion
            motionframe = frame.copy()
            frame_contours, motionframe = detectMotionOnFrame(motionframe, prev_frame_gray, frame_roi_gray, site_roi, visualize=visualize, output_folder=output_folder, frame_num=frame_num)

            #cv2.imwrite(f"{test_output}/{frame_num}_pre_motion_frame.png", motionframe)

            # Update the previous frame
            prev_frame_gray = frame_roi_gray

            if len(frame_contours) > 0: # motion detected
                inferenceframe = frame.copy()
                _, labels, inferenceframe = runInfereceOnFrame(inferenceframe, frame_roi, model, site_roi, visualize, output_folder, frame_num)

                # save frames 
                #cv2.imwrite(f"{test_output}/{frame_num}_pre_inference_frame.png", inferenceframe)

                if len(labels) > 0: # confirm object detection
                    if visualize:
                        cv2.rectangle(inferenceframe, (x1, y1), (x2, y2), (0,0,0), 2)
                        video_output.write(inferenceframe)
                    break
                else:
                    if visualize:
                        cv2.rectangle(inferenceframe, (x1, y1), (x2, y2), (0,0,0), 2)
                        video_output.write(inferenceframe)
                    continue
            else:
                if visualize:
                    cv2.rectangle(motionframe, (x1, y1), (x2, y2), (0,0,0), 2)
                    video_output.write(motionframe)

            

        return frame_num
    except Exception as e:
        print(e)
        print(f"Error at frame {frame_num}")
        return frame_num+1


def visualizeTracking(frame, tracks, site_roi):
    
        x1, y1 , x2, y2 = site_roi
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
        for track in tracks:
            #for bbox in track:
            x0, y0, w1, y1 = track[0]
            x0, y0, w1, y1 = int(x0), int(y0), int(w1), int(y1)
            cv2.rectangle(frame, (x0, y0), (w1, y1), (0, 255, 255), 2)
            cv2.putText(frame, f"Track {track[1]}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
    
        return frame

def detectionAndTracking(model, cap, frame_num, res_height, res_width, site_roi, visualize=False, video_output = None, output_folder=None, track_id=0):
    
    try:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1) # use frame before motion was detected

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            print('Tracking No frame')
            return [], frame_num+1, []

        # Resize the frame
        frame = cv2.resize(frame, (res_width, res_height))

        # focus on roi
        x1, y1, x2, y2 = site_roi
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        frame_roi = frame[y1:y2, x1:x2]

        prev_frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        no_motion_counter = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_detections_dict = {}

        tracker = Tracker(track_start_id=track_id)

        test_output  = output_folder

        while frame_num < total_frames-1:

            # Read the next frame
            #frame_num += 1

            # Set the frame position
            #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # Read the frame
            ret, frame = cap.read()

            if not ret:
                break

            # Resize the frame
            frame = cv2.resize(frame, (res_width, res_height))

            # focus on roi
            frame_roi = frame[y1:y2, x1:x2]

            # Get the ROI in grayscale
            #frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

            frame_num_detections = 0
            
            # just DL
            inference_frame = frame.copy()
            boxes, labels, frame = runInfereceOnFrame(inference_frame, frame_roi, model, site_roi, visualize, test_output, frame_num)
            tracked_objects = tracker.update(boxes, frame_num)
            frame_num_detections = len(boxes)
            # just DL end 

            
            # store labels and frame number
            frame_detections_dict[frame_num] = boxes
                    

            if frame_num_detections == 0: # motion detected
                no_motion_counter += 1

            if frame_num_detections > 0 and no_motion_counter > 0: # motion detected
                no_motion_counter = 0
                
            if no_motion_counter > 30: # no motion detected for 10 frames
                break

            if visualize:
                frame = visualizeTracking(frame, tracked_objects, site_roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                
                video_output.write(frame)
            
            # Read the next frame
            frame_num += 1

        
    
        return tracker.getTracks(), frame_num, frame_detections_dict
    except Exception as e:
        # print tracke of error
        print(e)
        print(f'Error at frame number {frame_num}')
        traceback.print_exc() 
        return [], frame_num+1, []



def detectMotionAndObjects(video, site_roi, model, res_height, res_width, visualize=False, output_folder="/Users/edwardamoah/Documents/GitHub/BeeVision/monitoring_data"):

    # check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #print(f'Total frames: {total_frames}')

    frame_num = 0

    frames = []
    tracks = []
    tracking_detections = []

    track_id = 0


    output_video = None
    if visualize:
        filename = video.split('/')[-1].split('.')[0]
        output_file = f"{output_folder}/processed_video_{filename}.mp4"

        # Define the codec and create VideoWriter object
        output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (res_width, res_height))

    while frame_num < total_frames:

        track = []
        tracking_detection = []
        
            # motion detection
        if frame_num < total_frames:
            frame_num = motionDetection(cap, frame_num, res_height, res_width, site_roi, visualize, model, output_video, output_folder)

        if frame_num == None:
            break

        #activity_start = frame_num
        motion_frame = frame_num

        #print(f'Motion detected at frame {frame_num}')

        #'''
        # decrement frame number by 1 to get the frame before motion was detected

        # avoid looping at the end of the video
        frame_num -= 5
        if frame_num < 0:
            frame_num = 0

        # avoid looping at the end of the video
        if frame_num + 5 > total_frames:
            frame_num = frame_num + 5
       #'''

        activity_start = frame_num

        #print(f'Tracking started at frame {frame_num}')
            
        # detection and tracking
        if frame_num < total_frames:
            track, frame_num, tracking_detection =  detectionAndTracking(model, cap, frame_num, res_height, res_width, site_roi, visualize, output_video, output_folder, track_id)

        #print(f'Tracking ended at frame {frame_num}')
        
        activity_end = frame_num

        if activity_end < motion_frame: # avoid overlapping frames
            frame_num = activity_end = motion_frame+6
            track = []

        #print(f'Tracking ended {frame_num}')

        
        frames.append((activity_start, activity_end))
        tracks.append(track)
        tracking_detections.append(tracking_detection)

        track_id += len(track)

        # print(detections)
        # print(activity_start, activity_end)
        # print(track)
        # if len(track) > 0:
        #     frames.append((activity_start, activity_end))
        #     tracks.append(track)

    if visualize:
        output_video.release()

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame({'frame_number': frames,  'tracks': tracks, 'detections': tracking_detections})

    return df

