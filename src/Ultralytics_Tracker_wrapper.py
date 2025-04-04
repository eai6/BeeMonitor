from collections import defaultdict
import cv2

from ultralytics import YOLO

def getTracks(video_path, model, tracker_path="src/bytetrack.yaml"):
    """
    Get the trajectories from a video using YOLOv8 tracking.

    Args:
        video_path (str): Path to the input video.
        model (YOLO): YOLOv8 model instance for tracking.
        tracker_path (str): Path to the tracker configuration file.[ByteTrack, Botsort]

    Returns:
        list: A list of trajectories, where each trajectory is a tuple containing:
            - track_id (int): The ID of the track.
            - track (list): A list of tuples representing the (x, y) coordinates of the track.
            - []: An empty list (placeholder for future use).
            - track_frame (list): A list of frame numbers corresponding to the track.
    
    """

    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])
    track_frame_history = defaultdict(lambda: [])

    # Loop through the video frames
    frame_num = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True, tracker = tracker_path)[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()


                # update the track history
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track_frame = track_frame_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    track_frame.append(frame_num)
          
            frame_num += 1

        else:
            break

    trajectories = [] # trac
    for track in track_history:
        trajectories.append((track, track_history[track], [], track_frame_history[track]))

    return trajectories