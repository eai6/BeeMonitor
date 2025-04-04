import cv2

def synthesize(video_path, events, motion, nest, output_folder, res_height=720, res_width=1280):

    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    filename = video_path.split('/')[-1].split('.')[0]
    output_video = cv2.VideoWriter(f"{output_folder}/synthesized_video_{filename}_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (res_width, res_height))


    frame_numbers = events.frame_number.tolist()
    nest_holes = events.nest.tolist()
    actions = events.action.tolist()

    class Track:
        def __init__(self, track_id, trajectory, frame_numbers):
            self.track_id = track_id
            self.trajectory = trajectory
            self.frame_numbers = frame_numbers
        
        def is_in_frame(self, frame_num):
            return frame_num in self.frame_numbers
        
        def get_bbox(self, frame_num):
            idx = self.frame_numbers.index(frame_num)
            return self.trajectory[idx]
        
        def getID(self):
            return self.track_id


    for i in range(len(motion.frame_number.tolist())):
        try:
            period = motion.frame_number.tolist()[i]
            tracks = motion.tracks.tolist()[i]
            track_objects = []
            for track in tracks:
                track_objects.append(Track(track[0], track[2], track[3]))
            for frame_num in range(period[0], period[1]+1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                # resize frame
                frame = cv2.resize(frame, (res_width, res_height))

                # print nests on frame
                for nest_hole in nest['nests']:
                    id = nest_hole.split('_')[-1]
                    x1, y1, x2, y2 = nest['nests'][nest_hole]

                    x1 -= 5
                    y1 -= 7
                    x2 += 5
                    y2 += 7
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                    cv2.putText(frame, f"{id}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

                for track in track_objects:
                    if track.is_in_frame(frame_num):
                        bbox = track.get_bbox(frame_num)
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cv2.putText(frame, f"{track.getID()}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                        
                        

                # print action on frame
                if frame_num in frame_numbers:
                    idx = frame_numbers.index(frame_num)
                    action = actions[idx]
                    nest_id = nest_holes[idx]

                    if action == "Exit":
                        cv2.putText(frame, f"{action} at nest {nest_id}", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
                    else:
                        cv2.putText(frame, f"{action} at nest {nest_id}", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    # loop on the same frame for a second
                    for i in range(30):
                        output_video.write(frame)

                output_video.write(frame)
        except:
            pass

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    

    return f"{output_folder}/synthesized_{filename}_video.mp4"