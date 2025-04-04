# Author: Edward Amoah
# Date: 05-14-2021
# Purpose: helper file for processing nest detection and tracking data

import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import math

def getNestDetection(video_path, nest_model, res_height=360, res_width=640):
    '''
    Takes the video path, and nest model path and return a csv file with all the nest detections for the first 10 or 100 frames
    Input - video_path: path to the video file
            nest_model: YOLO model for nest detection

    Output - the csv file
    '''

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    frame_counter = 0
    nest_detections = [[]]
    nest_state = []
    frames = []
    confs = []
    # filenames = []
    
    while len(nest_detections[0]) < 50:
        # frame_counter += 1
        # Read a frame from the video

        # set cap to frame_counter
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        success, frame = cap.read()

        # resize frame 
        frame = cv2.resize(frame, (res_width, res_height))

        if success:
            # Run YOLOv8 inference on the frame
            results = nest_model.predict(frame, conf=0.9, verbose=False)

            # Visualize the results on the frame
            # annotated_frame = results[0].plot()

            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)

            # get detection 
            boxes = results[0].boxes.xyxy.tolist()
            boxes = [(x,y,x1,y1) for (x,y,x1,y1) in boxes]
            #nest_detections.append(boxes)
            nest_detections = [boxes]

            # nest labels
            labels = results[0].boxes.cls.tolist()
            nest_state = [labels]
            #nest_state.append(labels)

            # confidence
            conf = results[0].boxes.conf.tolist()
            confs = [conf]
            #confs.append(conf)

            #frames.append(frame_counter)
            frames = [frame_counter]


            frame_counter += 30 # skip 30 frames or 1 second

    nest_df = pd.DataFrame({'frame': frames, 'coordinates': nest_detections, 'state': nest_state, "confidence": confs})

    return nest_df


def processNestDetection(nest, res_height=360, res_width=640):
    def getNestCoordinates(nest, index=0):

        def get_midpoint(nest_coords):
            x1, y1, x2, y2 = nest_coords
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            return (int(midpoint_x), int(midpoint_y))

        states = nest.iloc[index]['state']
        coordinates = nest.iloc[index]['coordinates']

        nest_coords1 = []
        for i in range(len(states)):
            if states[i] == 2.0: # 2.0 is the label for nest_hole
                nest_coords1.append(coordinates[i])

        return [get_midpoint(nest_hole) for nest_hole in nest_coords1]
    

    # Function to cluster points into rows and columns
    def cluster_points(points, row_threshold=2, col_threshold=10):
        # Sort points by y-coordinate to cluster into rows
        points_sorted_y = sorted(points, key=lambda x: x[1])
        rows = []
        current_row = [points_sorted_y[0]]
        
        for point in points_sorted_y[1:]:
            if abs(point[1] - current_row[-1][1]) < row_threshold:
                current_row.append(point)
            else:
                rows.append(current_row)
                current_row = [point]
        rows.append(current_row)
        
        # Sort each row by x-coordinate to cluster into columns
        for i in range(len(rows)):
            rows[i] = sorted(rows[i], key=lambda x: x[0])

        # remove row with less than 5 
        rows = [row for row in rows if len(row) > 5]
        
        return rows
    
        # functions to fix missing holes in a row
    def getRowAverageNestDistance(row):
        nums = np.diff([x[0] for x in row]).tolist()

        nums = sorted(nums)

        s_nums = []
        current_num = nums[0]
        for num in nums:
            if abs(current_num - num) < 10:
                s_nums.append(num)
            current_num = num

        return int(np.mean(s_nums))

    def getRowAverageNestYDistance(row):
        nums = ([x[1] for x in row])#.tolist()

        nums = sorted(nums)

        s_nums = []
        current_num = nums[0]
        for num in nums:
            if abs(current_num - num) < 10:
                s_nums.append(num)
            current_num = num

        return int(np.mean(s_nums))

    def getAverageX(nums):

        nums = sorted(nums)

        s_nums = []
        current_num = nums[0]
        for num in nums:
            if abs(current_num - num) < 10:
                s_nums.append(num)
            current_num = num

        return int(np.mean(s_nums))


    def remove_overlapping_points(points, threshold=20):
        # Convert the list of points to a NumPy array for easier manipulation
        points = np.array(points)
        
        # List to store the indices of points to keep
        keep_indices = []
        
        # Iterate through each point
        for i in range(len(points)):
            keep = True
            # Compare with all previously kept points
            for j in keep_indices:
                # Calculate the Euclidean distance between the two points
                distance = np.linalg.norm(points[i] - points[j])
                if distance < threshold:
                    keep = False
                    break
            if keep:
                keep_indices.append(i)
        
        # Return the filtered points
        return points[keep_indices].tolist()


    def fixRowCords(row, first_hole_x, last_hole_x, pixel_threshold=10, x_average_width=0):
        new_row_cords = []
        #x_average_width = getRowAverageNestDistance(row)
        y_average = getRowAverageNestYDistance(row)

        for i in range(len(row)-1):
            if i == 0:
                # check that the first hole is close to the beginning of the range
                if abs(first_hole_x - row[i][0]) > pixel_threshold:
                    new_row_cords.append((first_hole_x, y_average))

            current_hole = row[i]
            next_hole = row[i+1]
            x_diff = next_hole[0] - current_hole[0]
            if abs(x_diff-x_average_width) < pixel_threshold:
                new_row_cords.append(current_hole)

                # if the last hole
                if i == len(row)-2:
                    # add the last hole 
                    new_row_cords.append(next_hole)

                    if len(new_row_cords) < pixel_threshold: # if the row has less than 10 holes check if the last hole is close to the end of the range
                        # check that the last hole is close to the end of the range
                        new_row_cords.append((last_hole_x, y_average))
                        # if abs(last_hole_x - next_hole[0]) > 10:
                        #     new_row_cords.append((last_hole_x, y_average))
            else:
                # add holes one at a time
                new_row_cords.append(current_hole)
                y_average = int((current_hole[1] + next_hole[1]) / 2)

                num = math.ceil(x_diff / x_average_width)
                for i in range(num-1):
                    new_row_cords.append((current_hole[0] + int(x_average_width * (i+1)), y_average))

        new_row_cords = remove_overlapping_points(new_row_cords, threshold=30)
        if len(new_row_cords) > 10:
            # remove holes that overlap with others by about 20 pixels
            #new_row_cords = [new_row_cords[0]] + [new_row_cords[i] for i in range(1, len(new_row_cords)-1) if abs(new_row_cords[i][0] - new_row_cords[i-1][0]) > 20] + [new_row_cords[-1]]
            new_row_cords = remove_overlapping_points(new_row_cords, threshold=30)

        if len(new_row_cords) < 10: # if the row has less than 10 holes check if the last hole is close to the end of the range
            new_row_cords.append((last_hole_x, y_average))

        return new_row_cords




    nest_coords = getNestCoordinates(nest)
    dl_rows = cluster_points(nest_coords, row_threshold=10)


    # get the rows where holes are 10
    rows_10 = [row for row in dl_rows if len(row) == 10]
    hole_first = getAverageX([x[0][0] for x in rows_10])
    hole_last = getAverageX([x[-1][0] for x in rows_10])

    # get average x_distance of all rows
    x_s = []
    for row in dl_rows:
        x_s.append(getRowAverageNestDistance(row))
    x_distance = getAverageX(x_s)

    # fix the rows
    fixed_dl_rows = []
    for row in dl_rows:
        fixed_dl_rows.append(fixRowCords(row, hole_first, hole_last,x_average_width=x_distance)) 


    hole_top = getAverageX([x[1] for x in fixed_dl_rows[0]])
    hole_bottom = getAverageX([x[1] for x in fixed_dl_rows[-1]])

    # hotel coordinates
    hx, hy, hx1, hy2 = (hole_first -100, hole_top-50, hole_last+100,hole_bottom+50)
    if hx < 0:
        hx = 0
    if hy < 0:
        hy = 0
    if hx1 > res_width:
        hx1 = res_width
    if hy2 > res_height:
        hy2 = res_height


    # generate nest_hole coordinates
    width = 38 #template.shape[1]
    height = 28 #template.shape[0]

    nest_ids = {
        "hotel": (hx, hy, hx1, hy2),
        "nests": {}
    }
    for j in range(len(fixed_dl_rows)):
        row = fixed_dl_rows[j]
        sorted_row = sorted(row, key=lambda x: x[0])
        for i, hole in enumerate(sorted_row):
            nest_ids["nests"][f'nest_{(i+1) + j * 10}'] = (hole[0] - width//2, hole[1] - height//2, hole[0] + width//2, hole[1] + height//2)

    return nest_ids
        
    


    
