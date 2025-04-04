import numpy as np
import pandas as pd

# Function to check if a point is inside a bounding box
def is_inside_bbox(bee_position, bbox, padding=20):
    """
    Checks if a bee's position (x, y) is inside a given bounding box with padding.
    """
    x, y = bee_position
    x_min, y_min, x_max, y_max = bbox
    x_min -= padding
    y_min -= int(padding + padding/2)
    x_max += padding
    y_max += int(padding + padding/2)
    return x_min <= x <= x_max and y_min <= y <= y_max

# Function to check if the bee enters or exits a hole
def detect_entry_exit(bee_trajectory, hole_bboxes, window_size=3, padding=20):
    """
    Detects if the bee enters or exits any of the holes based on its trajectory.
    Args:
        bee_trajectory: List of (x, y) positions representing the bee's movement over time.
        hole_bboxes: Dictionary of hole bounding boxes {hole_id: (x_min, y_min, x_max, y_max)}.
        window_size: Number of frames to consider for analyzing entry/exit.
    
    Returns:
        A dictionary indicating whether the bee entered or exited a hole, with the hole id.
    """

    if len(bee_trajectory) < window_size :
        window_size = int(len(bee_trajectory)/2)

    # Define the start and end trajectories for analysis
    start_trajectory = bee_trajectory[:window_size]  # First few frames
    end_trajectory = bee_trajectory[-window_size:]   # Last few frames
    
    # Loop through each hole
    start_id = -1
    for hole_id, bbox in hole_bboxes.items():
        #print(hole_id)
        
        # Check if the bee started inside the hole and later moved out (Exit)
        sti = [is_inside_bbox(pos, bbox, padding=padding) for pos in start_trajectory]
        #print(sti)
        start_inside = all(sti)
        #print(start_inside)
        if start_inside:
            start_id = hole_id
            #print(start_id)
            break  # Bee exited, no need to check further


    # Loop through each hole
    end_id = -1
    for hole_id, bbox in hole_bboxes.items():
        #print(hole_id)

        eti = [is_inside_bbox(pos, bbox, padding=padding) for pos in end_trajectory]
        #print(eti)
        end_inside = all(eti)
        #print(end_inside)  

        if end_inside:
            end_id = hole_id
            #print(end_id)
            break 


    return start_id, end_id


def detect_entry(bee_trajectory, hole_bboxes, window_size=3, padding=20):
    """
    Detects if the bee enters a hole based on its trajectory.
    Args:
        bee_trajectory: List of (x, y) positions representing the bee's movement over time.
        hole_bboxes: Dictionary of hole bounding boxes {hole_id: (x_min, y_min, x_max, y_max)}.
        window_size: Number of frames to consider for analyzing entry.
    
    Returns:
        A dictionary indicating whether the bee entered a hole, with the hole id.
    """

    if len(bee_trajectory) < window_size :
        window_size = int(len(bee_trajectory)/2)

    # Define the start and end trajectories for analysis
    start_trajectory = bee_trajectory[:window_size]  # First few frames
    end_trajectory = bee_trajectory[-window_size:]   # Last few frames
    
    # Loop through each hole
    start_id = -1
    for hole_id, bbox in hole_bboxes.items():
        #print(hole_id)
        
        # Check if the bee started inside the hole and later moved out (Exit)
        sti = [is_inside_bbox(pos, bbox, padding=padding) for pos in start_trajectory]
        #print(sti)
        start_inside = all(sti)
        #print(start_inside)
        if start_inside:
            start_id = hole_id
            #print(start_id)
            break  # Bee exited, no need to check further

    return start_id

def detect_exit(bee_trajectory, hole_bboxes, window_size=3, padding=20):
    """
    Detects if the bee exits a hole based on its trajectory.
    Args:
        bee_trajectory: List of (x, y) positions representing the bee's movement over time.
        hole_bboxes: Dictionary of hole bounding boxes {hole_id: (x_min, y_min, x_max, y_max)}.
        window_size: Number of frames to consider for analyzing exit.
    
    Returns:
        A dictionary indicating whether the bee exited a hole, with the hole id.
    """

    if len(bee_trajectory) < window_size :
        window_size = int(len(bee_trajectory)/2)

    # Define the start and end trajectories for analysis
    start_trajectory = bee_trajectory[:window_size]  # First few frames
    end_trajectory = bee_trajectory[-window_size:]   # Last few frames
    
    # Loop through each hole
    end_id = -1
    for hole_id, bbox in hole_bboxes.items():
        #print(hole_id)

        eti = [is_inside_bbox(pos, bbox, padding=padding) for pos in end_trajectory]
        #print(eti)
        end_inside = all(eti)
        #print(end_inside)  

        if end_inside:
            end_id = hole_id
            #print(end_id)
            break 

    return end_id

def getAction(movement, nest, window_size=3, padding=20):
    start_id, end_id = detect_entry_exit(movement[1], nest['nests'], window_size=window_size, padding=padding)
    if start_id == -1 and end_id == -1:
        return None
    elif start_id != -1 and end_id == -1:
        return {
            "action": "Exit",
            "nest": f"{start_id}",
            "frame_number": movement[3][0],
            "notes" : "Bee exited the nest"
        }
    elif start_id == -1 and end_id != -1:
        return {
            "action": "Entry",
            "nest": f"{end_id}",
            "frame_number": movement[3][-1],
            "notes" : "Bee entered the nest"
        }
    elif start_id != -1 and end_id != -1:
        return [{
            "action": "Exit",
            "nest": f"{start_id}",
            "frame_number": movement[3][0],
            "notes" : f"Bee exited the nest to move to another hole { end_id }"
            }, 
            {
                "action": "Entry",
                "nest": f"{end_id}",
                "frame_number": movement[3][-1],
                "notes" : f"Bee entered the nest from another hole { start_id }"
            }
        
        ]
    
    # elif start_id != end_id and end_id != -1:
    #     return {
    #         "action": "Exit",
    #         "nest": f"{start_id}",
    #         "frame_number": movement[3][0]
    #     }
    
def getEntryAction(movement, nest, window_size=3, padding=20):
    start_id = detect_entry(movement[1], nest['nests'], window_size=window_size, padding=padding)
    if start_id == -1:
        return None
    return {
        "action": "Entry",
        "nest": f"{start_id}",
        "frame_number": movement[3][0]
    }

def getExitAction(movement, nest, window_size=3, padding=20):
    end_id = detect_exit(movement[1], nest['nests'], window_size=window_size, padding=padding)
    if end_id == -1:
        return None
    return {
        "action": "Exit",
        "nest": f"{end_id}",
        "frame_number": movement[3][-1]
    }
    
def calculate_speed(trajectory):
    """
    Calculates the speed from a given trajectory.
    Args:
        trajectory: List of (x, y) positions representing the object's movement over time.
    
    Returns:
        List of speeds corresponding to each position in the trajectory.
    """
    speeds = []
    for i in range(1, len(trajectory)):
        x1, y1 = trajectory[i-1]
        x2, y2 = trajectory[i]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = distance / 1  # Assuming time interval of 1 unit
        speeds.append(speed)
    return speeds


def calculate_acceleration(speeds):

    """
    Calculates the acceleration from a given speed trajectory.
    Args:
        speeds: List of speeds representing the object's movement over time.
    
    Returns:
        List of accelerations corresponding to each speed in the trajectory.
    """
    accelerations = []
    for i in range(1, len(speeds)):
        speed1 = speeds[i-1]
        speed2 = speeds[i]
        acceleration = speed2 - speed1
        accelerations.append(acceleration)
    return accelerations

def check_start_and_end_speed(movement):
    """
    Checks the start and end speed of the movement.
    Args:
        movement: List of (x, y) positions representing the object's movement over time.
    
    Returns:
        Tuple of start and end speed.
    """
    speeds = calculate_speed(movement[1])
    return speeds[0], speeds[-1]

def isEntry(movement, start_speed_threshold=10, end_speed_threshold=10):
    start_speed , end_speed = check_start_and_end_speed(movement)
    #if start_speed > start_speed_threshold and end_speed < end_speed_threshold:
    if end_speed < end_speed_threshold:
        return True
    
def isExit(movement, start_speed_threshold=10, end_speed_threshold=10):
    start_speed, end_speed = check_start_and_end_speed(movement)
    #if start_speed < start_speed_threshold and end_speed > end_speed_threshold:
    if start_speed < start_speed_threshold:
        return True
    
def isEntryAndExit(movement, start_speed_threshold=5, end_speed_threshold=5):
    start_speed, end_speed = check_start_and_end_speed(movement)
    if start_speed < start_speed_threshold and end_speed < end_speed_threshold:
        return True

def processTracking(motion, nest):

    movements = []
    for period in motion.tracks:
        for track in period:
            movements.append(track)

    actions = []
    for movement in movements:
        if len(movement[1]) < 5:
            continue
        if isExit(movement):
            action = getAction(movement, nest, 3, 20)
            if action:
                #for a in action:
                if type(action) == list:
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)
        elif isEntry(movement):
            action = getAction(movement, nest, 6, 10)
            if action:
                #for a in action:
                if type(action) == list:
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)
                
        else:
            action = getAction(movement, nest, 3, 20)
            if action:
                #for a in action:
                if type(action) == list:
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)

        

    #print(actions)
    return pd.DataFrame(actions)



def processYoloTracks(movements, nest):

    # movements = []
    # for period in motion.tracks:
    #     for track in period:
    #         movements.append(track)

    actions = []
    for movement in movements:
        if len(movement[1]) < 5:
            continue
        if isExit(movement):
            action = getAction(movement, nest, 3, 20)
            if action:
                #for a in action:
                if type(action) == list:
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)
        elif isEntry(movement):
            action = getAction(movement, nest, 6, 10)
            if action:
                #for a in action:
                if type(action) == list:
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)
                
        else:
            action = getAction(movement, nest, 3, 20)
            if action:
                #for a in action:
                if type(action) == list:
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)

        

    #print(actions)
    return pd.DataFrame(actions)
