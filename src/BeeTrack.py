import numpy as np

def compute_centroid(bbox):
    # bbox is in the format (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def associate_detections(tracked_objects, detections,  threshold=50):
    """
    Associate detections with tracked objects based on the nearest centroid.

    Args:
    detections: List of bounding boxes for detected objects [(x1, y1, x2, y2), ...]
    tracked_objects: List of predicted bounding boxes [(x1, y1, x2, y2), ...]

    Returns:
    List of associations [(det_index, track_index), ...]
    """
    associations = []
    for i, det in enumerate(detections):
        det_centroid = compute_centroid(det)

        min_dist = float('inf')
        best_match = None

        for j, track in enumerate(tracked_objects):
            track_centroid = compute_centroid(track[0])
            distance = np.linalg.norm(np.array(det_centroid) - np.array(track_centroid))

            if distance < min_dist:
                min_dist = distance
                best_match = j

        # Associate detection i with tracked object best_match if distance is below a threshold
        if min_dist < threshold:  # You can tune the threshold based on your scenario
            associations.append((i, best_match))

    return associations



def compute_iou(box1, box2):
    # box1, box2 are in (x1, y1, x2, y2) format
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def associate_detections_iou(tracked_objects, detections,  iou_threshold=0.5):

    associations = []
    for i, det in enumerate(detections):
        best_iou = 0
        best_match = None

        for j, track in enumerate(tracked_objects):
            iou = compute_iou(det, track[0])

            if iou > best_iou:
                best_iou = iou
                best_match = j

        if best_iou > iou_threshold:  # Tune the IoU threshold
            associations.append((i, best_match))

    return associations


from scipy.optimize import linear_sum_assignment

def Hungarian_algorithm(tracked_objects, detections, threshold=200):
    """
    Hungarian algorithm for optimal assignment.
    """

    # Cost matrix: rows are tracked objects, columns are detections
    cost_matrix = np.zeros((len(tracked_objects), len(detections)))

    for i, track in enumerate(tracked_objects):
        for j, det in enumerate(detections):
            # Use Euclidean distance between the predicted position and the detection
            track_centroid = compute_centroid(track[0]) # The predicted state is the first element of the track
            det_centroid = compute_centroid(det)
            distance = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
            cost_matrix[i, j] = distance

    # Solve the assignment problem using the Hungarian algorithm
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    associations = []
    for track_index, det_index in zip(track_indices, det_indices):
        if cost_matrix[track_index, det_index] < threshold:
            associations.append((det_index, track_index))

    return associations


import numpy as np

def predict_position(D_1, D_2, distance_threshold=50):
    """
    Predict the position of the insect in the next frame using the previous two positions.
    
    Args:
    D_k_1: Tuple (x, y, x1, y2) - Detected position of the insect in the last frame (k-1)
    D_k_2: Tuple (x, y, x1, y2) - Detected position of the insect in the second last frame (k-2)
    
    Returns:
    Tuple (x_p, y_p, x_p1, y_p2) - Predicted position in the next frame (k)
    """

    # extrack height and width
    height = 20 #(D_k_1[0] - D_k_1[2])/2
    width = 20 #(D_k_1[1] - D_k_2[3])/2

    # Compute the centroids of the detected positions
    D_k_1 = compute_centroid(D_1)
    D_k_2 = compute_centroid(D_2)

    # Matrix A as described in the equation
    A = np.array([
        [2, 0, -1, 0],
        [0, 2, 0, -1]
    ])
    
    # Flatten the positions into a single vector [x_k-1, y_k-1, x_k-2, y_k-2]
    D_vector = np.array([D_k_1[0], D_k_1[1], D_k_2[0], D_k_2[1]])
    
    # Perform matrix multiplication to predict the next position
    predicted_position = np.dot(A, D_vector)

    predict_position = (predicted_position[0]- width , predicted_position[1] - height, predicted_position[0] + width, predicted_position[1]+ height)

   # caclulate the distance between the predicted position and the last detected position
    distance = np.linalg.norm(np.array(D_k_1) - np.array(predicted_position))
    # if the distance is too large, we return the last detected position
    if distance > distance_threshold:
        return D_1
    else:
        return predict_position


class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id
        self.age = 0
        self.dead = False
        self.trajectory = []
        self.trajectory_frame_numbers = []
        self.is_DL_predictions = []

    def predict(self, distance_threshold):
        #self.age += 1
        if len(self.trajectory) > 1:
            #self.bbox = predict_position(self.trajectory[-1], self.trajectory[-2])
            self.bbox = predict_position(self.trajectory[-1], self.trajectory[-2], distance_threshold)

        else:
            self.bbox = predict_position(self.bbox, self.bbox)

    def update(self, bbox, frame_number, is_DL_prediction=False):
        if self.dead == False:
            self.bbox = bbox
            self.trajectory.append(bbox)
            self.trajectory_frame_numbers.append(frame_number)
            self.age = 0
            self.is_DL_predictions.append(is_DL_prediction)

    def get_state(self):
        return self.bbox, self.track_id
    
    def get_trajectory(self):
        #trajectory = self.trajectory
        
        # only keep trajector where the prediction is a DL prediction
        trajectory = [self.trajectory[i] for i in range(len(self.trajectory)) if self.is_DL_predictions[i]]
        trajectory_centroids = [compute_centroid(bbox) for bbox in trajectory]
        trajectory_frame_numbers = [self.trajectory_frame_numbers[i] for i in range(len(self.trajectory_frame_numbers)) if self.is_DL_predictions[i]]
        return self.track_id, trajectory_centroids, trajectory, trajectory_frame_numbers

class Tracker:
    def __init__(self, max_age=30, track_start_id = 0, distance_threshold=50, association_threshold=150):
        self.max_age = max_age
        self.objects = []
        self.next_id = track_start_id
        self.association_threshold = distance_threshold
        self.distance_threshold = association_threshold

    def getTracks(self):
        return [obj.get_trajectory() for obj in self.objects]

    def getNumOfLiveTracks(self):
        return len([obj for obj in self.objects if obj.dead != True])

    def update(self, detections, frame_number):
        # First, predict the state of all existing objects
        for obj in self.objects:
            obj.predict(distance_threshold=self.distance_threshold)

        # Associate detections with existing objects
        if len(detections) > 0:
            if len(self.objects) == 0: # If there are no existing objects, create new ones
                for det in detections:
                    self.objects.append(Track(det, self.next_id))
                    self.next_id += 1
            else:
                # Use the Hungarian algorithm to associate detections to existing objects
                associations = Hungarian_algorithm([obj.get_state() for obj in self.objects], detections, threshold=self.association_threshold)
                detections_idx = set(range(len(detections)))
                tracks_idx = set(range(len(self.objects)))

                for det_idx, track_idx in associations:
                    self.objects[track_idx].update(detections[det_idx], frame_number, True) # might need to add the frame number here
                    self.objects[track_idx].age = 0  # everytime we have an association, we reset the age
                    detections_idx.remove(det_idx)
                    try:
                        tracks_idx.remove(track_idx)
                    except:
                        print('error')
                        

                # Create new objects for unmatched detections
                for det_idx in detections_idx:
                    self.objects.append(Track(detections[det_idx], self.next_id))
                    self.next_id += 1

                # Update tracks without associated detections with predictions
                # This is useful for tracking objects that are occluded
                # Set tracks to dead if they are not associated with any detection for longer than max_age
                for track_idx in tracks_idx:
                    #self.objects[track_idx].update(self.objects[track_idx].bbox, frame_number, False) # update trakectory with the predicted position
                    self.objects[track_idx].age += 1
                    if self.objects[track_idx].age > self.max_age:
                        self.objects[track_idx].dead = True

        else: # update tracks with predictions if there is no DL detections
            for obj in self.objects:
                obj.update(obj.bbox, frame_number, False) # update the trajectory with the predicted position
                obj.age += 1
                if obj.age > self.max_age:
                    obj.dead = True
            

        # Remove dead objects
        live_objects = [obj for obj in self.objects if not obj.dead]

        return [obj.get_state() for obj in live_objects]