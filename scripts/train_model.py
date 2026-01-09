from ultralytics import YOLO

print("Starting training of bee tracking model...")

bee_tracking_model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking.pt')

bee_tracking_data_path = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/datasets/bee_tracking/data.yaml"

results = bee_tracking_model.train(data=bee_tracking_data_path, epochs=10, workers=8, device='mps')

print("Training complete. Results:")
print(results)

# nest_detection_model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/nest.pt')

# nest_detection_data_path = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/datasets/nest_detection/data.yaml"

# print("Starting training of nest detection model...")
# results = nest_detection_model.train(data=nest_detection_data_path, epochs=10, workers=8)

# print("Training complete. Results:")
# print(results)