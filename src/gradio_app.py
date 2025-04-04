import gradio as gr
from VideoAnalyzer import VideoAnalyzer
import cv2

def video_identity(video):
    return video

nest_model = '/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/models/nest_detection_model.pt'
tracking_model = '/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/runs/detect/train10/weights/best.pt'
#tracking_model = '/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/models/bee_tracking_model.pt'
analyzer = VideoAnalyzer(nest_model, tracking_model, 720, 1280)
def processVideo(video, analyzer=analyzer):
        
        output_folder = '/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/scripts/processor/output/'

        nest_ids = analyzer.getNestDetection(video)
        nest = analyzer.processNestDetection(nest_ids)

        motion_fn = analyzer.getMotionTracking(video, nest['hotel'], False, output_folder)
        events = analyzer.processMotionTracking(motion_fn, nest)

        csv = analyzer.synthesizeCSV(events, video)

        #return 
        #vd = analyzer.synthesizeVideo(video, events, motion_fn, nest, output_folder)

        #return gr.DataFrame(csv)

        # load first frame of video
        cap = cv2.VideoCapture(video)
        success, frame = cap.read()

        # print all nest on the frame 
        for nest_hole in nest['nests']:
                    id = nest_hole.split('_')[-1]
                    x1, y1, x2, y2 = nest['nests'][nest_hole]

                    # x1 -= 10
                    # y1 -= 15
                    # x2 += 10
                    # y2 += 15

                    x1 -= 5
                    y1 -= 7
                    x2 += 5
                    y2 += 7
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                    cv2.putText(frame, f"{id}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        return [csv, frame]
    

demo = gr.Interface(processVideo,
                    gr.Video(),
                    #"dataframe",
                    ['dataframe', 'image']
                    )

if __name__ == "__main__":
    demo.launch()