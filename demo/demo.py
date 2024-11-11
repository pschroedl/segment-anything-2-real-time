import os
import torch
import numpy as np
import cv2
from flask import Flask, Response
import threading
from sam2.build_sam import build_sam2_camera_predictor

# URL for accessing the raw webcam feed from the local machine
video_feed_url = "http://66.27.122.32:5050/video_feed?key=12903hjk1230"

# Flask app to serve the processed video stream
app = Flask(__name__)

# Frame lock and processed frame storage
frame_lock = threading.Lock()
processed_frame = None

# Load SAM2 model and configure for GPU
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_camera_predictor(sam2_checkpoint, model_cfg)

# Function to capture and process the frames from the raw feed
def process_frames():
    global processed_frame
    cap = cv2.VideoCapture(video_feed_url)

    if not cap.isOpened():
        raise RuntimeError("Unable to open video feed from URL.")

    if_init = False
    frame_count = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            frame_count += 1

            # Only perform initialization on the first frame
            if not if_init:
                predictor.load_first_frame(frame)
                if_init = True
                obj_id = 1
                frame_idx = 0

                # Define the point prompt at one-third from the right, centered vertically
                point = [int(width * 2 / 3), int(height / 2)]
                points = [point]
                labels = [1]  # Positive prompt

                # Initialize segmentation with the point prompt
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx, obj_id, points=points, labels=labels)
            else:
                # Track the object in subsequent frames
                out_obj_ids, out_mask_logits = predictor.track(frame)

            # Process output mask only if it's non-empty
            if out_mask_logits.shape[0] > 0:
                mask = (out_mask_logits[0, 0] > 0).cpu().numpy().astype("uint8") * 255
            else:
                mask = np.zeros((height, width), dtype="uint8")

            # Invert and prepare the mask for overlay (not displayed)
            inverted_mask_colored = cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)
            overlayed_frame = cv2.addWeighted(frame, 0.7, inverted_mask_colored, 0.3, 0)

            # Update the processed frame for Flask
            with frame_lock:
                processed_frame = overlayed_frame

    cap.release()

# Start a background thread to process the frames continuously
processing_thread = threading.Thread(target=process_frames)
processing_thread.daemon = True
processing_thread.start()

# Flask endpoint to stream the processed video
@app.route('/processed_feed')
def processed_feed():
    def generate_processed_frames():
        global processed_frame
        while True:
            with frame_lock:
                if processed_frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
