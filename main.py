import cv2
import logging
import argparse
import numpy as np
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
OCCUPANCY_THRESHOLD_SECONDS = 5  
EMPTY_DEBOUNCE_SECONDS = 4

TABLE_ROI = (302, 518, 1176, 1051) # x, y, w, h
UPPER_SEAT_ROI = (601, 275, 1172, 703)
LOWER_SEAT_ROI = (277, 869, 1062, 1080)

# Separating axis theorem
def is_overlapping(zone: tuple, person_box: tuple) -> bool:
    x, y, w, h = zone
    zx1, zy1, zx2, zy2 = x, y, x + w, y + h
    bx1, by1, bx2, by2 = person_box

    return bx1 < zx2 and by1 < zy2 and bx2 > zx1 and by2 > zy1


def main():
    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Source video path')
    args = parser.parse_args()

    # Load model
    model = YOLO("yolov8n.pt")

    # Capture video
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(f'Error: Cannot open video file: {args.video}')
    
    # Get video info
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Occupancy and emptiness frames data
    occupancy_threshold_frames = int(fps * OCCUPANCY_THRESHOLD_SECONDS)
    occupancy_frames_timer = 0
    empty_threshold_frames = int(fps * EMPTY_DEBOUNCE_SECONDS)
    empty_frames_timer = 0

    logger.info(f"Video FPS: {fps}, Occupancy Threshold (frames): {occupancy_threshold_frames}")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Zone state machine variable
    table_state = "EMPTY"
    
    # Track continuous detection frames for occupancy threshold
    occupancy_frames_timer = 0
    
    # Colors for visualization
    COLOR_EMPTY = (0, 255, 0)      # Green
    COLOR_OCCUPIED = (0, 0, 255)   # Red
    COLOR_ROI_OUTLINE = (255, 255, 0) # Yellow for ROI boundaries

    # Main loop - get frame, do operations, write frame to a new video and show each modified frame.
    while True:
        ret, frame = capture.read()
        if not ret:
            logger.info('Video ended')
            break

        # Detect people
        results = model(
            frame,
            classes=[0],
            conf=0.4,
            verbose=False
        )
        boxes = results[0].boxes 

        # State machine logic
        is_anyone_in_zone = False
        # Check if any person overlaps with any of the defined zones (table + seats)
        for box in boxes:
            if is_overlapping(TABLE_ROI, box.xyxy[0]):
                is_anyone_in_zone = True
                break
            elif is_overlapping(UPPER_SEAT_ROI, box.xyxy[0]):
                is_anyone_in_zone = True
                break
            elif is_overlapping(LOWER_SEAT_ROI, box.xyxy[0]):
                is_anyone_in_zone = True
                break

        # State machine transitions
        if table_state == "OCCUPIED":
            if not is_anyone_in_zone:
                empty_frames_timer += 1
                # Table is empty if empty frames reach empty threshold
                if empty_frames_timer >= empty_threshold_frames:
                    table_state = "EMPTY"
                    occupancy_frames_timer = 0
                    logger.info(f"[STATE CHANGE] Table became EMPTY")
            else:
                # Reset empty frames timer when someone is detected again
                empty_frames_timer = 0
                
        elif table_state == "EMPTY":
            # When empty, track continuous occupancy frames
            if is_anyone_in_zone:
                occupancy_frames_timer += 1
                # Table is occupied if occupied frames rech occupied threshold
                if occupancy_frames_timer >= occupancy_threshold_frames:
                    table_state = "OCCUPIED"
                    logger.info(f"[STATE CHANGE] Table became OCCUPIED")
            else:
                # Reset occupancy frames when no one is detected
                occupancy_frames_timer = 0

        
        # Draw ROI outlines to verify alignment in output video
        cv2.rectangle(frame, TABLE_ROI[:2], (TABLE_ROI[0]+TABLE_ROI[2], TABLE_ROI[1]+TABLE_ROI[3]), COLOR_ROI_OUTLINE, 1)
        cv2.rectangle(frame, UPPER_SEAT_ROI[:2], (UPPER_SEAT_ROI[0]+UPPER_SEAT_ROI[2], UPPER_SEAT_ROI[1]+UPPER_SEAT_ROI[3]), COLOR_ROI_OUTLINE, 1)
        cv2.rectangle(frame, LOWER_SEAT_ROI[:2], (LOWER_SEAT_ROI[0]+LOWER_SEAT_ROI[2], LOWER_SEAT_ROI[1]+LOWER_SEAT_ROI[3]), COLOR_ROI_OUTLINE, 1)

        # Draw table status text
        status_color = COLOR_OCCUPIED if table_state == "OCCUPIED" else COLOR_EMPTY
        cv2.putText(frame, f"Table Status: {table_state}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

        # Draw detected people boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Check overlap with any of the relevant zones
            is_in_zone = False
            if is_overlapping(TABLE_ROI, box.xyxy[0]):
                is_in_zone = True
            elif is_overlapping(UPPER_SEAT_ROI, box.xyxy[0]):
                is_in_zone = True
            elif is_overlapping(LOWER_SEAT_ROI, box.xyxy[0]):
                is_in_zone = True
            
            # Person box is red if in zone, grey otherwise
            color = (0, 0, 255) if is_in_zone else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Detected person confidence
            label = f'Person {float(box.conf[0]):.2f}'
            cv2.putText(frame, label, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Write frame and display
        out.write(frame)
        cv2.imshow('Table Occupancy Monitor', frame)

        # Quit frames preview on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
