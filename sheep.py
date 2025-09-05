import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from queue import Queue
import concurrent.futures

# ---------------- Configuration ---------------- #
MAX_QUEUE_SIZE = 10  # Limit queue size to prevent memory issues
FRAME_SKIP = 1  # Process every nth frame
MODEL_SIZE = 320  # Smaller size for faster inference
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold

# ---------------- Helper Functions ---------------- #

def resize_to_screen(frame, max_width=1280, max_height=720):
    """Resize frame to fit within screen while keeping aspect ratio"""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def get_sheep_state_fast(animal_crop):
   
    
    """
    Improved classification of animal state (sleeping/awake)
    Uses multiple features for better accuracy
    """
    if animal_crop.size == 0:
        return "Unknown"
    
    h, w = animal_crop.shape[:2]
    if h < 10 or w < 10:  # Too small to analyze
        return "Unknown"
    
    # Convert to grayscale
    gray = cv2.cvtColor(animal_crop, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate features
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if w > h else h / w
    
    # Calculate extent (area ratio)
    area = cv2.contourArea(largest_contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    # Calculate solidity (area / convex hull area)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Calculate orientation using moments
    if len(largest_contour) >= 5:
        (_, _), (ma, ma_), angle = cv2.fitEllipse(largest_contour)
        orientation = angle
    else:
        orientation = 0
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(animal_crop, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # DEBUG: Print features for analysis
    # print(f"AR: {aspect_ratio:.2f}, Extent: {extent:.2f}, Solidity: {solidity:.2f}, Sat: {avg_saturation:.1f}")
    
    # Sleeping animals typically have:
    # 1. Higher aspect ratio (more elongated)
    # 2. Lower extent (less rectangular)
    # 3. Lower saturation (less vibrant colors)
    # 4. More horizontal orientation
    
    sleeping_score = 0
    
    # Aspect ratio: sleeping animals are more elongated
    if aspect_ratio > 2.0:
        sleeping_score += 2
    elif aspect_ratio > 1.5:
        sleeping_score += 1
    
    # Extent: sleeping animals have lower extent (less rectangular)
    if extent < 0.6:
        sleeping_score += 2
    elif extent < 0.7:
        sleeping_score += 1
    
    # Solidity: sleeping animals may have lower solidity
    if solidity < 0.8:
        sleeping_score += 1
    
    # Saturation: sleeping animals often have lower saturation
    if avg_saturation < 60:
        sleeping_score += 1
    
    # Orientation: sleeping animals are more horizontal
    if (orientation < 30 or orientation > 150) and orientation > 0:
        sleeping_score += 1
    
    # Value: sleeping animals might be in shadow
    if avg_value < 100:
        sleeping_score += 1
    
    # Determine state based on score
    if sleeping_score >= 4:
        return "Sleeping"
    else:
        return "Awake"

# ---------------- Thread Classes ---------------- #



import cv2

class ImageCapture:
    """Load a single image instead of video"""
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.width = 0
        self.height = 0

    def start(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Error: Could not load image.")
            return False

        self.height, self.width = self.image.shape[:2]
        return True

    def get_frame(self):
        """Return the single image once"""
        return self.image

    def stop(self):
        self.image = None


class VideoCaptureThread:
    """Thread for capturing video frames"""
    def __init__(self, video_source, frame_queue, max_queue_size=MAX_QUEUE_SIZE):
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.running = True
        self.cap = None
        self.fps = 0
        self.width = 0
        self.height = 0
        
    def start(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return False
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        return True
        
    def run(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
                
            # Limit queue size to prevent memory issues
            if self.frame_queue.qsize() < self.max_queue_size:
                self.frame_queue.put((frame_count, frame))
            else:
                # Drop frames if queue is full to maintain real-time performance
                time.sleep(0.001)
                
        self.cap.release()
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class ProcessingThread:
    """Thread for processing frames with YOLO"""
    def __init__(self, frame_queue, processed_queue, model_path="yolo11n-pose.pt"): # model_path="yolo11n-pose.pt" or "yolo11n.pt"
        self.frame_queue = frame_queue
        self.processed_queue = processed_queue
        self.model_path = model_path
        self.running = True
        self.model = None
        
    def start(self):
        # Load model in the same thread to avoid issues
        self.model = YOLO(self.model_path)
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        
    def run(self):
        # Pre-define colors
        sleep_color = (0, 0, 255)  # Red for sleeping
        awake_color = (0, 255, 0)  # Green for awake
        
        # Use ThreadPoolExecutor for parallel processing of detections
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while self.running:
                try:
                    # Get frame with timeout
                    frame_data = self.frame_queue.get(timeout=0.1)
                    frame_count, frame = frame_data
                    
                    # Process frame
                    results = self.model(frame, imgsz=MODEL_SIZE, verbose=False, conf=CONFIDENCE_THRESHOLD)
                    
                    processed_frame = frame.copy()
                    for result in results:
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confs = result.boxes.conf.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()

                            # Process detections in parallel
                            futures = []
                            for box, conf, cls in zip(boxes, confs, classes):
                                x1, y1, x2, y2 = map(int, box)
                                label = self.model.names[int(cls)]

                                # Detect both sheep and sheeps
                                if label == "sheep" and conf > CONFIDENCE_THRESHOLD:
                                    futures.append(executor.submit(
                                        self.process_detection, 
                                        processed_frame, x1, y1, x2, y2, sleep_color, awake_color
                                    ))
                            
                            # Wait for all detections to complete
                            for future in concurrent.futures.as_completed(futures):
                                future.result()
                    
                    # Put processed frame in queue
                    if self.processed_queue.qsize() < MAX_QUEUE_SIZE:
                        self.processed_queue.put((frame_count, processed_frame))
                        
                except Exception as e:
                    # Queue empty or other error, continue
                    continue
                    
    def process_detection(self, frame, x1, y1, x2, y2, sleep_color, awake_color):
        """Process a single detection (can be called in parallel)"""
        # Crop sheep with padding check
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(frame.shape[0], y2)
        x2 = min(frame.shape[1], x2)
        
        sheep_crop = frame[y1:y2, x1:x2]
        
        # Determine state (sleeping/awake)
        state = get_sheep_state_fast(sheep_crop)

        # Draw bounding box
        color = sleep_color if state == "Sleeping" else awake_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"sheep: {state}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# ---------------- Main Script ---------------- #

def main():
    # Input/output videos
    input_video = "sheep_video2.mp4"
    output_video = "sheep_output2.avi"

    # Create queues for inter-thread communication
    frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    processed_queue = Queue(maxsize=MAX_QUEUE_SIZE)

    # Create and start threads
    capture_thread = VideoCaptureThread(input_video, frame_queue)
    if not capture_thread.start():
        return
        
    processing_thread = ProcessingThread(frame_queue, processed_queue)
    processing_thread.start()

    # Get video properties for output
    fps = capture_thread.fps
    width = capture_thread.width
    height = capture_thread.height

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    fps_text = "FPS: 0"
    last_frame_time = time.time()

    print("Starting multi-threaded sheep detection. Press 'q' to quit.")

    try:
        while True:
            # Get processed frame with timeout
            try:
                frame_data = processed_queue.get(timeout=0.1)
                frame_count, processed_frame = frame_data
                
                # Calculate FPS
                current_time = time.time()
                elapsed = current_time - start_time
                if frame_count % 10 == 0:
                    fps_text = f"FPS: {frame_count/elapsed:.1f}"
                
                # Display FPS on frame
                cv2.putText(processed_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save frame to output video
                out.write(processed_frame)

                # Show frame resized to fit screen
                disp_frame = resize_to_screen(processed_frame, max_width=1280, max_height=720)
                cv2.imshow("Multi-Threaded sheep Detection", disp_frame)
                
                # Calculate processing time
                processing_time = time.time() - last_frame_time
                last_frame_time = time.time()
                
                # Control display speed
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                    
            except Exception as e:
                # Queue empty, continue
                continue
                
    except KeyboardInterrupt:
        print("Interrupted by user")
        
    finally:
        # Clean up
        capture_thread.stop()
        processing_thread.stop()
        out.release()
        cv2.destroyAllWindows()

        # Calculate final FPS
        end_time = time.time()
        total_fps = frame_count / (end_time - start_time) if end_time > start_time else 0
        print(f"✅ Processing complete. Average FPS: {total_fps:.1f}")
        print("Output saved as:", output_video)

if __name__ == "__main__":
    main()