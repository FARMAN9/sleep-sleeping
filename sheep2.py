import os
import cv2
import time
import threading
import queue
from ultralytics import YOLO

# ================= Config ================= #
IMAGE_DIR = "COCO/images/train2017/"   # folder with images
MODEL_PATH = "yolo11n-pose.pt"  # your trained model
MAX_QUEUE_SIZE = 5
# ========================================== #


class ImageLoaderThread:
    """Thread for loading images into queue"""
    def __init__(self, image_dir, frame_queue, max_queue_size=MAX_QUEUE_SIZE):
        self.image_dir = image_dir
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.running = True
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        image_files = [os.path.join(self.image_dir, f)
                       for f in os.listdir(self.image_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for idx, img_path in enumerate(image_files, start=1):
            if not self.running:
                break

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            while self.frame_queue.qsize() >= self.max_queue_size:
                time.sleep(0.01)

            self.frame_queue.put((idx, img_path, frame))

        self.running = False  # mark done

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class InferenceThread:
    """Thread for running YOLO inference on images"""
    def __init__(self, model_path, frame_queue, loader):
        self.model = YOLO(model_path)
        self.frame_queue = frame_queue
        self.loader = loader
        self.running = True
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                idx, img_path, frame = self.frame_queue.get()
                results = self.model.predict(frame, imgsz=640, conf=0.25, verbose=False)

                # Draw predictions
                annotated = results[0].plot()
                cv2.imshow("Sheep Behavior Detection", annotated)

                print(f"[{idx}] Processed: {os.path.basename(img_path)}")

                # wait for key or auto close after 1s
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    self.stop()
                    break

            elif not self.loader.running and self.frame_queue.empty():
                break  # done

            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ================= Main ================= #
if __name__ == "__main__":
    frame_queue = queue.Queue()

    loader = ImageLoaderThread(IMAGE_DIR, frame_queue)
    loader.start()

    inference = InferenceThread(MODEL_PATH, frame_queue, loader)
    inference.start()

    try:
        while loader.running or not frame_queue.empty():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        loader.stop()
        inference.stop()

