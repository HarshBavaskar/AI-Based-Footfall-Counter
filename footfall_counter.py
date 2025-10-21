# -*- coding: utf-8 -*-
"""
Footfall Counter
YOLOv8 + DeepSORT with GPU Acceleration, Multi-threading, Frame Skipping
Performance: 2-5x faster processing
"""

import cv2
import numpy as np
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import time
import threading
from queue import Queue
import torch



class ThreadedVideoCapture:
    """Multi-threaded video capture for faster frame reading"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.q = Queue(maxsize=128)
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return
                self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def release(self):
        self.stopped = True
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened() and not self.stopped

    def get(self, prop):
        return self.cap.get(prop)

class FootfallCounter:
    """AI-powered footfall counter with GPU acceleration"""

    def __init__(self, model_path='yolov8n.pt', roi_line_y=None, confidence_threshold=0.5, 
                 use_gpu=True, half_precision=True, skip_frames=0):

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")

        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to(self.device)

        self.half_precision = half_precision and self.device == 'cuda'
        if self.half_precision:
            print("⚡ Half-precision (FP16) enabled for 2x speed boost")

        embedder_gpu = True if self.device == 'cuda' else False
        self.tracker = DeepSort(
            max_age=60, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2,
            nn_budget=100, embedder='mobilenet', half=self.half_precision,
            embedder_gpu=embedder_gpu
        )
        self.model = YOLO("yolov8n-seg.pt")
        self.confidence_threshold = confidence_threshold
        self.roi_line_y = roi_line_y
        self.track_history = defaultdict(lambda: deque(maxlen=60))
        self.counted_ids = set()
        self.entry_count = 0
        self.exit_count = 0
        self.heatmap = None
        self.heatmap_decay = 0.95
        self.heatmap_update_interval = 3
        self.frame_counter = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        self.skip_frames = skip_frames
        self.skip_counter = 0
        self.prev_frame_gray = None

        self.colors = {
            'line': (0, 255, 255), 'bbox': (0, 255, 0),
            'entry': (0, 255, 0), 'exit': (0, 0, 255), 'text': (255, 255, 255)
        }

    def _get_roi_line(self, frame_height):
        return int(frame_height * 0.5) if self.roi_line_y is None else self.roi_line_y

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _check_line_crossing(self, track_id, current_y, roi_line_y):
        history = self.track_history[track_id]
        if len(history) < 2:
            return None
        prev_y = history[-2][1]
        if prev_y < roi_line_y and current_y >= roi_line_y:
            return 'entry'
        elif prev_y > roi_line_y and current_y <= roi_line_y:
            return 'exit'
        return None

    def _should_skip_frame(self, frame):
        if self.prev_frame_gray is None:
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False
        self.skip_counter += 1
        if self.skip_counter < self.skip_frames:
            return True
        self.skip_counter = 0
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_gray = current_gray
        return False

    def _update_heatmap(self, frame, centroid):
        if self.frame_counter % self.heatmap_update_interval != 0:
            return
        height, width = frame.shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((height, width), dtype=np.float32)
        cx, cy = centroid
        sigma = 30
        y_grid, x_grid = np.ogrid[:height, :width]
        gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
        self.heatmap += gaussian * 0.5
        self.heatmap *= self.heatmap_decay
        self.heatmap = np.clip(self.heatmap, 0, 10)

    def _draw_heatmap_overlay(self, frame):
        if self.heatmap is None:
            return frame
        heatmap_normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8) if self.heatmap.max() > 0 else np.zeros_like(self.heatmap, dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

    def _calculate_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if current_time != self.last_time else 0
        self.last_time = current_time
        self.fps_history.append(fps)
        return sum(self.fps_history) / len(self.fps_history)

    def process_frame(self, frame, show_heatmap=True, show_trajectories=True, force_process=False):
        self.frame_counter += 1
        if not force_process and self._should_skip_frame(frame):
            return frame
        height, width = frame.shape[:2]
        roi_line_y = self._get_roi_line(height)
        current_fps = self._calculate_fps()

        results = self.model(
            frame, classes=[0], conf=self.confidence_threshold, verbose=False,
            device=self.device, half=self.half_precision, imgsz=640
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        if show_heatmap:
            frame = self._draw_heatmap_overlay(frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = self._get_centroid([x1, y1, x2, y2])

            if show_heatmap:
                self._update_heatmap(frame, (cx, cy))
            self.track_history[track_id].append((cx, cy))

            if track_id not in self.counted_ids:
                crossing = self._check_line_crossing(track_id, cy, roi_line_y)
                if crossing == 'entry':
                    self.entry_count += 1
                    self.counted_ids.add(track_id)
                    color = self.colors['entry']
                elif crossing == 'exit':
                    self.exit_count += 1
                    self.counted_ids.add(track_id)
                    color = self.colors['exit']
                else:
                    color = self.colors['bbox']
            else:
                color = self.colors['bbox']

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if show_trajectories and len(self.track_history[track_id]) > 1:
                points = list(self.track_history[track_id])
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, thickness)
            cv2.circle(frame, (cx, cy), 5, color, -1)

        cv2.line(frame, (0, roi_line_y), (width, roi_line_y), self.colors['line'], 3)
        cv2.putText(frame, "COUNTING LINE", (10, roi_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['line'], 2)
        self._draw_statistics(frame, current_fps, len(tracks))
        return frame

    def _draw_statistics(self, frame, fps, active_tracks):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        total = self.entry_count + self.exit_count
        stats = [
            ("ENTRIES", self.entry_count, self.colors['entry']),
            ("EXITS", self.exit_count, self.colors['exit']),
            ("TOTAL", total, self.colors['text']),
            ("ACTIVE", active_tracks, (255, 165, 0)),
            ("FPS", f"{fps:.1f}", (0, 255, 255))
        ]
        y_offset = 45
        for i, (label, value, color) in enumerate(stats):
            cv2.putText(frame, label, (20, y_offset + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, str(value), (250, y_offset + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def process_video(self, input_path, output_path, show_heatmap=True, show_trajectories=True, status_callback=None):
        cap = ThreadedVideoCapture(input_path).start()
        time.sleep(1.0)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.process_frame(frame, show_heatmap, show_trajectories)
                out.write(processed_frame)
                frame_count += 1
                if status_callback and frame_count % 10 == 0:
                    progress = int((frame_count / total_frames) * 100)
                    status_callback({
                        'status': 'processing', 'progress': progress,
                        'entry_count': self.entry_count, 'exit_count': self.exit_count
                    })
        finally:
            cap.release()
            out.release()
        return {
            'entry_count': self.entry_count, 'exit_count': self.exit_count,
            'total_count': self.entry_count + self.exit_count, 'frames_processed': frame_count
        }
