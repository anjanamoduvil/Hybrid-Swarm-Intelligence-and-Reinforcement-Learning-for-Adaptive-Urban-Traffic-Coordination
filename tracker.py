import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(box1_area + box2_area - intersection_area)

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


class KalmanBoxTracker:
    """
    Tracks a single bounding box state using a Constant Velocity Kalman Filter.
    State representation:
        x = [u, v, s, r, u_dot, v_dot, s_dot]^T
    where (u, v) is box center, s is area, r is aspect ratio (w/h), and dots are velocities.
    """
    def __init__(self, bbox):
        # State vector
        self.x = np.zeros((7, 1))
        self.x[:4, 0] = self.bbox_to_z(bbox)
        
        # State Covariance matrix (high uncertainty initially)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # High initial uncertainty on velocity components

        # Transition matrix
        self.F = np.eye(7)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement matrix (only observes u, v, s, r)
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Noise matrices
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01  # process noise on velocities
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # measurement noise on scale/aspect-ratio

        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.age = 0

    def predict(self):
        """
        Advances the state vector and covariance matrix using the motion model.
        """
        # Maintain aspect ratio constraints
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.z_to_bbox(self.x[:4, 0]))
        return self.history[-1]

    def update(self, bbox):
        """
        Updates the state vector with a new detection.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        
        z = self.bbox_to_z(bbox).reshape(4, 1)
        
        # Kalman gain & update equations
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = np.dot((np.eye(7) - np.dot(K, self.H)), self.P)

    def bbox_to_z(self, bbox):
        """
        Converts bbox from [x1, y1, x2, y2] to measurement space [u, v, s, r].
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        u = bbox[0] + w / 2.0
        v = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 0.0
        return np.array([u, v, s, r])

    def z_to_bbox(self, z):
        """
        Converts state space [u, v, s, r] back to [x1, y1, x2, y2] format.
        """
        u, v, s, r = z
        if s <= 0:
            s = 1.0
        if r <= 0:
            r = 1.0
        w = np.sqrt(s * r)
        h = s / w
        x1 = u - w / 2.0
        y1 = v - h / 2.0
        x2 = u + w / 2.0
        y2 = v + h / 2.0
        return np.array([x1, y1, x2, y2])

    def get_velocity(self):
        """
        Returns estimated velocity vector components (dx/dt, dy/dt) in pixels per frame.
        """
        dx = self.x[4, 0]
        dy = self.x[5, 0]
        return float(np.sqrt(dx**2 + dy**2))


class VehicleTracker:
    """
    Manages active KalmanBoxTracker instances and associates new detections using IoU matching.
    """
    def __init__(self, max_age=5, min_hits=2, queue_speed_threshold=1.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.queue_speed_threshold = queue_speed_threshold
        self.trackers = []
        self.next_id = 1

    def update(self, bboxes):
        """
        Accepts detection bounding boxes [[x1, y1, x2, y2], ...] and updates tracking list.
        Returns:
            list: A list of dict containing {"id": int, "box": (x1, y1, x2, y2), "speed": float, "is_queued": bool}
        """
        # 1. Get predictions for all active trackers
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())

        # 2. Match predicted boxes against new detections using IoU and the Hungarian algorithm
        matches, unmatched_detections, unmatched_trackers = self.associate_detections(bboxes, predicted_boxes)

        # 3. Update matched trackers
        for t_idx, d_idx in matches:
            self.trackers[t_idx].update(bboxes[d_idx])

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(bboxes[d_idx])
            new_tracker.id = self.next_id
            self.next_id += 1
            self.trackers.append(new_tracker)

        # 5. Clean up old trackers (time_since_update > max_age)
        active_trackers = []
        output_tracks = []

        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                active_trackers.append(t)
                
                # Only output tracks that have been active long enough to prevent noise
                if t.hits >= self.min_hits or t.age < self.min_hits:
                    box = t.z_to_bbox(t.x[:4, 0])
                    speed = t.get_velocity()
                    
                    # Estimate if queued
                    is_queued = speed < self.queue_speed_threshold

                    output_tracks.append({
                        "id": t.id,
                        "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        "speed": round(speed, 2),
                        "is_queued": is_queued
                    })
        
        self.trackers = active_trackers
        return output_tracks

    def associate_detections(self, detections, predictions, iou_threshold=0.3):
        """
        Associates detections to trackers using Hungarian linear sum assignment on IoU distance.
        """
        if len(predictions) == 0:
            return [], list(range(len(detections))), []

        # IoU Cost Matrix
        iou_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float32)
        for t, pred in enumerate(predictions):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = calculate_iou(pred, det)

        # Solve Hungarian allocation (minimize 1 - IoU)
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

        matches = []
        unmatched_trackers = list(range(len(predictions)))
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        return matches, unmatched_detections, unmatched_trackers
