from deep_sort_realtime.deepsort_tracker import DeepSort
import math


class VehicleTracker:
    def __init__(self,
                 max_age=15,
                 n_init=1,
                 max_cosine_distance=0.5,
                 queue_speed_threshold=2):

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance
        )

        self.queue_speed_threshold = queue_speed_threshold

        # Previous center of each tracked vehicle
        self.previous_centers = {}

        # Complete trajectory history
        self.track_history = {}

    def update(self, detections, frame):
        """
        detections format:
        [
            [x1, y1, x2, y2, confidence],
            ...
        ]
        """

        ds_detections = []

        for det in detections:
            x1, y1, x2, y2, conf = det

            w = x2 - x1
            h = y2 - y1

            ds_detections.append(
                ([x1, y1, w, h], conf, "vehicle")
            )

        tracks = self.tracker.update_tracks(
            ds_detections,
            frame=frame
        )

        tracked = []

        for track in tracks:

            # Ignore tentative tracks
            if not track.is_confirmed():
                continue

            # Ignore predictions that weren't updated this frame
            if track.time_since_update > 0:
                continue

            x1, y1, x2, y2 = track.to_ltrb()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            track_id = track.track_id

            ###################################################
            # Speed estimation (pixels/frame)
            ###################################################

            speed = 0.0
            direction = "Stationary"

            if track_id in self.previous_centers:

                px, py = self.previous_centers[track_id]

                dx = center[0] - px
                dy = center[1] - py

                speed = math.sqrt(dx * dx + dy * dy)

                if abs(dx) > abs(dy):

                    if dx > 0:
                        direction = "Right"
                    elif dx < 0:
                        direction = "Left"

                else:

                    if dy > 0:
                        direction = "Down"
                    elif dy < 0:
                        direction = "Up"

            ###################################################
            # Queue Detection
            ###################################################

            is_queued = speed < self.queue_speed_threshold

            ###################################################
            # Save history
            ###################################################

            self.previous_centers[track_id] = center

            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append(center)

            ###################################################
            # Keep last 50 trajectory points
            ###################################################

            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id].pop(0)

            ###################################################
            # Output
            ###################################################

            tracked.append({

                "id": track_id,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": center,

                "speed": round(speed, 2),

                "direction": direction,

                "is_queued": is_queued,

                "trajectory": self.track_history[track_id]

            })

        return tracked
