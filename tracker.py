from deep_sort_realtime.deepsort_tracker import DeepSort
import math


class VehicleTracker:

    def __init__(
            self,
            max_age=15,
            n_init=3,
            max_cosine_distance=0.4,
            queue_speed_threshold=2.0,
            history_length=50
    ):

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance
        )

        # Queue threshold (pixels/frame)
        self.queue_speed_threshold = queue_speed_threshold

        # Maximum trajectory length
        self.history_length = history_length

        # Previous center of every vehicle
        self.previous_centers = {}

       

        # Queue start frame
        self.queue_start = {}

        # Total travelled distance
        self.total_distance = {}

        # Maximum speed
        self.max_speed = {}

        # Average speed calculation
        self.speed_history = {}

        # First frame where vehicle appeared
        self.first_seen = {}

        # Last frame where vehicle appeared
        self.last_seen = {}

        # Current frame number
        self.frame_count = 0

       

       

        # Complete trajectory history
        self.track_history = {}

    def update(self, detections, frame):

     # Increment frame number
     self.frame_count += 1

     """
     detections format...
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
     active_ids = set()
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
            active_ids.add(track_id)
            if track_id not in self.previous_centers:

               self.previous_centers[track_id] = center

               self.track_history[track_id] = [center]

               self.total_distance[track_id] = 0.0

               self.max_speed[track_id] = 0.0

               self.speed_history[track_id] = []

               self.first_seen[track_id] = self.frame_count

               self.last_seen[track_id] = self.frame_count

               self.queue_start[track_id] = None
            ###################################################
            # Speed estimation (pixels/frame)
            ###################################################

            speed = 0.0
            direction = "Stationary"

            prev_center = self.previous_centers[track_id]

            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]

            speed = math.sqrt(dx * dx + dy * dy)


            ###################################################
            # Total Distance
            ###################################################

            self.total_distance[track_id] += speed
            ###################################################
            # Queue Detection
            ###################################################

            is_queued = speed < self.queue_speed_threshold

            if is_queued:

                if self.queue_start[track_id] is None:
                    self.queue_start[track_id] = self.frame_count

                queue_time = self.frame_count - self.queue_start[track_id]

            else:

                  self.queue_start[track_id] = None

                  queue_time = 0

            ###################################################
             # Speed History
            ###################################################

             
            self.speed_history[track_id].append(speed)

            if len(self.speed_history[track_id]) > self.history_length:
                 self.speed_history[track_id].pop(0)

            ###################################################
            # Average Speed
            ###################################################

            avg_speed = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])

             ###################################################
            # Maximum Speed
            ###################################################

            if speed > self.max_speed[track_id]:
                 self.max_speed[track_id] = speed

            ###################################################
            # Direction
            ###################################################

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
            # Save history
            ###################################################

            self.previous_centers[track_id] = center

            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append(center)

            ###################################################
            # Keep last 50 trajectory points
            ###################################################

            if len(self.track_history[track_id]) > self.history_length:
                self.track_history[track_id].pop(0)
            self.last_seen[track_id] = self.frame_count 
           ###################################################
           # Vehicle Lifetime
          ###################################################

            frames_tracked = self.frame_count - self.first_seen[track_id] + 1

            tracked.append({

                  "id": track_id,

                   "box": (
                          x1,
                           y1,
                          x2,
                          y2
                         ),

                "center": center,

                "speed": round(speed,2),

               "avg_speed": round(avg_speed,2),

              "max_speed": round(self.max_speed[track_id],2),

             "distance": round(self.total_distance[track_id],2),

             "direction": direction,

             "is_queued": is_queued,

            "queue_time": queue_time,

             "frames_tracked": frames_tracked,

            "trajectory": self.track_history[track_id],
             "frame": self.frame_count,   

            })
      ###################################################
    # Cleanup old tracks
    ###################################################

     stored_ids = list(self.previous_centers.keys())

     for old_id in stored_ids:

         if old_id not in active_ids:

            self.previous_centers.pop(old_id, None)
            self.track_history.pop(old_id, None)
            self.queue_start.pop(old_id, None)
            self.total_distance.pop(old_id, None)
            self.max_speed.pop(old_id, None)
            self.speed_history.pop(old_id, None)
            self.first_seen.pop(old_id, None)
            self.last_seen.pop(old_id, None)
     return tracked
