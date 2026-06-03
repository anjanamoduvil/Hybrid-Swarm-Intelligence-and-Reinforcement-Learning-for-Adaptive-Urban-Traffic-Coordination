from collections import defaultdict

class VehicleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        tracked = []

        for det in detections:
            x1, y1, x2, y2 = det[:4]

            tracked.append({
                "id": self.next_id,
                "box": (x1, y1, x2, y2)
            })

            self.next_id += 1

        return tracked
