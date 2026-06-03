from tracker import VehicleTracker

tracker = VehicleTracker()

detections = [
    [100,100,200,200],
    [300,300,400,400]
]

tracks = tracker.update(detections)

print(tracks)
