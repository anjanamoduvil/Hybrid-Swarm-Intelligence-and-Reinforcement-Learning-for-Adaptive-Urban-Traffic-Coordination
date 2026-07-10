class TrafficMetrics:
    """
    Computes traffic statistics from the tracker output.
    """

    def __init__(self):

        self.total_vehicle_count = 0

        self.previous_vehicle_ids = set()

    def compute(self, tracked_objects):
        """
        tracked_objects is the output from tracker.update()

        Returns a dictionary containing useful traffic statistics.
        """

        vehicle_count = len(tracked_objects)

        ####################################################
        # Count new vehicles
        ####################################################

        current_ids = set()

        for vehicle in tracked_objects:

            current_ids.add(vehicle["id"])

        new_ids = current_ids - self.previous_vehicle_ids

        self.total_vehicle_count += len(new_ids)

        self.previous_vehicle_ids = current_ids

        ####################################################
        # Average speed
        ####################################################

        if vehicle_count > 0:

            average_speed = sum(
                vehicle["speed"]
                for vehicle in tracked_objects
            ) / vehicle_count

            maximum_speed = max(
                vehicle["speed"]
                for vehicle in tracked_objects
            )

        else:

            average_speed = 0.0
            maximum_speed = 0.0

        ####################################################
        # Queue statistics
        ####################################################

        queued = [
            vehicle
            for vehicle in tracked_objects
            if vehicle["is_queued"]
        ]

        queue_length = len(queued)

        if queue_length > 0:

            average_queue_time = sum(
                vehicle["queue_time"]
                for vehicle in queued
            ) / queue_length

        else:

            average_queue_time = 0.0

        ####################################################
        # Density
        ####################################################

        if vehicle_count < 5:

            density = "Low"

        elif vehicle_count < 15:

            density = "Medium"

        else:

            density = "High"

        ####################################################
        # Congestion
        ####################################################

        if average_speed < 2 and queue_length > 10:

            congestion = "Heavy"

        elif average_speed < 5:

            congestion = "Moderate"

        else:

            congestion = "Free Flow"

        ####################################################
        # Return statistics
        ####################################################

        return {

            "vehicle_count": vehicle_count,

            "total_vehicle_count": self.total_vehicle_count,

            "average_speed": round(average_speed, 2),

            "maximum_speed": round(maximum_speed, 2),

            "queue_length": queue_length,

            "average_queue_time": round(
                average_queue_time,
                2
            ),

            "density": density,

            "congestion": congestion

        }
