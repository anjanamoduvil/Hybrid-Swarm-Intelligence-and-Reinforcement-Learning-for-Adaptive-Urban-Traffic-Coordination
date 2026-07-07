class TrafficState:
    """
    Converts traffic metrics into a discrete state for the RL agent.
    """

    def get_state(self, traffic_stats):

        vehicle_count = traffic_stats["vehicle_count"]
        avg_speed = traffic_stats["average_speed"]
        queue_length = traffic_stats["queue_length"]
        avg_queue_time = traffic_stats["average_queue_time"]

        ####################################################
        # Vehicle Count Level
        ####################################################

        if vehicle_count < 5:
            vehicle_level = 0
        elif vehicle_count < 15:
            vehicle_level = 1
        else:
            vehicle_level = 2

        ####################################################
        # Speed Level
        ####################################################

        if avg_speed < 2:
            speed_level = 0
        elif avg_speed < 5:
            speed_level = 1
        else:
            speed_level = 2

        ####################################################
        # Queue Length Level
        ####################################################

        if queue_length < 3:
            queue_level = 0
        elif queue_length < 8:
            queue_level = 1
        else:
            queue_level = 2

        ####################################################
        # Queue Time Level
        ####################################################

        if avg_queue_time < 10:
            wait_level = 0
        elif avg_queue_time < 30:
            wait_level = 1
        else:
            wait_level = 2

        ####################################################
        # Final RL State
        ####################################################

        return (
            vehicle_level,
            speed_level,
            queue_level,
            wait_level
        )
