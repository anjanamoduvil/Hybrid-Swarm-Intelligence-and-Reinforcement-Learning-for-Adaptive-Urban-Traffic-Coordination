class RewardFunction:
    """
    Calculates the reward for the RL agent based on
    current traffic conditions.
    """

    def calculate_reward(self, traffic_stats):

        reward = 0

        ####################################################
        # Reward high average speed
        ####################################################

        reward += traffic_stats["average_speed"] * 2

        ####################################################
        # Penalize queue length
        ####################################################

        reward -= traffic_stats["queue_length"] * 5

        ####################################################
        # Penalize waiting time
        ####################################################

        reward -= traffic_stats["average_queue_time"] * 0.5

        ####################################################
        # Reward traffic throughput
        ####################################################

        reward += traffic_stats["vehicle_count"]

        ####################################################
        # Penalize congestion
        ####################################################

        if traffic_stats["congestion"] == "Heavy":
            reward -= 50

        elif traffic_stats["congestion"] == "Moderate":
            reward -= 20

        return round(reward, 2)
