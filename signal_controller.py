class RLSignalController:

    def __init__(self):

        self.current_signal = "North"

        self.green_time = 30

    def apply_action(self, action):

        if action == 0:
            self.current_signal = "North"

        elif action == 1:
            self.current_signal = "South"

        elif action == 2:
            self.current_signal = "East"

        elif action == 3:
            self.current_signal = "West"

        elif action == 4:
            self.green_time += 5

        elif action == 5:

            self.green_time = max(
                10,
                self.green_time - 5
            )

        return self.current_signal, self.green_time
