import unittest

from rl_agent import QLearningAgent


class TestQLearningAgent(unittest.TestCase):

    def setUp(self):

        self.agent = QLearningAgent()

    def test_initialization(self):

        self.assertEqual(self.agent.actions, 6)

    def test_choose_action(self):

        state = (0, 0, 0, 0)

        action = self.agent.choose_action(state)

        self.assertTrue(0 <= action < 6)

    def test_q_update(self):

        state = (0, 1, 1, 0)
        next_state = (1, 1, 1, 0)

        self.agent.update(
            state,
            2,
            10,
            next_state
        )

        q_values = self.agent.get_q_values(state)

        self.assertNotEqual(q_values[2], 0)

    def test_save_load(self):

        self.agent.save("test_q.pkl")

        self.agent.load("test_q.pkl")

        self.assertIsInstance(
            self.agent.q_table,
            dict
        )


if __name__ == "__main__":
    unittest.main()
