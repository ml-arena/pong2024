import numpy as np


def extract_positions(observation):
    y_bottom = 34
    y_top = 194

    o = observation[y_bottom:y_top, :, 0]
    y_mask_left = np.int_(o[:, 19] == 213)
    y_mask_right = np.int_(o[:, 140] == 92)
    ball_mask = np.int_(o == 236)
    assert np.any(y_mask_left)
    assert np.any(y_mask_right)

    y_range = np.linspace(0,1, y_top - y_bottom)
    x_range = np.linspace(0,1, o.shape[1])

    y_pos_left = (y_range * y_mask_left).sum() / max(1, np.sum(y_mask_left))
    y_pos_right = (y_range * y_mask_right).sum() / max(1, np.sum(y_mask_right))

    x_ball_mask = ball_mask.max(0)
    y_ball_mask = ball_mask.max(1)
    x_ball = (x_range * x_ball_mask).sum() / max(1, x_ball_mask.sum())
    y_ball = (y_range * y_ball_mask).sum() / max(1, y_ball_mask.sum())
    is_ball_present = np.any(ball_mask)

    return y_pos_left, y_pos_right, x_ball, y_ball, is_ball_present

class Agent:

    def __init__(self, env, player_name=None):
        self.player_name = player_name

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None, action_mask=None):

        y_pos_left, y_pos_right, x_ball, y_ball, is_ball_present = extract_positions(observation)
        if not is_ball_present:
            return np.random.randint(6)

        y_pos = y_pos_left if self.player_name == "second_0" else y_pos_right

        if y_pos > y_ball:
            return 2 # DOWN

        return 3