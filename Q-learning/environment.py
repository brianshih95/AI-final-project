from gym.envs.toy_text import discrete
import numpy as np

ROW = 218
COL = 182
HEIGHT = 120

MAP = [ "" for i in range(ROW+2)]
MAP[0] += '+'
for _ in range(COL*2-1):
    MAP[0] += '-'
MAP[0] += '+'
for i in range(1, ROW+1):
    MAP[i] += '|'
    for j in range(COL-1):
        MAP[i] += ' '
    MAP[i] += " |"
MAP[ROW+1] += '+'
for _ in range(COL*2-1):
    MAP[ROW+1] += '-'
MAP[ROW+1] += '+'

MAP3d = []
for i in range(HEIGHT):
    MAP3d.append(MAP)

WINDOW_SIZE = (550, 350)

class TaxiEnv(discrete.DiscreteEnv):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self):
        self.locs = locs = [(60, 60, 60)]
        self.step_size = 1
        num_rows = ROW
        num_columns = COL
        num_heights = HEIGHT
        num_states = num_rows * num_columns * num_heights
        max_row = num_rows - 1
        max_col = num_columns - 1
        max_height = num_heights - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for height in range(num_heights):
            for row in range(num_rows):
                for col in range(num_columns):
                    state = self.encode(row, col, height)
                    if row != locs[0][0] or col != locs[0][1] or height != locs[0][2]:
                        initial_state_distrib[state] += 1
                    for action in range(num_actions):
                        new_row, new_col, new_height = row, col, height
                        done = False
                        taxi_loc = (row, col, height)
                        outBound = 0
                        if action == 0:
                            new_row = min(row + self.step_size, max_row)
                            if row + self.step_size > max_row:
                                outBound = 1
                        elif action == 1:
                            new_row = max(row - self.step_size, 0)
                            if row - self.step_size < 0:
                                outBound = 1
                        elif action == 4:
                            new_col = min(col + self.step_size, max_col)
                            if col + self.step_size > max_col:
                                outBound = 1
                        elif action == 5:
                            new_col = max(col - self.step_size, 0)
                            if col - self.step_size < 0:
                                outBound = 1
                        elif action == 2:
                            new_height = min(height + self.step_size, max_height)
                            if height + self.step_size > max_height:
                                outBound = 1
                        elif action == 3:
                            new_height = max(height - self.step_size, 0)
                            if height - self.step_size < 0:
                                outBound = 1
                        if outBound:
                            reward = -100000
                        else:
                            reward = (((taxi_loc[0] - locs[0][0])**2 + (taxi_loc[1] - locs[0][1])**2 + (taxi_loc[2] - locs[0][2])**2)**(1/3) - ((locs[0][0] - new_row)**2 + (locs[0][1] - new_col)**2 + (locs[0][2] - new_height)**2)**(1/3))*1000
                        if(new_row == locs[0][0] and new_col == locs[0][1] and new_height == locs[0][2]):
                            done = True
                        new_state = self.encode(
                            new_row, new_col, new_height
                        )
                        P[state][action].append(
                            (1.0, new_state, reward, done)
                        )
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, taxi_height):
        col = COL
        height = HEIGHT
        i = taxi_row
        i *= col
        i += taxi_col
        i *= height
        i += taxi_height
        return i

    def decode(self, i):
        out = []
        col = COL
        height = HEIGHT
        out.append(i % height)
        i = i // height
        out.append(i % col)
        i = i // col
        out.append(i)
        return reversed(out)
