import numpy as np
import os
import gym
from tqdm import tqdm
from PIL import Image, ImageDraw
total_reward = []

class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.8, gamma=0.9):
        """
        Parameters:
            env: target enviornment.
            epsilon: Determinds the explore/expliot rate of the agent.
            learning_rate: Learning rate of the agent.
            gamma: discount rate of the agent.
        """
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize qtable
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))

        self.qvalue_rec = []

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.

        Parameters:
            state: A representation of the current state of the enviornment.
            epsilon: Determines the explore/expliot rate of the agent.

        Returns:
            action: The action to be evaluated.
        """
        """
        If the random number is less than epsilon, the agent will choose a random action.
        Otherwise, the agent will choose the action with the highest estimated value from
        its Q-table for the current state. This is the exploitation step.
        """
        if np.random.uniform() < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.qtable[state])
        return action

    def learn(self, state, action, reward, next_state, done, ep):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.

        Parameters:
            state: The state of the enviornment before taking the action.
            action: The exacuted action.
            reward: Obtained from the enviornment after taking the action.
            next_state: The state of the enviornment after taking the action.
            done: A boolean indicates whether the episode is done.

        Returns:
            None (Don't need to return anything)
        """
        """
        'next_max' calculates the maximum expected reward for any action that can be taken
        in the next state.
        'updated' is the estimate of the expected reward for taking action in state,
        calculated using the Q-learning update rule.
        """
        original = self.qtable[state][action]
        next_max = self.check_max_Q(next_state)
        updated = (1 - self.learning_rate) * original + \
                    self.learning_rate * (reward + self.gamma * next_max)
        self.qtable[state][action] = updated
        if done:
            np.save("./Tables/taxi_table.npy", self.qtable)

    def check_max_Q(self, state):
        """
        - Implement the function calculating the max Q value of given state.
        - Check the max Q value of initial state

        Parameter:
            state: the state to be check.
        Return:
            max_q: the max Q value of given state
        """
        """
        Returns the maximum expected reward for any action that can be taken in 
        the current state, as estimated by the Q-table.
        """
        max_q = max(self.qtable[state])
        return max_q        

def train(env):
    """
    Train the agent on the given environment.

    Paramenter:
        env: the given environment.

    Return:
        None
    """
    training_agent = Agent(env)
    episode = 3000
    rewards = []
    for ep in tqdm(range(episode)):
        state = env.reset()
        done = False

        count = 0
        while True:
            action = training_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            training_agent.learn(state, action, reward, next_state, done, ep)
            count += reward

            if done:
                rewards.append(count)
                break

            state = next_state

    total_reward.append(rewards)

def four_bits(i):
    if len(str(i)) == 1:
        return "000" + str(i)
    elif len(str(i)) == 2:
        return "00"+ str(i)
    elif(len(str(i))) == 3:
        return "0" + str(i)
    else:
        return str(i)

def test(env):
    """
    Test the agent on the given environment.

    Paramenters:
        env: the given environment.

    Return:
        None
    """
    testing_agent = Agent(env)
    testing_agent.qtable = np.load("./Tables/taxi_table.npy")
    errs = []
    for m in range(100):
        state = testing_agent.env.reset()
        count = 0
        points = []
        prev_state = -1
        times = 0
        step = 3
        oscillate = 0
        dist = 0
        while True:
            action = np.argmax(testing_agent.qtable[state])
            
            for i in range(step):
                next_state, reward, done, _ = testing_agent.env.step(action)
                
            taxi_row, taxi_col, taxi_height = env.decode(env.s)
            points.append((taxi_row, taxi_col, taxi_height, step))
            count += reward
            
            if prev_state == next_state:
                times += 1
                if times == 4:
                    oscillate = 1
            else:
                times = 0
            
            if oscillate:
                step -= 1
                times = 0
                oscillate = 0
                if step == 0:
                    break
            
            prev_state = state
            state = next_state

        num_frames = len(points)
        frame_duration = 1
        frames = []
        ideal = abs(points[0][0] - env.locs[0][0]) + abs(points[0][1] - env.locs[0][1]) + abs(points[0][2] - env.locs[0][2])
        for k in range(num_frames):
            image = Image.open(f"dataset/image.0001.{four_bits(points[k][2]+17)}.png")
            draw = ImageDraw.Draw(image, "RGBA")
            
            point_size = 1
            
            radius = points[k][3] * 10
            x = points[k][0]
            y = points[k][1]
            x_range = range(x - point_size, x + point_size + 1)
            y_range = range(y - point_size, y + point_size + 1)
            for i in x_range:
                for j in y_range:
                    draw.point((j, i), fill="blue")
                    draw.rectangle((y-radius, x-radius, y+radius, x+radius), outline="yellow")
            
            x = env.locs[0][0]
            y = env.locs[0][1]
            radius = abs(env.locs[0][2] - points[k][2])
            dist = ((points[k][0] - env.locs[0][0])**2 + (points[k][1] - env.locs[0][1])**2 + (points[k][2] - env.locs[0][2])**2)**(1/3)
            if k:
                prev_dist = ((points[k-1][0] - env.locs[0][0])**2 + (points[k-1][1] - env.locs[0][1])**2 + (points[k-1][2] - env.locs[0][2])**2)**(1/3)
            else:
                prev_dist = 1e9
            x_range = range(x - point_size, x + point_size + 1)
            y_range = range(y - point_size, y + point_size + 1)
            for i in x_range:
                for j in y_range:
                    draw.ellipse((j-1, i-1, j+1, i+1), outline=None, fill=(255, 0, 0, 255))
                    draw.ellipse((y-radius, x-radius, y+radius, x+radius), outline=None, fill=(255, 0, 0, 5))
                    draw.text((5, 5), f'Spacing {points[k][3]}', 'yellow')
                    
                    if dist <= prev_dist:
                        color = 'green'
                    else:
                        color = 'red'
                    draw.text((10, 195), f'Error : {dist}mm', color)
            
            frames.append(image)
        frames[0].save(f"animations/animation{m}.gif", format="GIF", append_images=frames[1:], save_all=True, duration=frame_duration)
        if (num_frames-1 - ideal) / ideal >= 0:
            errs.append((num_frames-1 - ideal) / ideal)
    print(errs)
    print(1 - np.mean(errs))
    print(np.std(errs))


if __name__ == "__main__":

    env = gym.make("Taxi-v3")
    os.makedirs("./Tables", exist_ok=True)

    # training section:
    print(f"training progress")
    train(env)   
    # testing section:
    test(env)
    os.makedirs("./Rewards", exist_ok=True)
    os.makedirs("./animations", exist_ok=True)

    np.save("./Rewards/taxi_rewards.npy", np.array(total_reward))
    
    env.close()
