import numpy as np
import matplotlib.pyplot as plt
import math
import time


def rolling_window(a, window, step_size):
    """Create a rolling window view of a numpy array."""

    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


ax = None
fig = None

def episode_reward_plot(rewards, frame_idx, window_size=5, step_size=1, updating=False):
    """Plot episode rewards rolling window mean, min-max range and standard deviation.

    Parameters
    ----------
    rewards : list
        List of episode rewards.
    frame_idx : int
        Current frame index.
    window_size : int
        Rolling window size.
    step_size: int
        Step size between windows.
    updating: bool
        You can try to set updating to True, which hinders matplotlib to create a new window for every plot.
        Doesn't work with my Pycharm SciView currently.
    """
    global ax
    global fig

    #plt.ion()
    rewards_rolling = rolling_window(np.array(rewards), window_size, step_size)
    mean = np.mean(rewards_rolling, axis=1)
    std = np.std(rewards_rolling, axis=1)
    min = np.min(rewards_rolling, axis=1)
    max = np.max(rewards_rolling, axis=1)
    x = np.arange(math.floor(window_size / 2), len(rewards) - math.floor(window_size / 2), step_size)

    if ax is None: #or not updating:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, facecolor='blue')
    ax.fill_between(x, min, max, alpha=0.1, facecolor='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    if updating:
        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(0.01)
        # plt.ion()
        # plt.show()
        # plt.clf()
    else:
        plt.show(block=False)
        plt.savefig('rewards.png')




def visualize_agent(env, agent, timesteps=500):
    """ Visualize an agent performing inside a Gym environment. """
    obs, _ = env.reset()
    for timestep in range(1, timesteps + 1):
        env.render()
        action = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        # 30 FPS
        time.sleep(0.033)

def video_agent(env, agent, n_episodes=500):
    """ Visualize an agent performing inside a Gym environment. """
    ce = 0
    obs, _ = env.reset()
    while True:
        env.render()
        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            ce=ce+1
            if ce>=n_episodes:
                break
    env.close()

