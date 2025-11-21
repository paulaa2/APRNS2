import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from replaybuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import itertools

#funcionaaaaa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TD3:

    def __init__(self, env, replay_size=1000000, batch_size=32, gamma=0.99, policy_delay=2, policy_noise=0.2, noise_clip=0.5):

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_delay = policy_delay  
        self.policy_noise = policy_noise  
        self.noise_clip = noise_clip      
        self.total_iterations = 0        

        self.Critic1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic1.initialize_weights()
        self.Critic1_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic1_target, self.Critic1)
        
        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2.initialize_weights()
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic2_target, self.Critic2)

        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor_target, self.Actor)

        q_params = itertools.chain(self.Critic1.parameters(), self.Critic2.parameters())
        self.optim_critic = optim.Adam(q_params, lr=0.001) 
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.0001)   


    def learn(self, timesteps):
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []

        GaussianNoise = NormalActionNoise(np.zeros(self.act_dim), sigma=0.1 * np.ones(self.act_dim))

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            epsilon = GaussianNoise.sample()
            epsilon = np.clip(epsilon, -self.noise_clip, self.noise_clip)
            action = np.clip(action + epsilon, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                batch = self.replay_buffer.get(self.batch_size)
                self.optim_critic.zero_grad()
                critic_loss = self.compute_critic_loss(batch)
                critic_loss.backward()
                self.optim_critic.step()

                self.total_iterations += 1
                if self.total_iterations % self.policy_delay == 0:
                    # Update actor network
                    self.optim_actor.zero_grad()
                    actor_loss = self.compute_actor_loss(batch)
                    actor_loss.backward()
                    self.optim_actor.step()

                    # Soft update target networks (only when actor is updated)
                    with torch.no_grad():
                        soft_update(self.Critic1_target, self.Critic1, tau=0.005)
                        soft_update(self.Critic2_target, self.Critic2, tau=0.005)
                        soft_update(self.Actor_target, self.Actor, tau=0.005)

            if timestep % (timesteps-1) == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                pass
            if len(all_rewards_eval)>10 and np.mean(all_rewards_eval[-5:]) > 220:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                break
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        a = self.Actor(torch.FloatTensor(s).to(device)).cpu().detach().numpy()
        return a


    def compute_critic_loss(self, batch):
        
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(device) 
        action_batch = torch.Tensor(action_batch).to(device) 
        next_state_batch = torch.FloatTensor(next_state_batch).to(device) 
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1) 
        terminated_batch = torch.FloatTensor(terminated_batch).to(device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(truncated_batch).to(device).unsqueeze(1)

        q1_expected = self.Critic1(state_batch, action_batch) 
        q2_expected = self.Critic2(state_batch, action_batch)

        with torch.no_grad():
            next_action = self.Actor_target(next_state_batch)
            # Target policy smoothing: add clipped Gaussian noise
            target_noise = NormalActionNoise(np.zeros(self.act_dim), sigma=self.policy_noise * np.ones(self.act_dim))
            noise = np.array([target_noise.sample() for _ in range(len(next_state_batch))])
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            noise = torch.FloatTensor(noise).to(device)
            next_action = torch.clamp(next_action + noise, -1, 1)
            
            q1_targets_next = self.Critic1_target(next_state_batch, next_action)
            q2_targets_next = self.Critic2_target(next_state_batch, next_action)
            q_targets_next = torch.min(q1_targets_next, q2_targets_next)
            
            target = reward_batch + (1-(terminated_batch)) * (1-(truncated_batch)) * self.gamma * q_targets_next

        criterion = nn.MSELoss()
        loss1 = criterion(q1_expected, target)
        loss2 = criterion(q2_expected, target)
        loss = loss1 + loss2

        return loss
    

    def compute_actor_loss(self, batch):
        
        state_batch, _, _, _, _, _ = batch

        state_batch = torch.FloatTensor(state_batch).to(device)
        
        action_batch = self.Actor(state_batch)
        loss = -torch.mean(self.Critic1(state_batch, action_batch))

        return loss


    def eval_episodes(self, n=3):
        lr = []
        for episode in range(n):
            tr = 0.0
            obs, _ = self.env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)


if __name__ == '__main__':
    env = gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array')

    td3 = TD3(env, replay_size=1000000, batch_size=64, gamma=0.99, policy_delay=2)

    td3.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array'), 'video_td3')    
    video_agent(env, td3, n_episodes=5)  
    pass

