import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorNetwork(nn.Module):
    """The actor network, which maps states to actions probabilities."""

    def __init__(self, num_obs, num_actions):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(num_obs, 400)
        self.n1 = nn.LayerNorm(400)
        self.l2 = nn.Linear(400, 300)
        self.n2 = nn.LayerNorm(300)
        self.l3 = nn.Linear(300, num_actions)

        # Initialize the weights
        self.initialize_weights()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.n1(self.l1(state))) # ReLU activation for the first layer
        x = F.relu(self.n2(self.l2(x)))     # ReLU activation for the second layer
        return torch.tanh(self.l3(x))       # tanh activation for the output layer to ensure action in [-1,1] range


    def initialize_weights(self):
        # Method to initialize the weights
        # Using DDPG paper suggestions
        s1 = self.l1.weight.data.size()[0]
        s2 = self.l2.weight.data.size()[0]

        self.l1.weight.data.uniform_(-1./np.sqrt(s1), 1./np.sqrt(s1))
        self.l1.bias.data.uniform_(-1./np.sqrt(s1), 1./np.sqrt(s1))

        self.l2.weight.data.uniform_(-1./np.sqrt(s2), 1./np.sqrt(s2))
        self.l2.bias.data.uniform_(-1./np.sqrt(s2), 1./np.sqrt(s2))

        self.l3.weight.data.uniform_(-0.003, 0.003)
        self.l3.bias.data.uniform_(-0.003, 0.003)



class CriticNetwork(nn.Module):
    """The critic network, which maps (state, action) pairs to Q-values."""
    
    def __init__(self, num_obs, num_actions):
        super(CriticNetwork, self).__init__()
 
        self.l1 = nn.Linear(num_obs, 400)
        self.n1 = nn.LayerNorm(400)         # Use of Normalization layers as recommended in the DDPG paper

        self.l2 = nn.Linear(400, 300)
        self.n2 = nn.LayerNorm(300)         # Use of Normalization layers as recommended in the DDPG paper

        self.action_layer = nn.Linear(num_actions, 300)

        self.l4 = nn.Linear(300, 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state, action):
        state_value = F.relu(self.n1(self.l1(state))) # ReLU activation for the first layer
        state_value = self.n2(self.l2(state_value))   # Layer normalization for the second layer

        action_value = self.action_layer(action)      # Action value is obtained separately

        # State and action values are combined here
        # In my experiments it worked better to apply ReLu after add instead of
        # applying ReLu to state_value and action_value separated and then adding
        state_action_value = F.relu(torch.add(state_value, action_value))

        # Return the Q-value estimate
        return self.l4(state_action_value)


    def initialize_weights(self):
        # Method to initialize the weights
        # Using DDPG paper suggestions
        s1 = self.l1.weight.data.size()[0]
        s2 = self.l2.weight.data.size()[0]
        s_action = self.action_layer.weight.data.size()[0]
        s4 = self.l4.weight.data.size()[0]

        self.l1.weight.data.uniform_(-1./np.sqrt(s1), 1./np.sqrt(s1))
        self.l1.bias.data.uniform_(-1./np.sqrt(s1), 1./np.sqrt(s1))

        self.l2.weight.data.uniform_(-1./np.sqrt(s2), 1./np.sqrt(s2))
        self.l2.bias.data.uniform_(-1./np.sqrt(s2), 1./np.sqrt(s2))

        self.action_layer.weight.data.uniform_(-1./np.sqrt(s_action), 1./np.sqrt(s_action))
        self.action_layer.bias.data.uniform_(-1./np.sqrt(s_action), 1./np.sqrt(s_action))

        self.l4.weight.data.uniform_(-0.003, 0.003)
        self.l4.bias.data.uniform_(-0.003, 0.003)


def copy_target(target, source):
    """Copies the parameters from the source network to the target network.
    Args:
     -target (nn.Module): The target network to copy the parameters to.
     -source (nn.Module): The source network to copy the parameters from.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target_model, local_model,tau):
      for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
          target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

