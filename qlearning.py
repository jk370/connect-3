import connect
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class Agent():
    def __init__(self, environment, alpha=0.1, epsilon=0.2):
        '''Initializes all required variables'''
        self.environment = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.rewards = [] 
    
    def random_move(self):
        '''Returns a random available move'''
        available_actions = self.environment.available_actions
        chosen_action = np.random.choice(available_actions)
        return chosen_action
    
class LearningAgent(Agent):
    def __init__(self, environment, alpha=0.1, epsilon=0.2, gamma=1):
        '''Initializes all required variables'''
        super(LearningAgent, self).__init__(environment, alpha, epsilon)
        self.gamma = gamma
        self.Q = {}
        self.opponent_rewards = []
    
    def choose_move(self):
        '''Returns the chosen move under the e-greedy policy'''
        # Generate Q-table key for state and symmetrical state
        state_t = np.copy(self.environment.grid)
        board_max_index = np.size(state_t, 1)-1
        symm_state_t = np.array_str(np.flip(state_t, 1))
        state_t = np.array_str(state_t)
        
        chance = np.random.uniform(0,1)
        if (chance < self.epsilon):
            # Generate state and symmetrical state key
            action_t = self.random_move()
            key = state_t + str(action_t)
            symm_key = symm_state_t + str(board_max_index-action_t)
            # Add state-action to Q-table if new
            if key not in self.Q:
                self.Q[key] = 0
            if symm_key not in self.Q:
                self.Q[symm_key] = 0
            return action_t
        
        else:
            available_actions = self.environment.available_actions
            # Discover available states
            for action in available_actions:
                key = state_t + str(action)
                symm_key = symm_state_t + str(board_max_index-action)
                if key not in self.Q:
                    self.Q[key] = 0
                if symm_key not in self.Q:
                    self.Q[symm_key] = 0
                    
            # Find max Q from state
            max_Q = max(self.Q[state_t + str(action)] for action in available_actions)
            
            # Choose random action that corresponds to max_Q
            max_actions = []
            for action in available_actions:
                if (self.Q[state_t + str(action)] == max_Q):
                    max_actions.append(action)
                    
            max_action = np.random.choice(max_actions)
            return max_action
    
    def learn(self, state_t, action_t, reward, state_t1):
        '''Updates Q-table accordingly to learn'''
        # Convert state-action representation to string for Q-table key
        board_max_index = np.size(state_t, 1)-1
        available_actions = self.environment.available_actions
        
        # Symmetrical variables
        symm_state_t = np.array_str(np.flip(state_t, 1))
        symm_state_t1 = np.array_str(np.flip(state_t1, 1))
        symm_action_t = board_max_index-action_t
        symm_update_key = symm_state_t + str(symm_action_t)
        
        # State variables
        state_t = np.array_str(state_t)
        state_t1 = np.array_str(state_t1)
        update_key = state_t + str(action_t)
        
        
        # Discover new state-actions for state and symmetrical state
        for action in available_actions:
            # Generate keys
            key = state_t1 + str(action)
            symm_key = symm_state_t1 + str(board_max_index-action)
            if key not in self.Q:
                self.Q[key] = 0
            if symm_key not in self.Q:
                self.Q[symm_key] = 0
                
        # Check if terminal state
        if self.environment.was_winning_move() or self.environment.grid_is_full():
            factor = 0
        else:
            # Same value for both state and symmetrical state
            factor = self.gamma * max([self.Q[state_t1 + str(action)] for action in available_actions])
        
        # Amend Q values for both states
        self.Q[update_key] += (self.alpha * (reward + factor - self.Q[update_key]))
        self.Q[symm_update_key] += (self.alpha * (reward + factor - self.Q[symm_update_key]))

class RandomAgent(Agent):
    def __init__(self, environment):
        '''Initializes all required variables'''
        super(RandomAgent, self).__init__(environment)
    
    def choose_move(self):
        '''Returns a random available move'''
        return self.random_move()
    
def test(agent, episodes=10):
    '''Performs policy evaluation over given number of episodes'''
    # Save old values to allow continuation
    old_epsilon = agent.epsilon
    old_environment = agent.environment
    # Set new values for policy evaluations
    new_environment = connect.Connect(verbose=False)
    agent.environment = new_environment
    agent.epsilon = 0 # Greedy action
    opponent = RandomAgent(environment=new_environment)
    total_reward = 0
    
    # Play episodes
    for _ in range(episodes):
        # Reset environment
        new_environment.reset(first_player='o')
        
        # Opponent takes first turn
        action_t = opponent.choose_move()
        new_environment.act(action = action_t)
        
        # Play episode until win or board is full
        while not (new_environment.was_winning_move() or new_environment.grid_is_full()):
            # Take action from state
            new_environment.change_turn()
            action_t = agent.choose_move()
            new_environment.act(action = action_t)
            
            # Observe response
            if new_environment.was_winning_move():
                reward = 1
            elif new_environment.grid_is_full():
                reward = 0
            else:
                response = opponent.choose_move()
                new_environment.change_turn()
                new_environment.act(action = response)
                if new_environment.was_winning_move():
                    reward = -1
                else:
                    reward = 0
        
        total_reward += reward
        
    # Save rewards and reset agent to before interruption and return
    agent.rewards.append(total_reward)
    agent.epsilon = old_epsilon
    agent.environment = old_environment
    return agent

def play(max_steps=30000, ver=False, n=1000):
    '''Allows agent to learn through interaction - policy improvement'''
    # Setup players and environment
    steps = 0
    env = connect.Connect(verbose=ver)
    opponent = RandomAgent(environment=env)
    agent = LearningAgent(environment=env)
    
    # Play all steps
    while steps <= max_steps:
        # Reset environment
        env.reset(first_player='o')
        
        # Opponent takes first turn
        action_t = opponent.choose_move()
        env.act(action = action_t)
        
        # Play episode until win or board is full - end at max steps
        while not (env.was_winning_move() or env.grid_is_full()) and steps <= max_steps:
            # Take action from state
            env.change_turn()
            state_t = np.copy(env.grid)
            action_t = agent.choose_move()
            env.act(action = action_t)
            
            # Observe response
            if env.was_winning_move():
                reward = 1
            elif env.grid_is_full():
                reward = 0
            else:
                response = opponent.choose_move()
                env.change_turn()
                env.act(action = response)
                if env.was_winning_move():
                    reward = -1
                else:
                    reward = 0
            
            state_t1 = np.copy(env.grid)
            # Update Q tables
            agent.learn(state_t, action_t, reward, state_t1)
                
            # Keep track of steps taken
            if (steps % n) == 0:
                # Play 10 episodes of policy evaluation
                agent = test(agent, episodes=10)
                opponent = test(opponent, episodes = 10)
            steps += 1
    
    # Save rewards for random agent and return single agent
    agent.opponent_rewards = opponent.rewards
    return agent
	