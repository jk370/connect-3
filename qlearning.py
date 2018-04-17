### Write all your code for Part 1 within or above this cell. 
import connect
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class Agent():
    def __init__(self, environment, alpha = 0.1, epsilon=0.2):
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
        # Generate state string for Q-table key
        state_t = np.copy(self.environment.grid)
        state_t = ''.join(state_t.flatten())
        
        chance = np.random.uniform(0,1)
        if (chance < self.epsilon):
            action_t = self.random_move()
            key = state_t + str(action_t)
            # Add state-action to Q-table if new
            if key not in self.Q:
                self.Q[key] = 0 
            return action_t
        
        else:
            available_actions = self.environment.available_actions
            # Discover available states
            for action in available_actions:
                key = state_t + str(action)
                if key not in self.Q:
                    self.Q[key] = 0
                    
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
        # Convert state, action representation to string for Q-table key
        available_actions = self.environment.available_actions
        state_t = ''.join(state_t.flatten())
        state_t1 = ''.join(state_t1.flatten())
        update_key = state_t + str(action_t)
        
        # Discover new state-actions
        for action in available_actions:
            key = state_t1 + str(action)
            if key not in self.Q:
                self.Q[key] = 0
                
        # Check if terminal state
        if self.environment.was_winning_move() or self.environment.grid_is_full():
            factor = 0
        else:
            factor = self.gamma * max([self.Q[state_t1 + str(action)] for action in available_actions])
        
        # Amend Q values
        self.Q[update_key] += (self.alpha * (reward + factor - self.Q[update_key]))

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
    new_environment = connect.Connect(verbose=False)
    agent.environment = new_environment
    agent.epsilon = 0
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

def play(max_steps = 25000, ver = False, n=1000):
    '''Allows agent to learn through interaction - policy improvement'''
    # Setup players and environment
    steps = 0
    env = connect.Connect(verbose=ver)
    opponent = RandomAgent(environment=env)
    agent = LearningAgent(environment=env)
    
    # Play all steps
    for _ in range(max_steps):
        # Reset environment
        env.reset(first_player='o')
        
        # Opponent takes first turn
        action_t = opponent.choose_move()
        env.act(action = action_t)
        
        # Play episode until win or board is full
        while not (env.was_winning_move() or env.grid_is_full()) and steps < max_steps:
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
                # Play 10 games with no learning and collect total reward gained versus baseline
                agent = test(agent, episodes=10)
                opponent = test(opponent, episodes = 10)
            steps += 1
    
    agent.opponent_rewards = opponent.rewards
    return agent

def plot_learning(agents = 5, step_number=20000, interrupt=500):
    '''Plots learning graph for episodes played'''
    plt.figure(1)
    agent_rewards = []
    random_rewards = []
    
    # Collect returns of all agents playing episodes
    for _ in range(agents):
        q_agent = play(max_steps = step_number, ver = False, n=interrupt)
        agent_rewards.append(q_agent.rewards)
        random_rewards.append(q_agent.opponent_rewards)
    
    # Average agents across all observations
    agent_rewards = np.mean(agent_rewards, axis=0, dtype=np.float64)
    random_rewards = np.mean(random_rewards, axis=0, dtype=np.float64)
    
    # Create title
    if agents == 1:
        title = "Q-learning for 1 agent compared to random baseline"
    elif agents > 1:
        agents_str = str(agents)
        title = "Q-Learning for " + agents_str + " agents compared to random baseline"
    else:
        title = "Too few agents given"
    
    # Plot figure
    plt.figure(1)
    plt.plot(agent_rewards)
    plt.plot(random_rewards)
    plt.title(title)
    plt.xlabel("Number of steps performed (n=" + str(interrupt) + ")")
    plt.ylabel("Average total reward after 10 episodes")
    plt.legend(["Q-Agent", "Random Agent"], loc = "best")

#plot_learning(agents=50, step_number=90000, interrupt=3000)