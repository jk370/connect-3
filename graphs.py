import qlearning
import minimax
import expectiminimax

def plot_learning(agents=20, step_number=30000, interrupt=1000):
    '''Plots learning graph for episodes played'''
    agent_rewards = []
    random_rewards = []
    
    # Collect returns of all agents playing episodes
    for _ in range(agents):
        q_agent = play(max_steps=step_number, ver=False, n=interrupt)
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
    
    return (agent_rewards, random_rewards)

def compare_learning(qagent_rewards, random_rewards, agents = 50, step_number=50000, interrupt=1000):
    '''Plots learning graph for episodes played'''
    magent_rewards = []
    
    # Collect returns of all agents playing episodes
    for _ in range(agents):   
        # Setup Minimax agent and environment
        test_env = connect.Connect(verbose=False)
        m_agent = MinmaxAgent(test_env)
        
        # Play required episodes for Minimax agent
        for _ in range(int(step_number/interrupt)):
            test_env.reset(first_player='o')
            test(m_agent)
            
        # Append reward for each agent
        magent_rewards.append(m_agent.rewards)
    
    # Average agent across all observations
    magent_rewards = np.mean(magent_rewards, axis=0, dtype=np.float64)
    
    # Create title
    if agents == 1:
        title = "Comparison of Q-Learning and Minimax Algorithms against a random baseline for 1 agent"
    elif agents > 1:
        agents_str = str(agents)
        title = "Comparison of Q-Learning and Minimax Algorithms against a random baseline for " + agents_str + " agents"
    else:
        title = "Too few agents given"
    
    # Plot figure for all agents
    plt.figure(1)
    plt.plot(qagent_rewards)
    plt.plot(magent_rewards)
    plt.plot(random_rewards)
    plt.title(title)
    plt.xlabel("Number of steps performed (n=" + str(interrupt) + ")")
    plt.ylabel("Average total reward after 10 episodes")
    plt.legend(["Q-Agent", "Minimax Agent", "Random Agent"], loc = "best")
    
    return magent_rewards

def compare_all_agents(qagent_rewards, magent_rewards, random_rewards, agents = 50, step_number=50000, interrupt=1000):
    '''Plots learning graph for episodes played'''
    smagent_rewards = []
    
    # Collect returns of all agents playing episodes
    for _ in range(agents):
        # Setup expectiminimax agent and environment
        test_env = connect.Connect(verbose=False)
        sm_agent = StochasticMinmaxAgent(test_env)
        
        # Play required episodes for agent
        for _ in range(int(step_number/interrupt)):
            test_env.reset(first_player='o')
            test(sm_agent)
            
        # Append reward for each agent
        smagent_rewards.append(sm_agent.rewards)
    
    # Average agents across all observations
    smagent_rewards = np.mean(smagent_rewards, axis=0, dtype=np.float64)
    
    # Create title
    if agents == 1:
        title = "Comparison of Q-Learning and Minimax Algorithms against a random baseline for 1 agent"
    elif agents > 1:
        agents_str = str(agents)
        title = "Comparison of Q-Learning and Minimax Algorithms against a random baseline for " + agents_str + " agents"
    else:
        title = "Too few agents given"
    
    # Plot figure
    plt.figure(1)
    plt.plot(qagent_rewards)
    plt.plot(magent_rewards)
    plt.plot(smagent_rewards)
    plt.plot(random_rewards)
    plt.title(title)
    plt.xlabel("Number of steps performed (n=" + str(interrupt) + ")")
    plt.ylabel("Average total reward after 10 episodes")
    plt.legend(["Q-Agent", "Minimax Agent", "Expectiminimax Agent", "Random Agent"], loc = "best")
    
    return smagent_rewards
	
#agent_rewards, random_rewards = plot_learning(agents=50, step_number=50000, interrupt=1000)
#solo_agent, solo_random = plot_learning(agents=1, step_number=35000, interrupt=1000)
#minimax_rewards = compare_learning(agent_rewards, random_rewards, agents=50, step_number=50000, interrupt=1000)
#solo_minimax = compare_learning(solo_agent, solo_random, agents=1, step_number=35000, interrupt=1000)
#expectiminimax_rewards = compare_all_agents(agent_rewards, minimax_rewards, random_rewards, agents=50, step_number=50000, interrupt=1000)
#solo_expectimax = compare_all_agents(solo_agent, solo_minimax, solo_random, agents=1, step_number=35000, interrupt=1000)