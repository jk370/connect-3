import qlearning
import minimax
import expectiminimax

def compare_all_agents(agents = 5, step_number=20000, interrupt=500):
    '''Plots learning graph for episodes played'''
    plt.figure(1)
    qagent_rewards = []
    magent_rewards = []
    smagent_rewards = []
    random_rewards = []
    
    # Collect returns of all agents playing episodes
    for _ in range(agents):
        # Collect rewards for Q-Agent and Random agent
        q_agent = play(max_steps = step_number, ver = False, n=interrupt)
        qagent_rewards.append(q_agent.rewards)
        random_rewards.append(q_agent.opponent_rewards)
        
        # Setup both Minimax agents and environment
        test_env = connect.Connect(verbose=False)
        m_agent = MinmaxAgent(test_env)
        sm_agent = StochasticMinmaxAgent(test_env)
        
        # Play required episodes for both Minimax agents
        for _ in range(int(step_number/interrupt)):
            test_env.reset(first_player='o')
            test(m_agent)
            test(sm_agent)
            
        # Append reward for each agent
        magent_rewards.append(m_agent.rewards)
        smagent_rewards.append(sm_agent.rewards)
    
    # Average agents across all observations
    qagent_rewards = np.mean(qagent_rewards, axis=0, dtype=np.float64)
    magent_rewards = np.mean(magent_rewards, axis=0, dtype=np.float64)
    smagent_rewards = np.mean(smagent_rewards, axis=0, dtype=np.float64)
    random_rewards = np.mean(random_rewards, axis=0, dtype=np.float64)
    
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