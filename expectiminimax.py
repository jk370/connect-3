### Write all your code for Part 3 in or above this cell.
from copy import deepcopy
import qlearning
import minimax

class StochasticMinmaxAgent(Agent):
    def __init__(self, environment):
        '''Initializes all required variables'''
        super(StochasticMinmaxAgent, self).__init__(environment)
        self.policy = {}
        
    def compute_policy(self, state):
        '''Searches tree from given state by calling expectiminimax'''
        origin = deepcopy(state)
        # Assume opponent has just taken turn
        if origin.player_at_turn == 'x':
            origin.change_turn()
            
        value = self.expectiminimax(origin, True)
        return value
        
    def expectiminimax(self, node, player):
        '''Recursive function to evaluate board with stochastic opponent'''
        # Add node to policy (or return if already found)
        key = np.copy(node.grid)
        key = ''.join(key.flatten())
        if key not in self.policy:
            self.policy[key] = 0
        else:
            return self.policy[key]
        
        # Check if node is terminal and reward last player
        if node.was_winning_move():
            if player:
                self.policy[key] = -1
                return -1
            else:
                self.policy[key] = 1
                return 1
        elif node.grid_is_full():
            # Game drawn
            return 0
        
        node.change_turn()
        # Maximizing player
        if player:
            bestValue = -1
            for a in node.available_actions:
                # Create child node and corresponding key
                child = deepcopy(node)
                child.act(action = a)
                new_key = np.copy(child.grid)
                new_key = ''.join(new_key.flatten())
                
                # Check if key already found
                if new_key not in self.policy:
                    self.policy[new_key] = self.expectiminimax(child, False)
                    
                bestValue = max(bestValue, self.policy[new_key])
        
        # Minimizing player
        else:
            bestValue = 0
            for a in node.available_actions:
                # Create child node and corresponding key
                child = deepcopy(node)
                child.act(action = a)
                new_key = np.copy(child.grid)
                new_key = ''.join(new_key.flatten())
                
                # Check if key already found
                if new_key not in self.policy:
                    self.policy[new_key] = self.expectiminimax(child, True)
                    
                bestValue += self.policy[new_key]
            
            # Average child nodes (due to equal weightings)
            bestValue = bestValue / len(node.available_actions)
        
        self.policy[key] = bestValue
        return bestValue
    
    def choose_move(self):
        '''Chooses best action for current state assuming optimal opponent'''
        # Create key from current state
        origin = deepcopy(self.environment)
        key = np.copy(origin.grid)
        key = ''.join(key.flatten())
        
        # Ensure turn always called from previous player
        if origin.player_at_turn == 'x':
            origin.change_turn()
        
        # Evaluate if new state
        if key not in self.policy:
            self.policy[key] = self.compute_policy(origin)
        
        # Find best action from available actions
        action_value = -2
        best_action = -1
        origin.change_turn()
            
        for a in origin.available_actions:
            # Create child node and corresponding key
            child = deepcopy(origin)
            child.act(action = a)
            new_key = np.copy(child.grid)
            new_key = ''.join(new_key.flatten())
            
            # Check key against current best action
            if self.policy[new_key] > action_value:
                action_value = self.policy[new_key]
                best_action = a
                
                # Take winning move if found
                if action_value == 1:
                    break
                
        return best_action