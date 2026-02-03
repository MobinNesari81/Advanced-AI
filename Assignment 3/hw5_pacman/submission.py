from util import manhattan_distance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    
    Reflex agents are simple agents that make decisions based on the immediate
    consequences of actions without looking ahead multiple steps. They evaluate
    the direct successor states and choose the action leading to the best
    immediate outcome.
    
    Attributes:
        last_positions (list): Tracks previous positions (for internal use).
        dc (None): Internal state variable.
    
    Methods:
        get_action(game_state): Returns the best action according to evaluation.
        evaluation_function(current_game_state, action): Evaluates an action's value.
    
    Note:
        The code below is provided as a guide. You are welcome to change
        it in any way you see fit, so long as you don't touch our method
        headers.
    """

    def __init__(self):
        self.last_positions = []
        self.dc = None

    def get_action(self, game_state: GameState):
        """
        get_action chooses among the best options according to the evaluation function.
        
        This method evaluates all legal moves and selects the one with the highest
        evaluation score. If multiple actions tie for the best score, one is chosen
        randomly among them.

        Args:
            game_state (GameState): The current state of the game.
        
        Returns:
            str: The selected action (one of Directions.NORTH, SOUTH, EAST, WEST, STOP).
        
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |game_state| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        game_state.get_legal_actions(agent_index):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        game_state.generate_successor(agent_index, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        game_state.get_pacman_state():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        game_state.get_ghost_states():
            Returns list of AgentState objects for the ghosts

        game_state.get_num_agents():
            Returns the total number of agents in the game

        game_state.get_score():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(
            game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(
            len(scores)) if scores[index] == best_score]
        # Pick randomly among the best
        chosen_index = random.choice(best_indices)


        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.
        
        This is a simple reflex-based evaluation that considers:
        - The game score of the resulting state
        - Distance to the nearest food pellet
        - Distance to ghosts (both active and scared)
        
        Args:
            current_game_state (GameState): The current game state before the action.
            action (str): The proposed action to evaluate (one of the Directions).
        
        Returns:
            float: A numerical score estimating the value of taking the action.
                  Higher scores indicate more desirable actions.
        
        Note:
            The code below extracts some useful information from the state, like the
            remaining food (old_food) and Pacman position after moving (new_pos).
            new_scared_times holds the number of moves that each ghost will remain
            scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(
            action)
        new_pos = successor_game_state.get_pacman_position()
        old_food = current_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [
            ghost_state.scared_timer for ghost_state in new_ghost_states]

        return successor_game_state.get_score()


def score_evaluation_function(current_game_state: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    
    This is a simple baseline evaluation function that only considers the
    current game score without any additional heuristics or lookahead.
    
    Args:
        current_game_state (GameState): The game state to evaluate.
    
    Returns:
        float: The score of the current game state.
    
    Note:
        This evaluation function is meant for use with adversarial search agents
        (not reflex agents). It serves as a simple fallback when no better
        evaluation function is provided.
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    
    This is an abstract base class for adversarial search agents that supports
    depth-limited search with pluggable evaluation functions. All multi-agent
    search algorithms share common attributes like search depth and evaluation
    function, which are configured through this class.
    
    Attributes:
        index (int): Agent index (Pacman is always 0).
        evaluation_function (callable): Function to evaluate terminal/leaf states.
        depth (int): Maximum depth for the search tree (number of full plies).
    
    Args:
        eval_fn (str): Name of the evaluation function to use. Default is
                      'score_evaluation_function'.
        depth (str): String representation of search depth. Default is '2'.
    
    Note:
        You *do not* need to make any changes here, but you can if you want to
        add functionality to all your adversarial search agents.  Please do not
        remove anything, however.

        Note: this is an abstract class: one that should not be instantiated.  It's
        only partially specified, and designed to be extended.  Agent (game.py)
        is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
    
    This agent implements the minimax algorithm for adversarial search in a
    multi-agent environment. It assumes all agents play optimally: Pacman
    maximizes the evaluation score while ghosts minimize it.
    
    The algorithm performs depth-limited search where one depth level consists
    of all agents (Pacman and all ghosts) each taking one action. At leaf nodes
    (terminal states or depth limit reached), the evaluation function is used
    to estimate state value.
    
    Key Characteristics:
        - Optimal play assumption for all agents
        - Complete search within depth limit
        - Exponential time complexity: O(b^(m*d)) where b is branching factor,
          m is number of agents, d is depth
        - No pruning (considers all branches)
    
    Inherits:
        - evaluation_function: Function to evaluate states
        - depth: Maximum search depth
        - index: Agent index (0 for Pacman)
    """

    def get_action(self, game_state: GameState) -> str:
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.
        
        The minimax algorithm assumes all agents (including ghosts) play optimally.
        Pacman (agent 0) maximizes the score, while ghosts (agents >= 1) minimize it.
        The search is depth-limited, where one depth level consists of all agents
        making one move each.
        
        Args:
            game_state (GameState): The current game state from which to determine
                                   the best action.
        
        Returns:
            str: The optimal action according to minimax (one of Directions.NORTH,
                SOUTH, EAST, WEST, or STOP).
        
        Algorithm:
            1. For each legal action of Pacman, compute the minimax value
            2. The minimax value is computed by alternating between max (Pacman)
               and min (ghosts) layers
            3. At terminal states or depth limit, use evaluation_function
            4. Return the action that leads to the maximum minimax value
        """

        # BEGIN_YOUR_CODE (our solution is 22 line(s) of code, but don't worry if you deviate from this)
        def minimax_value(state, depth, agent_index):
            """
            Computes the minimax value for a given state, depth, and agent.
            
            Args:
                state: Current game state
                depth: Current depth in the search tree
                agent_index: Index of the current agent (0 for Pacman, >= 1 for ghosts)
            
            Returns:
                float: The minimax value of the state
            """
            # Terminal state check
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
            
            # Get legal actions for current agent
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)
            
            # Calculate next agent and depth
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth + 1 if next_agent == 0 else depth
            
            # Pacman's turn: maximize
            if agent_index == 0:
                return max(minimax_value(state.generate_successor(agent_index, action),
                                       next_depth, next_agent)
                         for action in legal_actions)
            # Ghost's turn: minimize
            else:
                return min(minimax_value(state.generate_successor(agent_index, action),
                                       next_depth, next_agent)
                         for action in legal_actions)
        
        # Get legal actions for Pacman
        legal_actions = game_state.get_legal_actions(0)
        
        # Choose the action with maximum minimax value
        best_action = max(legal_actions,
                         key=lambda action: minimax_value(
                             game_state.generate_successor(0, action), 0, 1))
        
        return best_action
        # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (problem 2)
    
    This agent implements minimax with alpha-beta pruning, an optimization that
    eliminates branches in the search tree that cannot influence the final decision.
    It produces the same result as standard minimax but with significantly better
    performance.
    
    Alpha-beta pruning maintains two values:
        - Alpha: Best value (highest) the maximizer can guarantee so far
        - Beta: Best value (lowest) the minimizer can guarantee so far
    
    Pruning occurs when alpha >= beta, indicating that the current branch cannot
    affect the final decision and can be safely skipped.
    
    Key Characteristics:
        - Same result as minimax but more efficient
        - Best-case time complexity: O(b^(m*d/2))
        - Average case much better than minimax
        - Move ordering affects pruning efficiency
    
    Reference:
        For pseudocode and detailed explanation, see:
        en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    
    Inherits:
        - evaluation_function: Function to evaluate states
        - depth: Maximum search depth
        - index: Agent index (0 for Pacman)
    """

    def get_action(self, game_state: GameState) -> str:
        """
        Returns the minimax action using self.depth and self.evaluation_function
        with alpha-beta pruning for improved efficiency.
        
        Alpha-beta pruning eliminates branches in the search tree that cannot
        affect the final decision, significantly reducing the number of states
        evaluated while guaranteeing the same result as standard minimax.
        
        Args:
            game_state (GameState): The current game state from which to determine
                                   the best action.
        
        Returns:
            str: The optimal action according to minimax with alpha-beta pruning.
        
        Algorithm:
            - Alpha: The best value that the maximizer can guarantee at current level or above
            - Beta: The best value that the minimizer can guarantee at current level or above
            - Prune when alpha >= beta (no need to explore further)
        """

        # BEGIN_YOUR_CODE (our solution is 43 line(s) of code, but don't worry if you deviate from this)
        def alpha_beta_value(state, depth, agent_index, alpha, beta):
            """
            Computes minimax value with alpha-beta pruning.
            
            Args:
                state: Current game state
                depth: Current depth in the search tree
                agent_index: Index of current agent (0=Pacman, >=1=ghosts)
                alpha: Best value for maximizer found so far
                beta: Best value for minimizer found so far
            
            Returns:
                float: The minimax value with alpha-beta pruning
            """
            # Terminal state check
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
            
            # Get legal actions
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)
            
            # Calculate next agent and depth
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth + 1 if next_agent == 0 else depth
            
            # Pacman's turn: maximize with alpha-beta pruning
            if agent_index == 0:
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alpha_beta_value(successor, next_depth, next_agent, alpha, beta))
                    # Pruning: if value >= beta, the minimizer won't choose this branch
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            # Ghost's turn: minimize with alpha-beta pruning
            else:
                value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alpha_beta_value(successor, next_depth, next_agent, alpha, beta))
                    # Pruning: if value <= alpha, the maximizer won't choose this branch
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        
        # Get legal actions for Pacman
        legal_actions = game_state.get_legal_actions(0)
        
        # Initialize alpha and beta
        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        best_value = float('-inf')
        
        # Find the best action
        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = alpha_beta_value(successor, 0, 1, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_action = action
            
            alpha = max(alpha, best_value)
        
        return best_action
        # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 3)
    
    This agent implements the expectimax algorithm, which is designed for
    adversarial scenarios where opponents don't play optimally. Instead of
    assuming ghosts minimize Pacman's score, it models them as making random
    moves with uniform probability.
    
    The key difference from minimax:
        - Max nodes (Pacman): Still maximize, choose best action
        - Chance nodes (Ghosts): Compute expected value over all actions
          (weighted average assuming uniform distribution)
    
    This is more realistic for modeling:
        - Random or unpredictable opponents
        - Opponents with suboptimal strategies
        - Opponents with unknown policies
    
    Key Characteristics:
        - More realistic for non-optimal opponents
        - Cannot use alpha-beta pruning (need all branches for expectation)
        - Time complexity: O(b^(m*d)) (same as minimax)
        - Expected values are generally higher than minimax (less pessimistic)
    
    Algorithm:
        At max nodes: return max(successors)
        At chance nodes: return average(successors)
        At leaf nodes: return evaluation_function(state)
    
    Inherits:
        - evaluation_function: Function to evaluate states
        - depth: Maximum search depth
        - index: Agent index (0 for Pacman)
    """

    def get_action(self, game_state: GameState) -> str:
        """
        Returns the expectimax action using self.depth and self.evaluation_function.
        
        Unlike minimax which assumes optimal play from all agents, expectimax models
        ghosts as choosing uniformly at random from their legal moves. This is more
        realistic for modeling suboptimal or random opponents.
        
        Args:
            game_state (GameState): The current game state from which to determine
                                   the best action.
        
        Returns:
            str: The optimal action according to expectimax algorithm.
        
        Algorithm:
            1. Pacman still maximizes (same as minimax)
            2. Ghosts are modeled as chance nodes - compute expected value
               over all possible actions weighted by probability (uniform distribution)
            3. Expected value = average of all successor values
        """

        # BEGIN_YOUR_CODE (our solution is 22 line(s) of code, but don't worry if you deviate from this)
        def expectimax_value(state, depth, agent_index):
            """
            Computes the expectimax value for a given state, depth, and agent.
            
            Args:
                state: Current game state
                depth: Current depth in the search tree
                agent_index: Index of current agent (0=Pacman, >=1=ghosts)
            
            Returns:
                float: The expectimax value of the state
            """
            # Terminal state check
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
            
            # Get legal actions
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)
            
            # Calculate next agent and depth
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth + 1 if next_agent == 0 else depth
            
            # Pacman's turn: maximize (same as minimax)
            if agent_index == 0:
                return max(expectimax_value(state.generate_successor(agent_index, action),
                                          next_depth, next_agent)
                         for action in legal_actions)
            # Ghost's turn: compute expected value (average over all actions)
            else:
                successor_values = [expectimax_value(state.generate_successor(agent_index, action),
                                                    next_depth, next_agent)
                                  for action in legal_actions]
                return sum(successor_values) / len(successor_values)
        
        # Get legal actions for Pacman
        legal_actions = game_state.get_legal_actions(0)
        
        # Choose the action with maximum expectimax value
        best_action = max(legal_actions,
                         key=lambda action: expectimax_value(
                             game_state.generate_successor(0, action), 0, 1))
        
        return best_action
        # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def better_evaluation_function(current_game_state: GameState) -> float:
    """
    An advanced evaluation function that provides a more sophisticated estimate
    of a game state's value by considering multiple strategic factors.
    
    This evaluation function considers:
    1. Current game score (baseline)
    2. Distance to nearest food (encourages food collection)
    3. Distance to ghosts (encourages avoiding dangerous ghosts)
    4. Scared ghost timers (encourages chasing scared ghosts)
    5. Remaining food count (penalizes states with more food left)
    6. Capsule availability (encourages eating power pellets)
    
    Args:
        current_game_state (GameState): The game state to evaluate.
    
    Returns:
        float: A numerical score representing the desirability of the state.
               Higher scores indicate better states for Pacman.
    
    Design Rationale:
        - Food distance: Closer food is better (negative contribution)
        - Ghost distance: Stay away from active ghosts, chase scared ones
        - Scared timers: High value for opportunities to eat ghosts
        - Food count: Penalize having lots of remaining food
        - Capsules: Slight penalty for unused capsules (encourages eating them)
    """

    # BEGIN_YOUR_CODE (our solution is 16 line(s) of code, but don't worry if you deviate from this)
    # Terminal states
    if current_game_state.is_win():
        return float('inf')
    if current_game_state.is_lose():
        return float('-inf')
    
    # Extract game state information
    pacman_pos = current_game_state.get_pacman_position()
    food_list = current_game_state.get_food().as_list()
    ghost_states = current_game_state.get_ghost_states()
    capsules = current_game_state.get_capsules()
    
    # Start with current score
    score = current_game_state.get_score()
    
    # Food distance component: prioritize getting closer to food
    if food_list:
        min_food_dist = min(manhattan_distance(pacman_pos, food) for food in food_list)
        score += 10.0 / (min_food_dist + 1)  # Closer food = higher score
    
    # Ghost component: avoid active ghosts, chase scared ones
    for ghost_state in ghost_states:
        ghost_pos = ghost_state.get_position()
        ghost_dist = manhattan_distance(pacman_pos, ghost_pos)
        
        if ghost_state.scared_timer > 0:
            # Chase scared ghosts (closer = better)
            score += 100.0 / (ghost_dist + 1)
        else:
            # Avoid active ghosts (farther = better)
            if ghost_dist < 2:
                score -= 500  # Heavy penalty for being too close
            else:
                score += ghost_dist  # Reward for staying away
    
    # Penalize remaining food
    score -= 10 * len(food_list)
    
    # Penalize remaining capsules (encourages eating them)
    score -= 20 * len(capsules)
    
    return score
    # END_YOUR_CODE


# Abbreviation
better = better_evaluation_function
