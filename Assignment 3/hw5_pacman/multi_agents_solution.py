# multi_agents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattan_distance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    
    Reflex agents make decisions based on immediate consequences without
    looking ahead multiple moves. They evaluate each possible action and
    choose the one with the highest evaluation score.
    
    This implementation includes an improved evaluation function that considers:
    - Distance to nearest food
    - Distance to ghosts (active and scared)
    - Game score changes
    
    Attributes:
        None (inherits from Agent)
    
    Methods:
        get_action(game_state): Selects the best action based on evaluation scores.
        evaluation_function(current_game_state, action): Evaluates action quality.
    
    Note:
        The code below is provided as a guide. You are welcome to change
        it in any way you see fit, so long as you don't touch our method
        headers.
    """


    def get_action(self, game_state):
        """
        Selects the best action based on the evaluation function.
        
        This method evaluates all legal moves and chooses the one with the highest
        score. If multiple actions have equal scores, one is randomly selected.
        
        Args:
            game_state: The current GameState object containing game information.
        
        Returns:
            str: A direction from Directions (NORTH, SOUTH, EAST, WEST, or STOP).
        
        Note:
            You do not need to change this method, but you're welcome to.
            get_action chooses among the best options according to the evaluation function.
            Just like in the previous project, get_action takes a GameState and returns
            some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.
        
        Evaluates the quality of taking a specific action from the current state
        by considering multiple factors that contribute to a good game position.
        
        This implementation considers:
        1. Game score improvement from the action
        2. Distance to the nearest food pellet
        3. Distance to ghosts (avoid active ghosts, chase scared ones)
        4. Whether food is eaten by this action
        
        Args:
            current_game_state: The current GameState before taking the action.
            action: The proposed action to evaluate.
        
        Returns:
            float: A numerical score where higher values indicate better actions.
                  Combines multiple heuristics to guide Pacman's behavior.
        
        Implementation Details:
            The evaluation function takes in the current and proposed successor
            GameStates (pacman.py) and returns a number, where higher numbers are better.

            The code below extracts some useful information from the state, like the
            remaining food (old_food) and Pacman position after moving (new_pos).
            new_scared_times holds the number of moves that each ghost will remain
            scared because of Pacman having eaten a power pellet.

            Print out these variables to see what you're getting, then combine them
            to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(
            action)
        new_pos = successor_game_state.get_pacman_position()
        old_food = current_game_state.get_food()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [
            ghost_state.scared_timer for ghost_state in new_ghost_states]

        # Start with the game score
        score = successor_game_state.get_score()
        
        # Food distance component
        food_list = new_food.as_list()
        if food_list:
            min_food_distance = min(manhattan_distance(new_pos, food) for food in food_list)
            # Reward being closer to food
            score += 10.0 / (min_food_distance + 1)
        
        # Bonus for eating food
        if old_food[new_pos[0]][new_pos[1]]:
            score += 100
        
        # Ghost distance component
        for ghost_state in new_ghost_states:
            ghost_pos = ghost_state.get_position()
            ghost_distance = manhattan_distance(new_pos, ghost_pos)
            
            if ghost_state.scared_timer > 0:
                # Chase scared ghosts
                score += 200.0 / (ghost_distance + 1)
            else:
                # Avoid active ghosts - heavy penalty if too close
                if ghost_distance < 2:
                    score -= 1000
                elif ghost_distance < 4:
                    score -= 100
        
        # Small penalty for stopping (encourage movement)
        if action == Directions.STOP:
            score -= 5
        
        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    
    This is a simple baseline that considers only the current game score
    without any additional heuristics or strategic considerations.
    
    Args:
        current_game_state: The GameState to evaluate.
    
    Returns:
        float: The numerical score of the current state.
    
    Note:
        This evaluation function is meant for use with adversarial search agents
        (not reflex agents). It serves as a default when no better evaluation
        function is specified.
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    
    This abstract base class provides shared functionality for adversarial
    search agents including depth-limited search and pluggable evaluation
    functions. All multi-agent algorithms inherit these common attributes.
    
    Attributes:
        index (int): The agent's index (Pacman is always 0).
        evaluation_function (callable): Function to evaluate leaf/terminal states.
        depth (int): Maximum depth for search tree exploration.
    
    Args:
        eval_fn (str): Name of evaluation function to use. Defaults to
                      'score_evaluation_function'.
        depth (str): Search depth as string. Defaults to '2'.
    
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


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    
    Implements the minimax algorithm for adversarial game search where
    Pacman maximizes and ghosts minimize the evaluation score. This assumes
    all agents play optimally.
    
    The search is depth-limited: one depth consists of all agents making
    one move each. The algorithm recursively evaluates game trees by
    alternating between max (Pacman) and min (ghost) layers.
    
    Key Properties:
        - Optimal against perfect opponents
        - Complete within depth limit
        - Time complexity: O(b^(m*d)) where b=branching factor, m=agents, d=depth
        - Space complexity: O(m*d) for recursion stack
    
    Inherits:
        - index: Pacman's agent index (0)
        - evaluation_function: Evaluates terminal/leaf states
        - depth: Maximum search depth
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.
        
        Implements depth-limited minimax where Pacman (agent 0) maximizes and
        ghosts minimize. At leaf nodes (terminal states or depth limit), uses
        the evaluation function to estimate state value.
        
        Args:
            game_state: The current GameState to search from.
        
        Returns:
            str: The optimal action according to minimax (Directions.NORTH/SOUTH/EAST/WEST/STOP).
        
        Algorithm Overview:
            1. For each legal Pacman action, compute minimax value
            2. Recursively alternate between max/min layers for each agent
            3. At terminal states or depth limit, return evaluation
            4. Return action leading to maximum value
        
        Useful Methods:
            game_state.get_legal_actions(agent_index):
                Returns a list of legal actions for an agent
                agent_index=0 means Pacman, ghosts are >= 1

            Directions.STOP:
                The stop direction, which is always legal

            game_state.generate_successor(agent_index, action):
                Returns the successor game state after an agent takes an action

            game_state.get_num_agents():
                Returns the total number of agents in the game
        """
        def minimax_value(state, depth, agent_index):
            """
            Recursively computes minimax value for the given state.
            
            Args:
                state: Current game state
                depth: Current depth level
                agent_index: Index of current agent (0=Pacman, >=1=ghosts)
            
            Returns:
                float: Minimax value of the state
            """
            # Terminal conditions
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
            
            # Get legal actions
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)
            
            # Determine next agent and depth
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
        
        # Find action with maximum minimax value
        best_action = max(legal_actions,
                         key=lambda action: minimax_value(
                             game_state.generate_successor(0, action), 0, 1))
        
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    
    Implements minimax with alpha-beta pruning to efficiently eliminate
    branches that cannot affect the final decision. Produces identical
    results to standard minimax but with significantly better performance.
    
    Alpha-beta pruning maintains:
        - Alpha: Best value (highest) maximizer can guarantee
        - Beta: Best value (lowest) minimizer can guarantee
    
    Pruning occurs when alpha >= beta, indicating the current branch
    cannot influence the final decision.
    
    Key Properties:
        - Same result as minimax, more efficient
        - Best-case complexity: O(b^(m*d/2))
        - Average case: Much better than minimax
        - Move ordering affects pruning efficiency
    
    Reference:
        Alpha-beta pruning pseudocode:
        en.wikipedia.org/wiki/Alpha–beta_pruning#Pseudocode
    
    Inherits:
        - index: Pacman's agent index (0)
        - evaluation_function: Evaluates terminal/leaf states
        - depth: Maximum search depth
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        with alpha-beta pruning optimization.
        
        Alpha-beta pruning cuts off branches that cannot improve the final
        decision, maintaining the same result as minimax while evaluating
        fewer states.
        
        Args:
            game_state: The current GameState to search from.
        
        Returns:
            str: The optimal action (Directions.NORTH/SOUTH/EAST/WEST/STOP).
        
        Algorithm:
            - Maintain alpha (best for max) and beta (best for min)
            - Prune branches when alpha >= beta
            - Return same result as minimax with fewer evaluations
        """
        def alpha_beta_value(state, depth, agent_index, alpha, beta):
            """
            Computes minimax value with alpha-beta pruning.
            
            Args:
                state: Current game state
                depth: Current depth level
                agent_index: Index of current agent (0=Pacman, >=1=ghosts)
                alpha: Best maximizer value found so far
                beta: Best minimizer value found so far
            
            Returns:
                float: Minimax value with pruning
            """
            # Terminal conditions
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
            
            # Get legal actions
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)
            
            # Determine next agent and depth
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth + 1 if next_agent == 0 else depth
            
            # Pacman's turn: maximize with pruning
            if agent_index == 0:
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alpha_beta_value(successor, next_depth, 
                                                       next_agent, alpha, beta))
                    # Prune if value exceeds beta
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            # Ghost's turn: minimize with pruning
            else:
                value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alpha_beta_value(successor, next_depth,
                                                       next_agent, alpha, beta))
                    # Prune if value falls below alpha
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        
        # Initialize alpha and beta
        legal_actions = game_state.get_legal_actions(0)
        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        best_value = float('-inf')
        
        # Find best action with alpha-beta pruning
        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = alpha_beta_value(successor, 0, 1, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_action = action
            
            alpha = max(alpha, best_value)
        
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    
    Implements the expectimax algorithm for adversarial search against
    non-optimal opponents. Instead of assuming ghosts minimize Pacman's
    score, models them as choosing uniformly at random.
    
    Key Differences from Minimax:
        - Max nodes (Pacman): Still maximize (choose best action)
        - Chance nodes (Ghosts): Compute expected value (weighted average)
          assuming uniform probability distribution
    
    This is more realistic for:
        - Random or unpredictable opponents
        - Suboptimal adversaries
        - Opponents with unknown strategies
    
    Key Properties:
        - More realistic than minimax for non-optimal play
        - Cannot use alpha-beta pruning (need all branches)
        - Time complexity: O(b^(m*d)) (same as minimax)
        - Generally higher values than minimax (less pessimistic)
    
    Algorithm:
        Max nodes: return max(successors)
        Chance nodes: return average(successors)
        Leaf nodes: return evaluation_function(state)
    
    Inherits:
        - index: Pacman's agent index (0)
        - evaluation_function: Evaluates terminal/leaf states
        - depth: Maximum search depth
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function.
        
        Pacman maximizes while ghosts are modeled as chance nodes that choose
        uniformly at random from their legal moves. This produces more realistic
        behavior against suboptimal opponents.
        
        Args:
            game_state: The current GameState to search from.
        
        Returns:
            str: The optimal action according to expectimax (Directions.NORTH/SOUTH/EAST/WEST/STOP).
        
        Note:
            All ghosts should be modeled as choosing uniformly at random from their
            legal moves. Expected value is computed as the average of all successor
            values.
        """
        def expectimax_value(state, depth, agent_index):
            """
            Recursively computes expectimax value.
            
            Args:
                state: Current game state
                depth: Current depth level
                agent_index: Index of current agent (0=Pacman, >=1=ghosts)
            
            Returns:
                float: Expectimax value of the state
            """
            # Terminal conditions
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
            
            # Get legal actions
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)
            
            # Determine next agent and depth
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth + 1 if next_agent == 0 else depth
            
            # Pacman's turn: maximize (same as minimax)
            if agent_index == 0:
                return max(expectimax_value(state.generate_successor(agent_index, action),
                                          next_depth, next_agent)
                         for action in legal_actions)
            # Ghost's turn: expected value (average of all actions)
            else:
                successor_values = [expectimax_value(state.generate_successor(agent_index, action),
                                                    next_depth, next_agent)
                                  for action in legal_actions]
                return sum(successor_values) / len(successor_values)
        
        # Get legal actions for Pacman
        legal_actions = game_state.get_legal_actions(0)
        
        # Choose action with maximum expectimax value
        best_action = max(legal_actions,
                         key=lambda action: expectimax_value(
                             game_state.generate_successor(0, action), 0, 1))
        
        return best_action


def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    
    This advanced evaluation function provides a sophisticated assessment of
    game states by considering multiple strategic factors that contribute to
    winning the game efficiently.
    
    Strategic Considerations:
    1. **Current Score**: Baseline game score (most important)
    2. **Food Distance**: Proximity to nearest food pellet (encourages collection)
    3. **Food Count**: Number of remaining food pellets (fewer is better)
    4. **Ghost Behavior**: 
       - Active ghosts: Maintain safe distance (avoid)
       - Scared ghosts: Chase aggressively (high reward)
    5. **Capsules**: Remaining power pellets (encourage consumption)
    6. **Win/Loss States**: Infinite values for terminal states
    
    Args:
        current_game_state: The GameState to evaluate.
    
    Returns:
        float: A comprehensive score where higher values indicate better states.
               Returns infinity for winning states, negative infinity for losing.
    
    DESCRIPTION:
        This evaluation function combines multiple weighted heuristics:
        
        - Base score from game state (primary metric)
        - 1/(distance to nearest food) to encourage food collection
        - Penalty proportional to remaining food count
        - Heavy penalty for being near active ghosts (<2 distance)
        - High reward for being near scared ghosts
        - Moderate penalty for remaining capsules (encourages power-up usage)
        
        The weights are tuned to balance aggressive food collection with
        ghost avoidance, while opportunistically hunting scared ghosts.
        Terminal states (win/lose) receive infinite values to prioritize
        game-ending moves.
    """
    # Terminal states have infinite value
    if current_game_state.is_win():
        return float('inf')
    if current_game_state.is_lose():
        return float('-inf')
    
    # Extract game state information
    pacman_pos = current_game_state.get_pacman_position()
    food_list = current_game_state.get_food().as_list()
    ghost_states = current_game_state.get_ghost_states()
    capsules = current_game_state.get_capsules()
    
    # Start with current game score
    score = current_game_state.get_score()
    
    # Food distance component: encourage moving toward nearest food
    if food_list:
        min_food_dist = min(manhattan_distance(pacman_pos, food) for food in food_list)
        score += 10.0 / (min_food_dist + 1)  # Inverse distance reward
    
    # Ghost evaluation: avoid active ghosts, chase scared ones
    for ghost_state in ghost_states:
        ghost_pos = ghost_state.get_position()
        ghost_dist = manhattan_distance(pacman_pos, ghost_pos)
        
        if ghost_state.scared_timer > 0:
            # Scared ghost: chase aggressively (closer = better)
            score += 200.0 / (ghost_dist + 1)
            # Extra bonus if very close to scared ghost
            if ghost_dist <= 1:
                score += 500
        else:
            # Active ghost: maintain safe distance
            if ghost_dist < 2:
                score -= 1000  # Critical danger - heavy penalty
            elif ghost_dist < 4:
                score -= 50    # Moderate danger - lighter penalty
            else:
                score += ghost_dist * 2  # Reward for being far away
    
    # Penalize remaining food (encourage finishing game)
    score -= 4 * len(food_list)
    
    # Penalize remaining capsules (encourage using power-ups)
    score -= 10 * len(capsules)
    
    return score


# Abbreviation
better = better_evaluation_function


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
    
    An advanced competitive agent designed for the mini-contest using a
    combination of expectimax search with an optimized evaluation function.
    
    This agent assumes ghosts will generally move toward (or away from) Pacman
    rather than acting randomly, and uses aggressive strategies to maximize
    score while managing risk.
    
    Key Features:
        - Uses expectimax with adaptive depth
        - Advanced evaluation considering multiple factors
        - Balanced aggression for food collection and ghost avoidance
        - Opportunistic behavior with scared ghosts
    
    Strategy:
        The agent combines expectimax search (modeling non-optimal but
        predictable ghost behavior) with a sophisticated evaluation function
        that weighs food collection, ghost distances, capsule usage, and
        game completion urgency.
    
    Note:
        Ghosts don't behave randomly anymore, but they aren't perfect either --
        they'll usually just make a beeline straight towards Pacman (or away
        from him if they're scared!)
    
    Inherits:
        - index: Pacman's agent index (0)
        - evaluation_function: Advanced evaluation function
        - depth: Search depth (can be adjusted for speed/quality tradeoff)
    """

    def get_action(self, game_state):
        """
        Returns an action for the mini-contest using advanced search strategy.
        
        Uses expectimax algorithm with an optimized evaluation function tuned
        for competitive play. Balances search depth with time constraints.
        
        Args:
            game_state: The current GameState.
        
        Returns:
            str: The selected action (Directions.NORTH/SOUTH/EAST/WEST/STOP).
        
        Strategy Details:
            You can use any method you want and search to any depth you want.
            Just remember that the mini-contest is timed, so you have to trade
            off speed and computation.

            Ghosts don't behave randomly anymore, but they aren't perfect either --
            they'll usually just make a beeline straight towards Pacman (or away
            from him if they're scared!)
        """
        def contest_evaluation(state):
            """
            Optimized evaluation function for contest play.
            
            Args:
                state: GameState to evaluate
            
            Returns:
                float: Evaluation score optimized for competitive play
            """
            # Terminal states
            if state.is_win():
                return float('inf')
            if state.is_lose():
                return float('-inf')
            
            pacman_pos = state.get_pacman_position()
            food_list = state.get_food().as_list()
            ghost_states = state.get_ghost_states()
            capsules = state.get_capsules()
            
            score = state.get_score()
            
            # Food strategy: aggressive collection
            if food_list:
                min_food_dist = min(manhattan_distance(pacman_pos, food) for food in food_list)
                score += 15.0 / (min_food_dist + 1)
            
            # Ghost strategy: calculated risk-taking
            for ghost_state in ghost_states:
                ghost_pos = ghost_state.get_position()
                ghost_dist = manhattan_distance(pacman_pos, ghost_pos)
                
                if ghost_state.scared_timer > 0:
                    # Very aggressive with scared ghosts
                    score += 300.0 / (ghost_dist + 1)
                    if ghost_dist <= 1:
                        score += 1000  # Huge bonus for eating ghost
                else:
                    # Conservative with active ghosts
                    if ghost_dist <= 1:
                        score -= 2000  # Avoid certain death
                    elif ghost_dist == 2:
                        score -= 200   # High danger zone
                    elif ghost_dist <= 4:
                        score -= 50    # Caution zone
            
            # Completion urgency
            score -= 5 * len(food_list)
            score -= 15 * len(capsules)
            
            return score
        
        def expectimax_contest(state, depth, agent_index):
            """
            Expectimax search optimized for contest.
            
            Args:
                state: Current game state
                depth: Current depth
                agent_index: Current agent
            
            Returns:
                float: Expectimax value
            """
            # Adaptive depth limit (can be tuned)
            max_depth = 3
            
            if state.is_win() or state.is_lose() or depth == max_depth:
                return contest_evaluation(state)
            
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return contest_evaluation(state)
            
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth + 1 if next_agent == 0 else depth
            
            # Pacman: maximize
            if agent_index == 0:
                return max(expectimax_contest(state.generate_successor(agent_index, action),
                                             next_depth, next_agent)
                         for action in legal_actions)
            # Ghosts: expected value
            else:
                values = [expectimax_contest(state.generate_successor(agent_index, action),
                                            next_depth, next_agent)
                         for action in legal_actions]
                return sum(values) / len(values)
        
        # Get best action using expectimax
        legal_actions = game_state.get_legal_actions(0)
        
        # Avoid stopping unless necessary
        if len(legal_actions) > 1 and Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        
        best_action = max(legal_actions,
                         key=lambda action: expectimax_contest(
                             game_state.generate_successor(0, action), 0, 1))
        
        return best_action


