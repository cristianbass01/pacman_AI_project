# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import contest.util as util
from contextlib import contextmanager
import signal
import typing as t
from enum import Enum
from numpy import mean
from contest.util import nearestPoint

# Standard imports
from contest.captureAgents import CaptureAgent
import random
from contest.game import Directions, Actions  # basically a class to store data


# this is the entry points to instanciate you agents
def create_team(first_index, second_index, is_red,
                first='offensiveAgent', second='DefensiveReflexAgent', num_training=0):

    # capture agents must be instanciated with an index
    # time to compute id 1 second in real game
    return [eval(first)(first_index), eval(second)(second_index)]


class agentBase(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.index = index
        self.prev_food = None
        
        self.gridHeight = None
        self.gridWidth = None
        self.start = None

        self.safe_boundary = None
        self.danger_boundary = None

        self.boundary_pos = None
        self.danger_pos = None

        self.next_moves = None

        self.last_seen_enemy_pos = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        # the following initialises self.red and self.distancer
        CaptureAgent.register_initial_state(self, game_state)

        self.gridWidth = self.get_food(game_state).width
        self.gridHeight = self.get_food(game_state).height
        if self.red:
            self.safe_boundary = int(self.gridWidth / 2) - 1
            self.danger_boundary = int(self.gridWidth / 2)
        else:
            self.safe_boundary = int(self.gridWidth / 2)
            self.danger_boundary = int(self.gridWidth / 2) -1

        self.prev_food = len(self.get_food(game_state).as_list())

        self.boundary_pos = []
        for y in range(self.gridHeight):
            if not game_state.has_wall(self.safe_boundary, y):
                self.boundary_pos.append((self.safe_boundary, y))

        self.danger_pos = []
        for y in range(self.gridHeight):
            if not game_state.has_wall(self.danger_boundary, y):
                self.danger_pos.append((self.danger_boundary, y))

class offensiveAgent(agentBase):

    def choose_action(self, game_state):
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action
        no_moves = self.next_moves == None or (self.next_moves != None and len(self.next_moves) == 0)
        exist_threat = any([is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)])
        
        current_food = len(self.get_food(game_state).as_list())
        food_changed = self.prev_food > current_food
        start = self.start == game_state.get_agent_position(self.index)

        if food_changed:
            self.prev_food = len(self.get_food(game_state).as_list())

        if no_moves or exist_threat or food_changed or start:
            problem = FoodOffense(startingGameState=game_state, captureAgent=self)
            actions = aStarSearch(problem, heuristic=self.offensiveHeuristic, agent=self)
            self.next_moves = actions

        # this can occure if start in the goal state. In this case do not want to perform any action.
        if self.next_moves == []:
            self.next_moves = ["Stop"]

        next_action = self.next_moves[0]
        self.next_moves = self.next_moves[1:]
        return next_action
    
    def offensiveHeuristic(self, state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  
        
        captureAgent = problem.captureAgent
        index = captureAgent.index
        game_state = state[0]

        # check if we have reached a goal state and explicitly return 0
        if captureAgent.red == True:
            if game_state.data.score_change >= problem.MINIMUM_IMPROVEMENT:
                return 0
        # If blue team, want scores to go down
        else:
            if game_state.data.score_change <= - problem.MINIMUM_IMPROVEMENT:
                return 0

        agent_state = game_state.get_agent_state(index)
        food_carrying = agent_state.num_carrying

        myPos = game_state.get_agent_state(index).get_position()

        # this will be updated to be closest food location if not collect enough food
        return_home_from = myPos

        # still need to collect food
        dist_to_food = 0
        if food_carrying < problem.MINIMUM_IMPROVEMENT:
            # distance to the closest food
            food_list = captureAgent.get_food(game_state).as_list()

            min_pos = None
            min_dist = 99999999
            for food in food_list:
                dist = captureAgent.get_maze_distance(myPos, food)
                if dist < min_dist:
                    min_pos = food
                    min_dist = dist

            dist_to_food = min_dist
            return_home_from = min_pos
            return dist_to_food

        # Returning Home
        # WARNING: this assumes the maps are always semetrical, territory is divided in half, red on right, blue on left
        walls = list(game_state.get_walls())
        y_len = len(walls[0])
        x_len = len(walls)
        mid_point_index = int(x_len/2)
        if captureAgent.red:
            mid_point_index -= 1

        # find all the entries and find distance to closest
        entry_coords = []
        for i, row in enumerate(walls[mid_point_index]):
            if row is False:  # there is not a wall
                entry_coords.append((int(mid_point_index), int(i)))

        minDistance = min([captureAgent.get_maze_distance(return_home_from, entry)
                        for entry in entry_coords])
        return dist_to_food + minDistance"""

        game_state = state[0]

        my_pos = game_state.get_agent_position(self.index)
        prev_game_state = self.get_previous_observation()

        offence_food_pos = self.get_food(game_state).as_list()
        defence_food_pos = self.get_food_you_are_defending(game_state).as_list()

        if prev_game_state != None:
            prev_offence_food_pos = self.get_food(prev_game_state).as_list()
            prev_defence_food_pos = self.get_food_you_are_defending(prev_game_state).as_list()
        else: 
            prev_defence_food_pos = defence_food_pos
            prev_offence_food_pos = offence_food_pos

        offence_capsule_pos = self.get_capsules(game_state)
        defence_capsule_pos = self.get_capsules_you_are_defending(game_state)
        
        current_score = self.get_score(game_state)
        
        #Usefull
        # self.is_pacman
        # self.scared_timer
        # self.num_carrying
        # self.num_returned
        ### TODO ADD DISTRIBUTIONS
        """
        # Pacman
        FOOD_CARRYING_MUL = 2
        FOOD_RETURNED_MUL = 1
        EATING_CAPSULE_MUL = 10
        RETURN_HOME_ALL_FOOD_MUL = 10
        RETURN_HOME_CHASED_MUL = 1
        EAT_ENEMY_MUL = 10
        RETURN_HOME_ENOUGH_FOOD_MUL = 1
        ENEMY_PENALTY_MUL = 1
        TEAMMATE_PENALTY_MUL = 5
        SCORE_MUL = 30
        DEAD_MUL = 100
        
        # Not pacman
        DISTANCE_FOOD_MUL = 5
        DISTANCE_CLOSER_ENEMY_MUL = 100
        CLOSE_TO_DANGER_POS_MUL = 20
        DISTANCE_TO_DANGER_POS_MUL = 5

        # Worst if it dies
        
        agent_state = game_state.get_agent_state(self.index)

        if my_pos == self.start:
            return -10 * DEAD_MUL
        
        if isPacman(game_state, self):

            # If being chased the heuristic is based on the distance of the capsule or home    
            reward_chased = 0
            if any(is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)):
                # TODO check if we can make this part faster cause when I see
                # enemies the game slows down due to recalculating constantly

                # So I am being chased: 
                # The reward should be defined by the distance to the capsules, distance to home and distance to enemy
                distToClosestEnemy = distanceClosestEnemy(game_state, self)

                for capsule in offence_capsule_pos:  
                    distToCapsule = distance(game_state, self, capsule)
                    canKill = distToCapsule < distToClosestEnemy
                    reward_chased -= canKill * EATING_CAPSULE_MUL * distToCapsule # Adjust the reward value as needed
                
                # Distance home. For some reason crashes here. idk why TODO FIX THIS
                # reward_chased -= min([distance(game_state, self, border_pos) for border_pos in self.boundary_pos]) * RETURN_HOME_CHASED_MUL

                # Distance closest enemy
                # TODO This I think should be reworked based on if we want to kill the enemy
                reward_chased += distToClosestEnemy * DISTANCE_CLOSER_ENEMY_MUL

                # Mean distance to all enemies TODO maybe add this??
                # We are slow currently so maybe we should simplify
                # reward_chased

                # If in danger the only scope is to return home or eat a capsule, no more food
                return reward_chased

            # If the enemy is scared and the time to reach him is enough I can try also to kill him (but it is dangerous to follow him)
            reward_enemy_scared = 0
            closestEnemyAgent, closestEnemyDistance = closestEnemy(game_state, self)
            isScared = closestEnemyAgent.scared_timer != 0
            if isScared and closestEnemyAgent.scared_timer <= closestEnemyDistance:
                reward_enemy_scared -= closestEnemyDistance * EAT_ENEMY_MUL

            # Reward for eating food
            #food_reward = agent_state.num_carrying * FOOD_CARRYING_MUL

            # Reward for returning food
            #food_reward += agent_state.num_returned * FOOD_RETURNED_MUL

            # I want to be less distant from food
            distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offens_food_pos])
            food_reward = -distanceClosestFood * DISTANCE_FOOD_MUL

            # If I am carrying a lot of food, maybe I should do something else (returning home)
            food_reward -= agent_state.num_carrying * FOOD_CARRYING_MUL

            # Reward for returning home once having all the food for win
            home_reward = 0
            if len(offens_food_pos) <= 2:
                home_reward -= min([distance(game_state, self, border_pos) for border_pos in self.boundary_pos]) * RETURN_HOME_ALL_FOOD_MUL

            teammate_penalty = 0
            ## Penalty for being closer to a teammate while on offense
            for teammate in self.get_team(game_state):
                if game_state.get_agent_state(teammate).is_pacman:
                    teammate_pos = game_state.get_agent_position(teammate)
                    teammate_penalty += distance(game_state, self, teammate_pos) * TEAMMATE_PENALTY_MUL

            # Reward based on the score MAYBE DON'T NEEDED
            #score_reward = current_score * SCORE_MUL

            # Penalty for being closer to an enemy while on offense ALREADY IN BEING CHASED
            #enemy_penalty = sum(self.get_maze_distance(my_pos, enemy) for enemy in enemy_positions) * ENEMY_PENALTY_MUL

            # Combine rewards and penalties to form the heuristic
            heuristic = food_reward + home_reward + teammate_penalty + reward_enemy_scared
        else:
            reward_not_pacman = 0  

            # If not pacman means that it is in the safe zone
            # If it is in the safe zone, the scope will be to return to the enemy zone and eat!
            distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offens_food_pos])
            reward_not_pacman -= distanceClosestFood * DISTANCE_FOOD_MUL
              
            # Before returning to the enemy zone I need to see if there are other ghosts near me that can kill me or chased me
            if any(is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)):
                # Distance closest enemy
                reward_not_pacman -= distanceClosestEnemy(game_state, self) * DISTANCE_CLOSER_ENEMY_MUL
            
            teammate_penalty = 0
            ## Penalty for being closer to a teammate while on defence
            for teammate in self.get_team(game_state):
                if not game_state.get_agent_state(teammate).is_pacman:
                    teammate_pos = game_state.get_agent_position(teammate)
                    teammate_penalty += distance(game_state, self, teammate_pos) * TEAMMATE_PENALTY_MUL

            # Better if I will stay closer to the border in order to pass (but not necessary in order to 'go after the hill' if necessary)
            # Below currently not working TODO FIX
            #distanceToClosestDangerPos = min([distance(game_state, self, danger_pos) for danger_pos in self.danger_boundary])
            # reward_not_pacman -= distanceToClosestDangerPos * DISTANCE_TO_DANGER_POS_MUL
            
            heuristic = reward_not_pacman + teammate_penalty
        """
        # I want to be less distant from food
        distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
        heuristic = -distanceClosestFood * 0.1
        return heuristic
    
def is_threat(game_state, my_agent, enemy_agent):
    enemy_pos = game_state.get_agent_position(enemy_agent)
    # Also possible to use self.get_agent_distances for fuzzy estimate of
    # distance without knowing the enemy_pos (| NIKI) TODO
    if enemy_pos == None:
        return False
    isGhost = not game_state.get_agent_state(enemy_agent).is_pacman
    #isScared = game_state.get_agent_state(enemy_agent).is_scared
    scaredTimer = game_state.get_agent_state(enemy_agent).scared_timer

    # Control if enemy is a ghost which we can not kill
    if isGhost and scaredTimer <= distance(game_state, my_agent, enemy_pos):
        return True
    
    return False

def getCost(my_agent, my_pos, enemy_pos, maxCost = 30):
    if enemy_pos != None:
        return my_agent.get_maze_distance(my_pos, enemy_pos)
    return maxCost

def closestEnemy(game_state, my_agent):
    """
    Return the state of the closest enemy
    """
    positionsSorted = util.PriorityQueue()
    my_pos = game_state.get_agent_position(my_agent.index)

    for enemy in my_agent.get_opponents(game_state):

        enemy_pos = game_state.get_agent_position(enemy)
        positionsSorted.push(
            enemy, getCost(my_agent, my_pos, enemy_pos))

    closestEnemy = positionsSorted.pop()
    enemy_pos = game_state.get_agent_position(closestEnemy)
    enemy_state = game_state.get_agent_state(closestEnemy)
    return enemy_state, getCost(my_agent, my_pos, enemy_pos)

def getPos(game_state, agent):
    return game_state.get_agent_position(agent.index)

def isPacman(game_state, agent):
    return game_state.get_agent_state(agent.index).is_pacman

def distance(game_state, agent, toPos):
    my_pos = game_state.get_agent_position(agent.index)
    return agent.get_maze_distance(my_pos, toPos)

def distanceClosestEnemy(game_state, my_agent):
    _, closestEnemyDistance = closestEnemy(game_state, my_agent)
    return closestEnemyDistance

    
#################  problems and heuristics  ####################


class FoodOffense():
    '''
    This problem extends FoodOffense by updateing the enemy ghost to move to our pacman if they are adjacent (basic Goal Recognition techniques).
    This conveys to our pacman the likely effect of moving next to an enemy ghost - but doesn't prohibit it from doing so (e.g if Pacman has been trapped)
    Note: This is a SearchProblem class. It could inherit from search.Search problem (mainly for conceptual clarity).
    '''

    def __init__(self, startingGameState, captureAgent):
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        self.expanded = 0
        self.captureAgent = captureAgent
        self.startingGameState = startingGameState
        # Need to ignore previous score change, as everything should be considered relative to this state
        self.startingGameState.data.score_change = 0
        self.MINIMUM_IMPROVEMENT = 1
        self.DEPTH_CUTOFF = 20
        # WARNING: Capture agent doesn't update with new state, this should only be used for non state dependant utils (e.g distancer)
        self.my_agent = captureAgent
        self.goal_state_found = None

    def getStartState(self):
        # This needs to return the state information to being with
        return (self.startingGameState, self.startingGameState.get_score(), None)

    def isGoalState(self, state):
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        # Goal state when:
        # - Pacman is in our territory
        # - has eaten x food: This comes from the score changing
        # these are both captured by the score changing by a certain amount

        # Note: can't use CaptureAgent, at it doesn't update with game state
        game_state = state[0]

        agent_state = game_state.get_agent_state(self.my_agent.index)
        currentPos = game_state.get_agent_position(self.my_agent.index)

        # Try to get previous game_state (if exists)
        prev_game_state = state[2]
        prev_agent_state = None
        if prev_game_state != None:
            prev_agent_state = prev_game_state.get_agent_state(self.my_agent.index)

        # Being eaten is always not a goal state
        # Calculating the distance from the past position and the current position to see if I am being killed
        if prev_agent_state != None:
            pastPos = prev_game_state.get_agent_position(self.my_agent.index)
            distancePastToCurrentPos = self.my_agent.get_maze_distance(pastPos, currentPos)
            # I can cover a distance more than 1 only if I have being eaten
            if distancePastToCurrentPos > 2:
                return False


        if isPacman(game_state, self.my_agent):

            closestEnemyAgent, closestEnemyDistance = closestEnemy(game_state, self.my_agent)
            isScared = closestEnemyAgent.scared_timer != 0

            # If being chased my goal is not being chased anymore (so eating a capsule or return home)
            # So if there is a threat, I'm not in the goal state    
            if any(is_threat(game_state, self.my_agent, enemy) for enemy in self.my_agent.get_opponents(game_state)):
                if not closestEnemyDistance > 5:
                    # If the enemy is closer than 5 squares mean that he can follow me
                    # This is not a goal state
                    return False
                    

            # If the enemy is scared and the time to reach him is enough I can try also to kill him (but it is dangerous to follow him)
            if isScared and closestEnemyAgent.scared_timer <= closestEnemyDistance:
                # If it is scared and the distance is not enough to kill him it is not a goal
                return False

            # If there is no threat my goal is to eat or return home to collect the food that I am carrying
            # TODO I think these depends on how far we are from the boundary, so it will depend on the heuristic
            # But carrying more food or having returned more food is a goal state
            if prev_agent_state != None and prev_agent_state.num_carrying < agent_state.num_carrying:
                return True
                
            if prev_agent_state != None and prev_agent_state.num_returned < agent_state.num_returned:
                return True
        else:
            # If not pacman means that it is in the safe zone
            # The only reason to return to the safe zone if to escape from an agent or to collect food
            if prev_agent_state != None and prev_game_state!= None and any(is_threat(prev_game_state, self.my_agent, enemy) for enemy in self.my_agent.get_opponents(prev_game_state)):
                return True
            
            if prev_agent_state != None and prev_agent_state.num_returned < agent_state.num_returned:
                return True
            
        return False
        
    
    def get_successors(self, state, path = None):
        """
        Your get_successors function for the CapsuleSearchProblem goes here.
        Args:
          state: a tuple combineing all the state information required
        Return:
          the states accessable by expanding the state provide
        """
        # - game_state.data.timeleft records the total number of turns left in the game (each time a player nodes turn decreases, so should decriment by 4)
        # - capture.CaptureRule.process handles actually ending the game, using 'game' and 'game_state' object
        # As these rules are not capture (enforced) within out game_state object, we need capture it outselves
        # - Option 1: track time left explictly
        # - Option 2: when updating the game_state, add additional information that generateSuccesor doesn't collect
        #           e.g set game_state.data._win to true. If goal then check game_state.isOver() is not true

        game_state = state[0]

        actions = game_state.get_legal_actions(self.my_agent.index)

        successors
        for enemy in self.my_agent.get_opponents(game_state):


        successors = [((game_state.generate_successor(self.captureAgent.index, action),
                    0, game_state), action, 1) for action in actions]

        return successors

################# Search Algorithems ###################




def aStarSearch(problem, heuristic=None, agent = None):
    """Search the node that has the lowest combined cost and heuristic first."""
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []
    #MAX_DEPTH = 25
    # use a priority queue to store states
    frontier = util.PriorityQueue()

    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(), [], 0)

    game_state = problem.getStartState()[0]
    distributions = [util.Counter()]
    lowest_value = 0
    # as a priority there will be the total cost + heuristic
    # as first node this is irrelevant
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        (node, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        if problem.isGoalState(node):# or len(path) > MAX_DEPTH:
            # Last heuristic
            heur = heuristic(node, problem)
            my_pos = node[0].get_agent_position(agent.index)
            distributions[0][my_pos] = heur
            
            distributions[0].incrementAll(distributions[0].keys(), -min(distributions[0].values()))
            max_value = max(distributions[0].values())
            if max_value == 0:  max_value += 1
            distributions[0].divideAll(max(distributions[0].values()))
            agent.display_distributions_over_positions(distributions)
            return path

        # the algorithm control if the node is being expanded before
        if node not in expanded_nodes:
            expanded_nodes.append(node)

            # if not the algorithm search in his successor and insert them in the frontier to be expanded
            for (child, n_action, n_cost) in problem.get_successors(node, path):
                if child not in expanded_nodes:
                    # fut_cost must be passed and represent the cost to reach that position
                    fut_cost = cost + n_cost

                    # total cost is the fut cost + heuristic and is passed as the priority
                    heur = heuristic(child, problem)
                    total_cost = cost + n_cost + heur
                    total_path = path + [n_action]

                    my_pos = child[0].get_agent_position(agent.index)
                    distributions[0][my_pos] = heur
                    
                    frontier.push((child, total_path, fut_cost), total_cost)



########################## DEFENCEEEE ##################

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.prev_food = None
        self.safe_boundary = None
        self.gridHeight = None
        self.boundary = None
        self.gridWidth = None
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.gridWidth = self.get_food(game_state).width
        self.gridHeight = self.get_food(game_state).height
        if self.red:
            self.boundary = int(self.gridWidth / 2) - 1
            self.safe_boundary = int(self.gridWidth / 2) - 1
        else:
            self.boundary = int(self.gridWidth / 2)
            self.safe_boundary = int(self.gridWidth / 2)
        self.prev_food = len(self.get_food(game_state).as_list())



    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        for action in actions:
            successor = self.get_successor(game_state, action)
            if successor.get_agent_position(self.index) == self.start and len(actions) > 1:
                actions.remove(action)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        food_list = self.get_food_you_are_defending(successor).as_list()
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = mean([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        min_distance_to_bound = 9999
        for i in range(1, self.gridHeight - 1):
            next_pos = (self.boundary, i)
            if not game_state.has_wall(next_pos[0], next_pos[1]):
                min_distance_to_bound = min(min_distance_to_bound, self.get_maze_distance(my_pos, next_pos))
        features['distance_to_boundary'] = min_distance_to_bound

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -20, 'stop': -100, 'reverse': -2,
                'distance_to_food': -0.1, 'distance_to_boundary': -0.01}
