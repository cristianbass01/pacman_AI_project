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
from contest.game import Directions, Actions, AgentState  # basically a class to store data
from contest.capture import GameState

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

        self.dead_ends = None
        self.nearest_exit_from_ends = None

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

        # Detect and store dead ends
        self.detect_dead_ends(game_state)

        #distr = [util.Counter()]
        #for pos in self.nearest_exit_from_ends.values():
        #    distr[0][pos] = 1
        #self.display_distributions_over_positions(distr)

    def detect_dead_ends(self, game_state):
        self.dead_ends = []
        self.nearest_exit_from_ends = util.Counter()

        def is_valid_cell(x, y):
            return not game_state.has_wall(x, y)

        for x in range(self.gridWidth):
            for y in range(self.gridHeight):
                if is_valid_cell(x, y):
                    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    num_open_neighbors = sum(
                        is_valid_cell(nx, ny) for nx, ny in neighbors
                    )

                    if num_open_neighbors == 1:
                        opening_x, opening_y = next((nx, ny) for nx, ny in neighbors if is_valid_cell(nx, ny))

                        self.dead_ends.append((x, y))
                        new_dead_ends = [(x,y)]
                        while num_open_neighbors <= 2:
                            neighbor_neighbors = [
                                (opening_x + 1, opening_y),
                                (opening_x - 1, opening_y),
                                (opening_x, opening_y + 1),
                                (opening_x, opening_y - 1)
                            ]
                            num_open_neighbors = sum(
                                is_valid_cell(nx, ny) for nx, ny in neighbor_neighbors
                            )

                            if num_open_neighbors == 2:
                                self.dead_ends.append((opening_x, opening_y))
                                new_dead_ends.append((opening_x, opening_y))
                                opening_x, opening_y = next((nx, ny) for nx, ny in neighbor_neighbors
                                                            if is_valid_cell(nx, ny) and (nx, ny) not in self.dead_ends)
                        
                        for new_dead_end in new_dead_ends:
                            self.nearest_exit_from_ends[new_dead_end] = (opening_x, opening_y)
                                
                        

class offensiveAgent(agentBase):

    def choose_action(self, game_state):
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action
        
        # General
        agent_state = game_state.get_agent_state(self.index)
        start = self.start == game_state.get_agent_position(self.index)
        no_moves_left = self.next_moves == None or (self.next_moves != None and len(self.next_moves) == 0)
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        # Enemy
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        ghost_distances = [self.get_maze_distance(my_pos, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        
        are_scared = [a.scared_timer > 8 for a in enemies if not a.is_pacman and a.get_position() != None]
        exist_threat = thereIsAThreat(game_state, self)
        prev_game_state = self.get_previous_observation()
        if prev_game_state != None:
            used_to_be_threat = thereIsAThreat(prev_game_state, self)
        else: 
            used_to_be_threat = False
        

        # Food
        current_food = len(self.get_food(game_state).as_list())
        food_changed = self.prev_food > current_food
        is_carrying_food = agent_state.num_carrying > 0
        
        # Capsule
        capsules = self.get_capsules(game_state)
        capsule_distances = [self.get_maze_distance(my_pos, cap) for cap in capsules]
        min_capsule_distance = min(capsule_distances)

        problem = FoodOffense(startingGameState=game_state, captureAgent=self)
        





        return next_action
    

################# Heuristics ###################################

    def heuristicToPos(self, data)
        game_state = data['game']
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        num_carrying = gameState.get_agent_state(self.index).num_carrying
        ghost_distances = [self.get_maze_distance(start, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        are_scared = [enemy.scared_timer > 5 for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        min_ghost_distance = 100
        if len(ghost_distances) > 0:
            min_ghost_distance = min(ghost_distances)
            min_ghost_distance_index = ghost_distances.index(min_ghost_distance)
            is_scared = are_scared[min_ghost_distance_index]
            if is_scared:
                min_ghost_distance = 0
        return self.get_maze_distance(data['pos'], self.target) + min_ghost_distance * num_carrying


    
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
        self.index = captureAgent.index
        self.startingGameState = startingGameState
        # Need to ignore previous score change, as everything should be considered relative to this state
        self.startingGameState.data.score_change = 0
        self.target = self.startingGameState.get_agent_position(self.index)
        

    def getStartState(self):
        # This needs to return the state information to being with

        return self.startingGameState.get_agent_position(self.index)
        
    def getStartData(self):
        # This needs to return the state information to being with
        data = util.Counter()
        data['prev_game'] = None
        data['game'] = self.startingGameState
        data['action'] = None
        data['cost'] = 0
        data['pos'] = self.startingGameState.get_agent_position(self.index)
        data['agent'] = self.captureAgent

        return data

    def isGoalStatePosition(self, data):
        if data['pos'] == self.target:
            return True
        return False

    def get_successors(self, old_data):
        """
        Your get_successors function for the CapsuleSearchProblem goes here.
        Args:
          state: a tuple combining all the state information required
        Return:
          the states accessable by expanding the state provide
        """
        # - game_state.data.timeleft records the total number of turns left in the game 
        #   (each time a player nodes turn decreases, so should decriment by 4)
        # - capture.CaptureRule.process handles actually ending the game, using 'game' 
        #   and 'game_state' object
        # As these rules are not capture (enforced) within out game_state object, we need
        # to capture it outselves
        # - Option 1: track time left explictly
        # - Option 2: when updating the game_state, add additional information that 
        #             generateSuccesor doesn't collect e.g set game_state.data._win to true.
        #             If goal then check game_state.isOver() is not true
        game_state = old_data['game']
        current_pos = old_data['pos']
        my_agent = old_data['agent']

        my_actions = game_state.get_legal_actions(self.index)

        successors = []

        enemy_positions = [game_state.get_agent_position(enemy) for enemy in my_agent.get_opponents(game_state)]
        
        for next_action in my_actions:
            fut_pos = Actions.get_successor(current_pos, next_action)
            
            if not fut_pos in enemy_positions:
                data = util.Counter()
                data['prev_game'] = game_state
                data['game'] = game_state.generate_successor(self.index, action)
                data['action'] = next_action
                data['cost'] = 1
                data['pos'] = fut_pos
                data['agent'] = self.captureAgent
                successors.append(data)

        return successors

################# Search Algorithems ###################




def aStarSearch(problem, heuristic=None, goal = None):
    """Search the node that has the lowest combined cost and heuristic first."""
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = set()
    min_cost = util.Counter()
    min_cost[problem.getStartState()] = 0
    #MAX_DEPTH = 25
    # use a priority queue to store states
    frontier = util.PriorityQueue()

    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(), problem.getStartData(),  [], 0)

    distributions = [util.Counter()]
    
    # as a priority there will be the total cost + heuristic
    # as first node this is irrelevant
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        (node, data, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        if goal(data):
            updateDistributions(distributions, data, heuristic)
            return path

        # if not the algorithm search in his successor and insert them in the frontier to be expanded
        for new_data in problem.get_successors(data):
            fut_cost = cost + new_data['cost']
            new_pos = new_data['pos']

            if new_pos not in expanded_nodes or new_cost < min_cost[new_pos]:
                min_cost[new_pos] = fut_cost
                expanded_nodes.add(new_pos)
                
                # the priority is the fut cost + heuristic
                priority = fut_cost + heuristic(new_data)
                
                next_action = new_data['action']
                total_path = path + [next_action]
                
                updateDistributions(distributions, data, heuristic, show = False)
                    
                frontier.push((new_pos, new_data, total_path, fut_cost), priority)

def showAndNormalize(distributions, agent):
    distributions[0].incrementAll(distributions[0].keys(), -min(distributions[0].values()))
    distributions[0].normalize()
    agent.display_distributions_over_positions(distributions)

def updateDistributions(distributions, data, heuristic, show = True):
    distributions[0][data['pos']] = heuristic(data)
    if show:
        showAndNormalize([distributions[0].copy()], data['agent'])

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
