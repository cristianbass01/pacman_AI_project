# Mister-Pacman Team.py
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


# Mister-Pacman Team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from numpy import mean
from contest.util import nearestPoint
from contest.distanceCalculator import Distancer
from contest.capture import GameState

MAX_DEPTH = 20

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

# Rewards:
# Eating food
# Eating a capsule when being chased by an enemy
# returning home once have enough food or being chased
# returning home when only two food left

# Penalty
# Closer to an enemy while offence
# closer to the teammate while offence
# reaching the corner that the agent might get trapped

class AStarAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.index = index
        self.prev_food = None
        self.safe_boundary = None
        self.gridHeight = None
        self.gridWidth = None
        self.start = None
        self.safe_pos = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)

        CaptureAgent.register_initial_state(self, game_state)

        self.gridWidth = self.get_food(game_state).width
        self.gridHeight = self.get_food(game_state).height
        if self.red:
            self.safe_boundary = int(self.gridWidth / 2) - 1
        else:
            self.safe_boundary = int(self.gridWidth / 2)

        self.prev_food = len(self.get_food(game_state).as_list())

        self.safe_pos = []
        for y in range(self.gridHeight):
            if not game_state.has_wall(self.safe_boundary, y):
                self.safe_pos += (self.safe_boundary, y)
    
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

class OffensiveAgent(AStarAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def choose_action(self, game_state):
        """
                Picks among the actions with the highest Q(s,a).
        """

        return aStarSearch(game_state, self.heuristic)
    
    def heuristic(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        prev_game_state = self.get_previous_observation()

        difend_food_pos = self.get_food(game_state).as_list()
        offens_food_pos = self.get_food_you_are_defending(game_state).as_list()

        prev_difend_food_pos = self.get_food(prev_game_state).as_list()
        prev_offens_food_pos = self.get_food_you_are_defending(prev_game_state).as_list()

        difend_capsule_pos = self.get_capsules(game_state)
        offens_capsule_pos = self.get_capsules_you_are_defending(game_state)

        teammate_positions = [game_state.get_agent_position(agent) for agent in self.get_team(game_state) if agent != self.index]
        enemy_positions = [game_state.get_agent_position(agent) for agent in self.get_opponents(game_state)]

        current_score = self.get_score(game_state)
        
        #Usefull
        # self.is_pacman
        # self.scared_timer
        # self.num_carrying
        # self.num_returned
        ### TODO ADD DISTRIBUTIONS

        FOOD_CARRYING_MUL = 10
        FOOD_RETURNED_MUL = 10
        EATING_CAPSULE_MUL = 10
        RETURN_HOME_ALL_FOOD_MUL = 100
        RETURN_HOME_CHASED_MUL = 10
        RETURN_HOME_ENOUGH_FOOD_MUL = 10
        ENEMY_PENALTY_MUL = 10
        TEAMMATE_PENALTY_MUL = 10

        # Reward for eating food
        food_reward = FOOD_CARRYING_MUL * self.num_carryng

        # Reward for eating a capsule when being chased
        if game_state.get_agent_state(self.index).is_pacman:
            for agent in self.get_opponents(game_state):
                if game_state.get_agent_state(agent).is_ghost and not game_state.get_agent_state(agent).scared_timer <= self.get_maze_distance(my_pos, game_state.get_agent_position(agent)) :
                    for capsule in offens_capsule_pos:  
                        if self.get_maze_distance(my_pos, capsule) < self.get_maze_distance(my_pos, game_state.get_agent_position(agent)):
                            food_reward += EATING_CAPSULE_MUL * self.get_maze_distance(my_pos, capsule)  # Adjust the reward value as needed
                
        # Reward for returning food
        food_reward += self.num_returned * FOOD_RETURNED_MUL

        # Reward for returning home once having all the food for win
        if len(offens_food_pos) <= 2:
            home_reward = -min([self.get_maze_distance(my_pos, border_pos) for border_pos in self.safe_pos]) * RETURN_HOME_ALL_FOOD_MUL
        else:
            home_reward = 0

        # Reward for returning home once being chased
        if any(game_state.get_agent_state(agent).is_ghost and not game_state.get_agent_state(agent).is_scared for agent in self.get_opponents(game_state)):
            home_reward -= min([self.get_maze_distance(my_pos, border_pos) for border_pos in self.safe_pos]) * RETURN_HOME_CHASED_MUL
        else:
            home_reward = 0

        # Penalty for being closer to a teammate while on offense
        teammate_penalty = sum(self.get_maze_distance(my_pos, teammate) for teammate in teammate_positions) * TEAMMATE_PENALTY_MUL

        # Combine rewards and penalties to form the heuristic
        heuristic = food_reward + home_reward - teammate_penalty

        return heuristic

class DefensiveAgent(AStarAgent):
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
        
        return aStarSearch(game_state, self.heuristic)
    
    def heuristic(game_state):

        return 0

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(agent, game_state, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []

    # use a priority queue to store states
    frontier = util.PriorityQueue()

    # A node in the stack is composed by :
    # his game_state
    # his path from the initial state
    # his total cost to reach that position
    start_node = (game_state, [], 0)

    # as a priority there will be the total cost + heuristic
    # as first node this is irrelevant
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        (suc_game, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        # goal state depends on the problem
        if isGoal(suc_game):
            return path

        # the algorithm control if the node is being expanded before
        if suc_game not in expanded_nodes:
            expanded_nodes.append(suc_game)

            # if not the algorithm search in his successor and insert them in the frontier to be expanded
            for n_action in suc_game.get_legal_actions(agent.index):
                child = agent.get_successor(suc_game, n_action)
                n_cost = 1
                if child not in expanded_nodes:
                    # fut_cost must be passed and represent the cost to reach that position
                    fut_cost = cost + n_cost

                    #total cost is the fut cost + heuristic and is passed as the priority
                    total_cost = cost + n_cost + heuristic(child)
                    total_path = path + [n_action]
                    frontier.push((child, total_path, fut_cost), total_cost)

def isGoal(game_state):
    return False