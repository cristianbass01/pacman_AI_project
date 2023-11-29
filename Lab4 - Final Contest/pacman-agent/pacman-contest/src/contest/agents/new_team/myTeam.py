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
import util

from captureAgents import CaptureAgent
from game import Directions
from numpy import mean
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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
# Eaing a capsule when being chased by an enemy
# returning home once have enough food or being chased
# returning home when only two food left

# Penalty
# Closer to an enemy while offence
# closer to the teammate while offence
# reaching the corner that the agent might get trapped

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


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
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


        if not game_state.get_agent_state(self.index).is_pacman:
            self.prev_food = len(self.get_food(game_state).as_list())

        food_left = len(self.get_food(game_state).as_list())

        food_eated = self.prev_food - food_left

        if food_eated >= 1 or food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist != 0 and dist < best_dist and action != Directions.STOP:
                    best_action = action
                    best_dist = dist
            if best_action is None: best_action = Directions.STOP
            return best_action

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}



"""
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        my_pos = game_state.get_agent_position(self.index)

        # Compute distance to the nearest food
        next_pos = successor.get_agent_state(self.index).get_position()
        min_distance = min([self.get_maze_distance(next_pos, food) for food in food_list])
        features['distance_to_food'] = min_distance

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        attackers = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        features['num_attackers'] = len(attackers)
        min_distance_to_bound = 9999
        for i in range(1, self.gridHeight - 1):
            next_pos = (self.safe_boundary, i)
            if not game_state.has_wall(next_pos[0], next_pos[1]):
                min_distance_to_bound = min(min_distance_to_bound, self.get_maze_distance(my_pos, next_pos))

        if len(attackers) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in attackers]
            features['attackers_distance'] = 1/min(dists)
            features['distance_to_food'] = 0
            features['distance_to_boundary'] = 1/(min_distance_to_bound+1)
        else:
            features['distance_to_boundary'] = 0
            features['attackers_distance'] = 0

        my_food = self.prev_food -  len(food_list)
        if min_distance_to_bound < min_distance and game_state.get_agent_state(self.index).is_pacman and my_food > 0:
            features['distance_to_food'] = 0
            features['distance_to_boundary'] = 1/(min_distance_to_bound+1)

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_boundary': -4, 'num_attackers': -100, 'attackers_distance': -1000}

"""

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
