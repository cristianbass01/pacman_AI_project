# myTeam.py
# ---------
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
import pickle

from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from numpy import mean
from util import nearestPoint

#################
# Team creation #
#################

def create_team(firstIndex, secondIndex, is_red,
               first='QLearningOffense', second='DefensiveReflexAgent', numTraining=0):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class QLearningOffense(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        self.start = None
        self.q_values = None
        self.epsilon = 0
        self.alpha = 0
        self.discount = 0
        self.index = index
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state):
        self.epsilon = 0.2
        self.alpha = 0.2
        self.discount = 0.9
        self.start = game_state.get_agent_position(self.index)
        self.q_values = self.loadQValues()
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, state):
        """
        Picks among the actions with the highest Q(s,a).
    """
        legalActions = state.get_legal_actions(self.index)

        for action in legalActions:
            successor = self.get_successor(state, action)
            self.update(state, action, successor, self.getReward(state, successor))

        # flip coin is choose the action from the q value or randomly
        if util.flipCoin(self.epsilon):
            # choose the action randomly
            action = random.choice(legalActions)
        else:
            # choose the action from learning

            action = self.computeActionFromQValues(state)

        return action

    def getQValue(self, pos, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        # features vector
        return self.q_values[(pos, action)]

    def update(self, state, action, next_state, reward):
        """
        Should update your weights based on transition
        """
        pos = state.get_agent_position(self.index)

        new_Qvalue = (1 - self.alpha) * self.getQValue(pos, action)
        if len(next_state.get_legal_actions(self.index)) == 0:
            # if there are not possible actione the final part is 0 so erase it
            new_Qvalue += self.alpha * reward
        else:
            # if there is at least one possible action calculate the actual formula
            new_Qvalue += self.alpha * (reward + (self.discount * max(
                [self.getQValue(next_state.get_agent_position(self.index), next_action) for next_action in next_state.get_legal_actions(self.index)])))
        # save the new Q value
        self.q_values[(pos, action)] = new_Qvalue

    def getReward(self, state, next_state):
        agentPos = state.get_agent_position(self.index)
        nextPos = next_state.get_agent_position(self.index)
        reward = -self.get_maze_distance(agentPos, self.closest_food(next_state))

        # check if I have updated the score
        if self.get_score(next_state) > self.get_score(state):
            diff = self.get_score(next_state) - self.get_score(state)
            reward += diff * 10

        # check if food eaten in next_state
        myFoods = self.get_food(state).as_list()
        distToFood = min([self.get_maze_distance(agentPos, food) for food in myFoods])
        # I am 1 step away, will I be able to eat it?
        if distToFood == 1:
            nextFoods = self.get_food(next_state).as_list()
            if len(myFoods) - len(nextFoods) == 1:
                reward += 30

        # check if I am eaten
        enemies = [(state.get_agent_state(i),i) for i in self.get_opponents(state)]
        ghosts = [a for a,_ in enemies if not a.is_pacman and a.get_position() != None]
        if len(ghosts) > 0:
            minDistGhost = min([self.get_maze_distance(agentPos, g.get_position()) for g in ghosts])
            if minDistGhost == 1:
                if nextPos == self.start:
                    # I die in the next state
                    return -100

        """
        pacmans = [a for a in enemies if a[0].is_pacman and a[0].get_position() != None]
        if len(pacmans) > 0:
            closest_pacman = None
            dist = 9999
            for pacman in pacmans:
                if dist > self.get_maze_distance(agentPos, pacman[0].get_position()):
                    dist = self.get_maze_distance(agentPos,  pacman[0].get_position())
                    closest_pacman = pacman
            if nextPos == closest_pacman[0].get_position() and next_state.get_agent_position(closest_pacman[1]) == None:
                return 100
        """

        my_cap = self.get_capsules(state)
        if len(my_cap) > 0:
            dist_cap = min([self.get_maze_distance(agentPos, food) for food in my_cap])
            # I am 1 step away, will I be able to eat it?
            if dist_cap == 1:
                next_cap = self.get_food(next_state).as_list()
                if len(my_cap) - len(next_cap) == 1:
                    reward += 5

        return reward

    def closest_food(self, state):
        closest_food = None
        dist = 9999
        my_pos = state.get_agent_position(self.index)
        for food in self.get_food(state).as_list():
            if dist > self.get_maze_distance(my_pos, food):
                dist = self.get_maze_distance(my_pos, food)
                closest_food = food
        return closest_food

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        self.saveQValues()
        CaptureAgent.final(self, state)
        # print(self.weights)
        # did we finish training?

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

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** MY CODE HERE ***"
        q_values = []
        # calculate the q value for every possible action
        for action in self.get_legal_actions(state):
            q_values.append(self.getQValue(state.get_agent_position(self.index), action))
        # see if there is no possible action, and return 0.0 in that case
        if len(self.get_legal_actions(state)) == 0:
            return 0.0
        else:
            #if there is a possible action return the higher value
            return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # max action is none for terminal state
        max_action = None
        maxQvalue = 0
        for action in state.get_legal_actions(self.index):
            q_val = self.getQValue(state.get_agent_position(self.index), action)
            # initialize the q value or save it if the new one is higher than the previous
            if q_val > maxQvalue or max_action is None:
                maxQvalue = q_val
                # max action = action if the this q value is higher
                max_action = action
        return max_action

    def getPolicy(self, game_state):
        return self.computeActionFromQValues(game_state)

    def getValue(self, game_state):
        return self.computeValueFromQValues(game_state)

    def saveQValues(self):
        with open(".\\agents\\qvalues.pkl", "wb") as tf:
            pickle.dump(self.q_values, tf)

    def loadQValues(self):
        try:
            with open(".\\agents\\qvalues.pkl", "rb") as tf:
                return pickle.load(tf)
        except:
            return util.Counter()


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
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
        else:
            self.boundary = int(self.gridWidth / 2)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

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


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

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
        for i in range(1,self.gridHeight-1):
            next_pos = (self.boundary, i)
            if not game_state.has_wall(next_pos[0], next_pos[1]):
                min_distance_to_bound = min(min_distance_to_bound, self.get_maze_distance(my_pos, next_pos))
        features['distance_to_boundary'] = min_distance_to_bound

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -20, 'stop': -100, 'reverse': -2, 'distance_to_food': -0.1, 'distance_to_boundary': -0.01}
