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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

NUM_TRAINING = 0
TRAINING = False


def create_team(firstIndex, secondIndex, isRed,
               first='ApproxQLearningOffense', second='DefensiveReflexAgent', numTraining=0):
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
    NUM_TRAINING = numTraining
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class ApproxQLearningOffense(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        self.epsilon = 0
        self.alpha = 0
        self.discount = 0
        self.numTraining = 0
        self.episodesSoFar = 0
        self.weights = {}
        self.start = None
        self.featuresExtractor = None
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state):
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.9
        self.numTraining = NUM_TRAINING
        self.episodesSoFar = 0

        self.weights = {'closest-food': -3.099192562140742,
                        'bias': -9.280875042529367,
                        '#-of-ghosts-1-step-away': -16.6612110039328,
                        'eats-food': 11.127808437648863}

        self.start = game_state.get_agent_position(self.index)
        self.featuresExtractor = FeaturesExtractor(self)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
    """
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None

        foodLeft = len(self.get_food(game_state).as_list())

        if foodLeft <= 2:
            bestDist = 9999
            for action in legalActions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        action = None
        if TRAINING:
            for action in legalActions:
                self.updateWeights(game_state, action)
        if not util.flipCoin(self.epsilon):
            # exploit
            action = self.getPolicy(game_state)
        else:
            # explore
            action = random.choice(legalActions)
        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, game_state, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        # features vector
        features = self.featuresExtractor.getFeatures(game_state, action)
        return features * self.weights

    def update(self, game_state, action, nextState, reward):
        """
       Should update your weights based on transition
    """
        features = self.featuresExtractor.getFeatures(game_state, action)
        oldValue = self.getQValue(game_state, action)
        futureQValue = self.getValue(nextState)
        difference = (reward + self.discount * futureQValue) - oldValue
        # for each feature i
        for feature in features:
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight
        # print(self.weights)

    def updateWeights(self, game_state, action):
        nextState = self.get_successor(game_state, action)
        reward = self.getReward(game_state, nextState)
        self.update(game_state, action, nextState, reward)

    def getReward(self, game_state, nextState):
        reward = 0
        agentPosition = game_state.get_agent_position(self.index)

        # check if I have updated the score
        if self.getScore(nextState) > self.getScore(game_state):
            diff = self.getScore(nextState) - self.getScore(game_state)
            reward = diff * 10

        # check if food eaten in nextState
        myFoods = self.get_food(game_state).as_list()
        distToFood = min([self.get_maze_distance(agentPosition, food) for food in myFoods])
        # I am 1 step away, will I be able to eat it?
        if distToFood == 1:
            nextFoods = self.get_food(nextState).as_list()
            if len(myFoods) - len(nextFoods) == 1:
                reward = 10

        # check if I am eaten
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        if len(ghosts) > 0:
            minDistGhost = min([self.get_maze_distance(agentPosition, g.get_position()) for g in ghosts])
            if minDistGhost == 1:
                nextPos = nextState.get_agent_state(self.index).get_position()
                if nextPos == self.start:
                    # I die in the next state
                    reward = -100

        return reward

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
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

    def computeValueFromQValues(self, game_state):
        """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        allowedActions = game_state.get_legal_actions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.getPolicy(game_state)
        return self.getQValue(game_state, bestAction)

    def computeActionFromQValues(self, game_state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None
        actionVals = {}
        bestQValue = float('-inf')
        for action in legalActions:
            targetQValue = self.getQValue(game_state, action)
            actionVals[action] = targetQValue
            if targetQValue > bestQValue:
                bestQValue = targetQValue
        bestActions = [k for k, v in actionVals.items() if v == bestQValue]
        # random tie-breaking
        return random.choice(bestActions)

    def getPolicy(self, game_state):
        return self.computeActionFromQValues(game_state)

    def getValue(self, game_state):
        return self.computeValueFromQValues(game_state)


class FeaturesExtractor:

    def __init__(self, agentInstance):
        self.agentInstance = agentInstance

    def getFeatures(self, game_state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.agentInstance.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.agentInstance.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None]
        # ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        agentPosition = game_state.get_agent_position(self.agentInstance.index)
        x, y = agentPosition
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        # if len(ghosts) > 0:
        #   minGhostDistance = min([self.agentInstance.get_maze_distance(agentPosition, g) for g in ghosts])
        #   if minGhostDistance < 3:
        #     features["minGhostDistance"] = minGhostDistance

        # successor = self.agentInstance.get_successor(game_state, action)
        # features['successorScore'] = self.agentInstance.getScore(successor)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # capsules = self.agentInstance.getCapsules(game_state)
        # if len(capsules) > 0:
        #   closestCap = min([self.agentInstance.get_maze_distance(agentPosition, cap) for cap in self.agentInstance.getCapsules(game_state)])
        #   features["closestCapsule"] = closestCap

        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        # print(features)
        return features

    def closestFood(self, pos, food, walls):
        """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.get_food(game_state).as_list())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

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
        features = self.getFeatures(game_state, action)
        weights = self.getWeights(game_state, action)
        return features * weights

    def getFeatures(self, game_state, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, game_state, action):
        """
    Normally, weights do not depend on the game_state.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

    def getFeatures(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        myState = successor.get_agent_state(self.index)
        myPos = myState.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.is_pacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(myPos, a.get_position()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, game_state, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
