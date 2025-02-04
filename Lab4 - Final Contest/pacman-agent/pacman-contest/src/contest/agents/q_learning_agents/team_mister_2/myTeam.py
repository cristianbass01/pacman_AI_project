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
               first='SmartOffense', second='HardDefense', numTraining=0):
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


class HardDefense(CaptureAgent):
    """
    A simple reflex agent that takes score-maximizing actions. It's given 
    features and weights that allow it to prioritize defensive actions over any other.
    """

    def register_initial_state(self, state):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        """

        CaptureAgent.register_initial_state(self, state)
        self.myAgents = CaptureAgent.get_team(self, state)
        self.opAgents = CaptureAgent.get_opponents(self, state)
        self.myFoods = CaptureAgent.get_food(self, state).as_list()
        self.opFoods = CaptureAgent.get_food_you_are_defending(self, state).as_list()

    # Finds the next successor which is a grid position (location tuple).
    def getSuccessor(self, state, action):
        successor = state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    # Returns a counter of features for the state
    def getFeatures(self, state, action):
        features = util.Counter()
        successor = self.getSuccessor(state, action)

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
        rev = Directions.REVERSE[state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    # Returns a dictionary of features for the state
    def getWeights(self, state, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    # Computes a linear combination of features and feature weights
    def evaluate(self, state, action):
        features = self.getFeatures(state, action)
        weights = self.getWeights(state, action)
        return features * weights

    # Choose the best action for the current agent to take
    def choose_action(self, state):
        agentPos = state.get_agent_position(self.index)
        actions = state.get_legal_actions(self.index)

        # Distances between agent and foods
        distToFood = []
        for food in self.myFoods:
            distToFood.append(self.distancer.getDistance(agentPos, food))

        # Distances between agent and opponents
        distToOps = []
        for opponent in self.opAgents:
            opPos = state.get_agent_position(opponent)
            if opPos != None:
                distToOps.append(self.distancer.getDistance(agentPos, opPos))

        # Get the best action based on values
        values = [self.evaluate(state, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)


class SmartOffense(CaptureAgent):
    """
    The offensive agent uses q-learing to learn an optimal offensive policy 
    over hundreds of games/training sessions. The policy changes this agent's 
    focus to offensive features such as collecting pellets/capsules, avoiding ghosts, 
    maximizing scores via eating pellets etc.
    """

    def __init__(self, index, time_for_computing=.1):
        self.epsilon = 0
        self.alpha = 0
        self.discountRate = 0
        self.weights = {}

        super().__init__(index, time_for_computing)

    def register_initial_state(self, state):
        CaptureAgent.register_initial_state(self, state)

        self.epsilon = 0.9  # exploration prob
        self.alpha = 0.4  # learning rate
        self.discountRate = 0.9
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION
        """
        try:
            with open('./weights.txt', "r") as file:
                self.weights = eval(file.read())
        except IOError:
            self.weights = {'closest-food': 1,
                        'bias': 1,
                        '#-of-ghosts-1-step-away': 1,
                        'successorScore': 1,
                        'eats-food': 1}
        

    # ------------------------------- Q-learning Functions -------------------------------

    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """

    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        return features * self.weights

    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """

    def getValue(self, state):
        qVals = []
        legalActions = state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qVals.append(self.getQValue(state, action))
            return max(qVals)

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """

    def getPolicy(self, state):
        values = []
        legalActions = state.get_legal_actions(self.index)
        legalActions.remove(Directions.STOP)
        if len(legalActions) == 0:
            return None
        else:
            for action in legalActions:
                self.updateWeights(state, action)
                values.append((self.getQValue(state, action), action))
        return max(values)[1]

    """
    Calculate probability of 0.1.
    If probability is < 0.1, then choose a random action from
    a list of legal actions.
    Otherwise use the policy defined above to get an action.
    """

    def choose_action(self, state):
        # Pick Action
        legalActions = state.get_legal_actions(self.index)
        action = None

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon)
            if prob:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    # ------------------------------ Features And Weights --------------------------------

    # Define features to use. NEEDS WORK
    def getFeatures(self, state, action):
        # Extract the grid of food and wall locations
        if self.red:
            food = state.get_red_food()
        else:
            food = state.get_blue_food()
            
        walls = state.get_walls()
        ghosts = []
        opAgents = CaptureAgent.get_opponents(self, state)
        # Get ghost locations and states if observable
        if opAgents:
            for opponent in opAgents:
                opPos = state.get_agent_position(opponent)
                opIsPacman = state.get_agent_state(opponent).is_pacman
                if opPos and not opIsPacman:
                    ghosts.append(opPos)

        # Initialize features
        features = util.Counter()
        successor = self.getSuccessor(state, action)

        # Successor Score
        features['successorScore'] = self.get_score(successor)

        # Bias
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.get_agent_position(self.index)
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Number of Ghosts scared
        # features['#-of-scared-ghosts'] = sum(state.get_agent_state(opponent).scaredTimer != 0 for opponent in opAgents)

        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # Normalize and return
        features.divideAll(10.0)
        return features

    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """

    def updateWeights(self, state, action):
        features = self.getFeatures(state, action)
        nextState = self.getSuccessor(state, action)

        # Calculate the reward. NEEDS WORK
        reward = nextState.get_score() - state.get_score()

        for feature in features:
            correction = (reward + self.discountRate * self.getValue(nextState)) - self.getQValue(state, action)
            self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]

    # -------------------------------- Helper Functions ----------------------------------

    # Finds the next successor which is a grid position (location tuple).
    def getSuccessor(self, state, action):
        successor = state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def closestFood(self, pos, food, walls):
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

    # Update weights file at the end of each game
    def final(self, state):
        file = open('./weights.txt', 'w')
        file.write(str(self.weights))