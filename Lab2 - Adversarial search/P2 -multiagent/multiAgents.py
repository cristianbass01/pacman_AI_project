# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** MY CODE HERE ***"
        penalty = 0
        # calculate distance from pacman to foods
        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFood.asList()]
        if len(foodDistances) > 0:
            # calculate the closest food dot
            closestFood = min(foodDistances)
        else:
            # if no food assume infinite (we can't leave it 0 because of the reciproc)
            closestFood = float('inf')

        # calculate distance from ghost
        ghostsDistances = [manhattanDistance(newPos, ghostPosition) for ghostPosition in successorGameState.getGhostPositions()]
        if len(ghostsDistances) > 0:
            # find the closest ghost
            closestGhost = min(ghostsDistances)
            # if the ghost is closed add a small penalty
            if closestGhost < 2.5:
                penalty -= 4
            # if the ghost is really closed add a bigger penalty
            if closestGhost < 1.5:
                penalty -= 30

        # return the actual score + reciproc of the closest food + the penalty
        return successorGameState.getScore() + 1.0 / closestFood + penalty




def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** MY CODE HERE ***"
        result = self.getVal(gameState, 0, 0)
        return result[1]

    def getVal(self, gameState, agentIndex, depth):
        # base case (finish state if depth is been reached or if there's no more actions)
        if depth == self.depth or len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState), ""

        # we see if agent is pacman or a ghost (pacman = 0, ghosts > 0)
        if agentIndex == 0:
            # if the agent is pacman, we search for the max value of the ghosts
            return self.maxVal(gameState, agentIndex, depth)
        else:
            # if the agent is a ghost we
            return self.minVal(gameState, agentIndex, depth)

    def maxVal(self, gameState, agentIndex, depth):
        # takes legal moves
        legalMoves = gameState.getLegalActions(agentIndex)
        max_value = float("-inf")
        max_action = ""

        # for every legal move we evaluate the successor
        for action in legalMoves:
            # generate successor gamestate
            successor_game = gameState.generateSuccessor(agentIndex, action)

            # nextAgent is the number of the agent in the following node in the tree that has the same father
            nextAgent = (agentIndex + 1)%gameState.getNumAgents()

            # if the successor is pacman then we are searching in the following depth in the tree
            new_depth = depth if nextAgent > 0 else depth + 1

            # evaluate the successor
            current_value = self.getVal(successor_game, nextAgent, new_depth)[0]

            # see if current_value is the current max value
            if current_value > max_value:
                max_value = current_value

                # store action iff this is the current max node
                max_action = action

        return max_value, max_action

    def minVal(self, gameState, agentIndex, depth):
        legalMoves = gameState.getLegalActions(agentIndex)
        min_value = float("inf")
        min_action = ""

        # for every legal move we evaluate the successor
        for action in legalMoves:
            # generate successor gamestate
            successor_game = gameState.generateSuccessor(agentIndex, action)

            # nextAgent is the number of the agent in the following node in the tree that has the same father
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()

            # if the successor is pacman then we are searching in the following depth in the tree
            new_depth = depth if nextAgent > 0 else depth + 1

            # evaluate the successor
            current_value = self.getVal(successor_game, nextAgent, new_depth)[0]

            # see if current_value is the current min value
            if current_value < min_value:
                min_value = current_value

                # store action iff this is the current max node
                min_action = action

        return min_value, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** MY CODE HERE ***"
        result = self.getVal(gameState, 0, 0, float('-inf'), float('inf'))
        return result[1]

    def getVal(self, gameState, agentIndex, depth, alpha, beta):
        # base case (finish state if depth is been reached or if there's no more actions)
        if depth == self.depth or len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState), ""

        # we see if agent is pacman or a ghost (pacman = 0, ghosts > 0)
        if agentIndex == 0:
            # if the agent is pacman, we search for the max value of the ghosts
            return self.maxVal(gameState, agentIndex, depth, alpha, beta)
        else:
            # if the agent is a ghost we
            return self.minVal(gameState, agentIndex, depth, alpha, beta)

    def maxVal(self, gameState, agentIndex, depth, alpha, beta):
        # takes legal moves
        legalMoves = gameState.getLegalActions(agentIndex)
        max_value = float("-inf")
        max_action = ""

        # for every legal move we evaluate the successor
        for action in legalMoves:
            # generate successor gamestate
            successor_game = gameState.generateSuccessor(agentIndex, action)

            # nextAgent is the number of the agent in the following node in the tree that has the same father
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()

            # if the successor is pacman then we are searching in the following depth in the tree
            new_depth = depth if nextAgent > 0 else depth + 1

            # evaluate the successor
            current_value = self.getVal(successor_game, nextAgent, new_depth, alpha, beta)[0]

            # see if current_value is the current max value
            if current_value > max_value:
                max_value = current_value

                # store action iff this is the current max node
                max_action = action

            if max_value > beta:
                return max_value, action

            # make alpha be the highest value
            alpha = max(alpha, max_value)

        return max_value, max_action

    def minVal(self, gameState, agentIndex, depth, alpha, beta):
        legalMoves = gameState.getLegalActions(agentIndex)
        min_value = float("inf")
        min_action = ""

        # for every legal move we evaluate the successor
        for action in legalMoves:
            # generate successor gamestate
            successor_game = gameState.generateSuccessor(agentIndex, action)

            # nextAgent is the number of the agent in the following node in the tree that has the same father
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()

            # if the successor is pacman then we are searching in the following depth in the tree
            new_depth = depth if nextAgent > 0 else depth + 1

            # evaluate the successor
            current_value = self.getVal(successor_game, nextAgent, new_depth, alpha, beta)[0]

            # see if current_value is the current min value
            if current_value < min_value:
                min_value = current_value

                # store action iff this is the current max node
                min_action = action

            # see if current min value is lower than alpha and return it without search in other successor
            if min_value < alpha:
                return min_value, action

            # make beta be the lowest value
            beta = min(beta, min_value)

        return min_value, min_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
