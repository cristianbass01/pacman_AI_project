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

# Standard imports
from contest.captureAgents import CaptureAgent
import random
from contest.game import Directions, Actions  # basically a class to store data


# this is the entry points to instanciate you agents
def create_team(first_index, second_index, is_red,
                first='offensiveAgent', second='defensiveAgent', num_training=0):

    # capture agents must be instanciated with an index
    # time to compute id 1 second in real game
    return [eval(first)(first_index), eval(second)(second_index)]


class agentBase(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # start position of our agent
        self.start = None
        # Agent identificator
        self.index = index
        # Accumulated food
        self.prev_food = None
        # Position on our side adjacent to the boundry.
        # The positions after which you cross the boundry
        self.safe_pos = None
        # The y of the safe positions
        self.safe_boundary = None
        # Map size
        self.gridHeight = None
        self.gridWidth = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        # the following initialises self.red and self.distancer
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Required.
        This is called each turn to get an agent to choose and action
        Return:
        This find the directions by going through game_state.getLegalAction
        - don't try and generate this by manually
        """
        actions = game_state.get_legal_actions(self.index)

        return random.choice(actions)


class offensiveAgent(agentBase):

    def choose_action(self, game_state):
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action

        problem = FoodOffenseWithAgentAwareness(startingGameState=game_state, captureAgent=self)


        actions = aStarSearch(problem, heuristic=offensiveHeuristic)
        # this can occure if start in the goal state. In this case do not want to 
        # perform any action.
        if actions == []:
            actions = ["Stop"]

        return actions[0]


#################  problems and heuristics  ####################



class FoodOffenseWithAgentAwareness():
    '''
    This problem extends FoodOffense by updateing the enemy ghost to move to 
    our pacman if they are adjacent (basic Goal Recognition techniques).
    This conveys to our pacman the likely effect of moving next to an enemy
    ghost - but doesn't prohibit it from doing so (e.g if Pacman has been trapped)
    Note: This is a SearchProblem class. It could inherit from search.Search problem
    (mainly for conceptual clarity).
    '''

    def __init__(self, startingGameState, captureAgent):
        # tracks number of states expanded. Is not really used in the code
        # It is just a counter
        self.expanded = 0
        self.startingGameState = startingGameState
        # Need to ignore previous score change, as everything should be 
        # considered relative to this state
        self.startingGameState.data.score_change = 0
        self.MINIMUM_IMPROVEMENT = 1
        self.DEPTH_CUTOFF = 20
        # WARNING: Capture agent doesn't update with new state, this should 
        # only be used for non state dependant utils (e.g distancer)
        self.captureAgent = captureAgent
        self.goal_state_found = None

    def getStartState(self):
        # This needs to return the state information to being with
        return (self.startingGameState, self.startingGameState.get_score())

    def isGoalState(self, state):
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        # Goal state when:
        # - Pacman is in our territory
        # - has eaten x food: This comes from the score changing
        # these are both captured by the score changing by a certain amount

        # Note: can't use CaptureAgent, as it doesn't update with game state
        game_state = state[0]

        # If red team, want scores to go up
        if self.captureAgent.red == True:
            if game_state.data.score_change >= self.MINIMUM_IMPROVEMENT:
                self.goal_state_found = state
                return True
            else:
                return False
        # If blue team, want scores to go down
        else:
            if game_state.data.score_change <= -self.MINIMUM_IMPROVEMENT:
                self.goal_state_found = state
                return True
            else:
                return False

    def isPacman(self, idx, game_state):
        return game_state.get_agent_state(
                idx).is_pacman
    
    def legalDepth(self, path):
        return len(path) <= self.DEPTH_CUTOFF

    def enemyVisible(self, enemy_idx, gameState):
        return gameState.get_agent_position(enemy_idx) is not None

    def enemyAdjacent(self, agentPos, enemyPos, game_state):
        return self.captureAgent.get_maze_distance(agentPos,
                         game_state.get_agent_state(enemyPos).get_position()) <= 1
    
    def enemyDangerous(self, enemyPos, game_state):
        enemyNotScared = game_state.get_agent_state(enemyPos).scared_timer <= 0
        return (not self.isPacman(enemyPos, game_state)) and enemyNotScared
    
    def getAdjacentGhosts(self, game_state):
        """Returns indexes of enemies adjacent to our pacman who are ghosts"""
        agentIndex = self.captureAgent.index
        # get enemies
        enemy_indexes = self.captureAgent.get_opponents(game_state)

        # keep only enemies that are close enough to catch pacman.
        close_enemy_indexes = [enemy_idx for enemy_idx in enemy_indexes 
                                if self.enemyVisible(enemy_idx, game_state)]
                
        agentPos = game_state.get_agent_state(agentIndex).get_position()
                
        adjacent_enemy_indexs = list(filter(
            lambda x: self.enemyAdjacent(
                agentPos, x, game_state), close_enemy_indexes
            )
        )

        # check in enemies are in the right state
        adjacent_ghost_indexs = list(filter(
             lambda x: self.enemyDangerous(x, game_state), adjacent_enemy_indexs))
        
        return adjacent_ghost_indexs
    
    def getKillActions(self, adjacentGhosts, game_state, next_game_state):
        """Returns the actions that if taken will result in our pacman agent 
           dying in the next state"""
        agentIndex = self.captureAgent.index
        agentPosition = game_state.get_agent_state(agentIndex).get_position()

        ghostKillActions = []
        for idx in adjacentGhosts:
            ghostPosition = next_game_state.get_agent_state(idx).get_position()

            for action in game_state.get_legal_actions(self.captureAgent.index):
                newGhostPosition = Actions.get_successor(ghostPosition, action)
                if newGhostPosition == agentPosition:
                    ghostKillActions.append(action)
                    break

        return ghostKillActions

    def get_successors(self, state, path = None):
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
        game_state = state[0]

        actions = game_state.get_legal_actions(self.captureAgent.index)
        # not interested in exploring the stop action as the state will be the same as our
        # current one.
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        next_game_states = [game_state.generate_successor(
                                self.captureAgent.index, action)
                            for action in actions]

        ## (confusing comment | Niki)
        # if planning close to agent, include expected ghost activity
        # current_depth_of_search = len(path)

        agentIndex = self.captureAgent.index

        # we are only concerned about being eaten when we are pacman
        if self.legalDepth(path) and self.isPacman(agentIndex, game_state):
            self.expanded += 1  # track number of states expanded

            # If in the next state you are next to a ghost assume it will eat you
            # if legal
            for i, next_game_state in enumerate(next_game_states):
                adjacentGhosts = self.getAdjacentGhosts(next_game_state)
                ghostKillActions = self.getKillActions(adjacentGhosts,
                                                game_state, next_game_state)

                # update state assuming that if for a ghost there is an action
                # that kills our pacman agent in the next state it takes it:
                for enemyIndex, direction in zip(adjacentGhosts, ghostKillActions):
                    self.expanded += 1
                    next_game_state[i] = next_game_state.generate_successor(
                        enemyIndex, direction)

        # Why do we do (next_game_state,) and not just (next_game_state)?
        successors = [((next_game_state,), action, 1)
                      for action, next_game_state in zip(actions, next_game_states)]

        return successors

def goalStateReached(captureAgent, game_state, problem):
    # check if we have reached a goal state and explicitly return 0
    if captureAgent.red:
        if game_state.data.score_change >= problem.MINIMUM_IMPROVEMENT:
            return True
    # If blue team, want scores to go down
    else:
        if game_state.data.score_change <= - problem.MINIMUM_IMPROVEMENT:
            return True

def getClosestFood(agentPosition, captureAgent, game_state):
    """Return minimum distance to food and its position"""
    food_list = captureAgent.get_food(game_state).as_list()

    min_pos = None
    min_dist = 99999999
    for food in food_list:
        dist = captureAgent.get_maze_distance(agentPosition, food)
        if dist < min_dist:
            min_pos = food
            min_dist = dist
    
    return min_dist, min_pos

def getMinReturnHomeDistanceFrom(pos, captureAgent, game_state):
    # Returning Home
    # WARNING: this assumes the maps are always symmetrical, territory is
    # divided in half, red on right, blue on left
    # Why list??
    walls = list(game_state.get_walls())
    # Shouldn't row_len and col_len be the opposite??????
    # I changed the names based on how it is used
    # but doesn't seem correct TODO: Check this out WTF
    row_len = len(walls[0])
    column_len = len(walls)
    mid_point_index = int(column_len/2)
    if captureAgent.red:
        mid_point_index -= 1

    # find all the entries and find distance to closest
    entry_coords = []
    for i, row in enumerate(walls[mid_point_index]):
        if row is False:  # there is not a wall
            entry_coords.append((int(mid_point_index), int(i)))

    return min([captureAgent.get_maze_distance(pos, entry)
                    for entry in entry_coords])

def offensiveHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. Thanks Mr Obvious  
    """
    captureAgent = problem.captureAgent
    agentIndex = captureAgent.index
    game_state = state[0]

    # If goal is reached we do nothing I think???
    if goalStateReached(captureAgent, game_state, problem):
        return 0

    agent_state = game_state.get_agent_state(agentIndex)
    food_carrying = agent_state.num_carrying
    agentPosition = agent_state.get_position()

    # this will be updated to be closest food location if not collect enough food
    return_home_from = agentPosition

    # still need to collect food
    dist_to_food = 0
    if food_carrying < problem.MINIMUM_IMPROVEMENT:
        # Why do we need return_home_from to be updated
        # Isn't this update just lost
        dist_to_food, return_home_from = getClosestFood(agentPosition,
                                                         captureAgent, game_state)
        # I feel this return should be removed
        # So the heuristic becomes dist_to_food + distance to go back
        return dist_to_food

    minDistance = getMinReturnHomeDistanceFrom(return_home_from,
                                                captureAgent, game_state)
    return dist_to_food + minDistance


################# Defensive problems and heuristics  ####################


class defensiveAgent(agentBase):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.prevMissingFoodLocation = None
        self.enemyEntered = False
        self.boundaryGoalPosition = None

    def choose_action(self, game_state):

        problem = defendTerritoryProblem(startingGameState=game_state, captureAgent=self)

        # actions = search.breadthFirstSearch(problem)
        # actions = aStarSearch(problem, heuristic=defensiveHeuristic)
        actions = aStarSearch(problem, heuristic=defensiveHeuristic)

        if actions != None:
            return actions[0]
        else:
            return random.choice(game_state.get_legal_actions(self.index))



class defendTerritoryProblem():
    def __init__(self, startingGameState, captureAgent):
        self.expanded = 0
        self.startingGameState = startingGameState
        self.captureAgent = captureAgent
        self.enemies = self.captureAgent.get_opponents(startingGameState)
        self.walls = startingGameState.get_walls()
        self.intialPosition = self.startingGameState.get_agent_position(
            self.captureAgent.index)
        self.gridWidth = self.captureAgent.get_food(startingGameState).width
        self.gridHeight = self.captureAgent.get_food(startingGameState).height
        if self.captureAgent.red:
            self.boundary = int(self.gridWidth / 2) - 1
        else:
            self.boundary = int(self.gridWidth / 2)
            
        self.myPreciousFood = self.captureAgent.get_food_you_are_defending(startingGameState)

        (self.viableBoundaryPositions,
         self.possibleEnemyEntryPositions) = self.getViableBoundaryPositions()

        self.GOAL_POSITION = self.getGoalPosition()
        self.goalDistance = self.captureAgent.get_maze_distance(self.GOAL_POSITION, self.intialPosition)

    def boundaryIsViable(self, boundryWidth, nextOffset, boundryHeight):
        """Checks if a boundry can be used to enter our territory. In other words
         there are no walls stopping an enemy pacman from entering or exiting """
        positionNotAWall = not(self.walls[boundryWidth][boundryHeight]) 
        adjacentPositionNotAWall = not(self.walls[boundryWidth + nextOffset][boundryHeight])
        return positionNotAWall and adjacentPositionNotAWall

    def getViableBoundaryPositions(self):
        myPos = self.startingGameState.get_agent_position(self.captureAgent.index)
        boundary = self.boundary
        boundaryPositions = []
        enemyEntryPositions = []

        for boundaryHeight in range(0, self.gridHeight):
            isRed = self.captureAgent.red
            # for red is +1 and for blue -1
            # We want to get previous or next cell depending on the team
            nextBoundryWidthOffset = isRed - (not isRed)
            if self.boundaryIsViable(boundary, nextBoundryWidthOffset,
                                boundaryHeight):
                if (boundary, boundaryHeight) != myPos:
                    boundaryPositions.append((boundary, boundaryHeight))
                enemyEntryPositions.append((boundary+nextBoundryWidthOffset,
                                        boundaryHeight))

        return (boundaryPositions, enemyEntryPositions)

    def getGoalPosition(self):
        isPacman = self.startingGameState.get_agent_state(
            self.captureAgent.index).is_pacman

        isScared = self.startingGameState.get_agent_state(
            self.captureAgent.index).scared_timer > 0

        if isScared:
            boundaryGoalPositions = self.closestPosition(
                self.intialPosition, self.viableBoundaryPositions)
            if self.captureAgent.boundaryGoalPosition == None:
                boundaryGoalPosition = boundaryGoalPositions.pop()
                self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
            else:
                if self.captureAgent.boundaryGoalPosition == self.intialPosition:
                    boundaryGoalPosition = boundaryGoalPositions.pop()
                    self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
                else:
                    boundaryGoalPosition = self.captureAgent.boundaryGoalPosition
            return boundaryGoalPosition

        missingFoodPosition = self.getMissingFoodPosition()

        if missingFoodPosition != None:
            self.captureAgent.prevMissingFoodLocation = missingFoodPosition
            return missingFoodPosition

        for enemy in self.enemies:
            if self.startingGameState.get_agent_state(enemy).is_pacman:
                self.captureAgent.enemyEntered = True
                if self.startingGameState.get_agent_position(enemy) != None:
                    return self.startingGameState.get_agent_position(enemy)
                else:
                    return self.getProbableEnemyEntryPointBasedOnFood()
                    # return self.getProbableEnemyEntryPoint()
            else:
                self.captureAgent.enemyEntered = False

        if self.captureAgent.prevMissingFoodLocation != None and self.captureAgent.enemyEntered:
            return self.captureAgent.prevMissingFoodLocation

        boundaryGoalPositions = self.closestPosition(
            self.intialPosition, self.viableBoundaryPositions)

        if self.captureAgent.boundaryGoalPosition == None:
            boundaryGoalPosition = boundaryGoalPositions.pop()
            self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
        else:
            if self.captureAgent.boundaryGoalPosition == self.intialPosition:
                boundaryGoalPosition = boundaryGoalPositions.pop()
                self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
            else:
                boundaryGoalPosition = self.captureAgent.boundaryGoalPosition

        return boundaryGoalPosition

    def closestPosition(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        for toPos in positions:
            positionsSorted.push(
                toPos, self.captureAgent.get_maze_distance(toPos, fromPos))
        return positionsSorted

    def getProbableEnemyEntryPoint(self):
        positionsSorted = util.PriorityQueue()
        positionsSorted = self.closestPosition(
            self.intialPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.get_maze_distance(self.intialPosition, possibleEntry) > 5:
                return possibleEntry
        return random.choice(self.possibleEnemyEntryPositions)

    def getProbableEnemyEntryPointBasedOnFood(self):
        positionsSorted = util.PriorityQueue()
        bestEnemyPosition = util.PriorityQueue()
        positionsSorted = self.closestPosition(
            self.intialPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.get_maze_distance(self.intialPosition, possibleEntry) > 5:
                closestFoodPosition = self.closestPosition(
                    possibleEntry, self.myPreciousFood.as_list()).pop()
                distancetoToClosestFoodFromPosition = self.captureAgent.get_maze_distance(
                    possibleEntry, closestFoodPosition)
                bestEnemyPosition.push(
                    possibleEntry, distancetoToClosestFoodFromPosition)

        bestEnemyEntryPosition = bestEnemyPosition.pop()

        if bestEnemyEntryPosition:
            return bestEnemyEntryPosition
        else:
            return random.choice(self.possibleEnemyEntryPositions)

    def getMissingFoodPosition(self):

        prevFood = self.captureAgent.get_food_you_are_defending(self.captureAgent.get_previous_observation()).as_list() \
            if self.captureAgent.get_previous_observation() is not None else list()

        currFood = self.captureAgent.get_food_you_are_defending(self.startingGameState).as_list()

        if prevFood:
            if len(prevFood) > len(currFood):
                foodEaten = list(set(prevFood) - set(currFood))
                if foodEaten:
                    return foodEaten[0]
        return None

    def getStartState(self):
        return (self.startingGameState, self.goalDistance)

    def isGoalState(self, state):

        game_state = state[0]

        (x, y) = myPos = game_state.get_agent_position(self.captureAgent.index)

        if myPos == self.GOAL_POSITION:
            return True
        else:
            return False

    def get_successors(self, state, path = None):
        self.expanded += 1

        game_state = state[0]

        actions= game_state.get_legal_actions(self.captureAgent.index)

        goalDistance = self.captureAgent.get_maze_distance(self.GOAL_POSITION, game_state.get_agent_position(self.captureAgent.index))

        successors_all = [((game_state.generate_successor(self.captureAgent.index, action), goalDistance), action, 1) for action in actions]

        successors = []

        for successor in successors_all:
            (xs, ys) = successor[0][0].get_agent_position(
                self.captureAgent.index)
            if self.captureAgent.red:
                if xs <= self.boundary:
                    successors.append(successor)
            else:
                if xs >= self.boundary:
                    successors.append(successor)

        return successors


def defensiveHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    game_state = state[0]
    currGoalDistance = state[1]

    succGoalDistance = problem.captureAgent.get_maze_distance(
        problem.GOAL_POSITION, game_state.get_agent_position(problem.captureAgent.index))

    if succGoalDistance < currGoalDistance:
        return 0
    else:
        return float('inf')

################# Search Algorithems ###################


class SolutionNotFound(Exception):
    pass


class Node():
    def __init__(self, *, name):
        self.name = name

    def add_info(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


def nullHeuristic(state, problem=None):
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []

    # use a priority queue to store states
    frontier = util.PriorityQueue()

    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(), [], 0)

    # as a priority there will be the total cost + heuristic
    # as first node this is irrelevant
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        (node, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        if problem.isGoalState(node):
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
                    total_cost = cost + n_cost + heuristic(child, problem)
                    total_path = path + [n_action]
                    
                    frontier.push((child, total_path, fut_cost), total_cost)