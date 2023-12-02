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
        self.start = None
        self.index = index
        self.prev_food = None
        self.safe_boundary = None
        self.gridHeight = None
        self.gridWidth = None
        self.start = None
        self.safe_pos = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        # the following initialises self.red and self.distancer
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


class offensiveAgent(agentBase):

    def choose_action(self, game_state):
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action

        problem = FoodOffenseWithAgentAwareness(startingGameState=game_state, captureAgent=self)


        actions = aStarSearch(problem, heuristic=self.offensiveHeuristic)

        # this can occure if start in the goal state. In this case do not want to perform any action.
        if actions == []:
            actions = ["Stop"]

        return actions[0]
    
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
        EATING_CAPSULE_MUL = 100
        RETURN_HOME_ALL_FOOD_MUL = 100
        RETURN_HOME_CHASED_MUL = 10
        RETURN_HOME_ENOUGH_FOOD_MUL = 10
        ENEMY_PENALTY_MUL = 10
        TEAMMATE_PENALTY_MUL = 5
        SCORE_MUL = 30

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

        # Reward based on the score
        score_reward = current_score * SCORE_MUL

        # Penalty for being closer to an enemy while on offense
        enemy_penalty = sum(self.get_maze_distance(my_pos, enemy) for enemy in enemy_positions) * ENEMY_PENALTY_MUL

        # Combine rewards and penalties to form the heuristic
        heuristic = food_reward + home_reward - teammate_penalty + score_reward - enemy_penalty

        return heuristic


#################  problems and heuristics  ####################



class FoodOffenseWithAgentAwareness():
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
        self.startingGameState = startingGameState
        # Need to ignore previous score change, as everything should be considered relative to this state
        self.startingGameState.data.score_change = 0
        self.MINIMUM_IMPROVEMENT = 1
        self.DEPTH_CUTOFF = 20
        # WARNING: Capture agent doesn't update with new state, this should only be used for non state dependant utils (e.g distancer)
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

        # Note: can't use CaptureAgent, at it doesn't update with game state
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

        actions = game_state.get_legal_actions(self.captureAgent.index)
        # not interested in exploring the stop action as the state will be the same as out current one.
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        next_game_states = [game_state.generate_successor(self.captureAgent.index, action) for action in actions]

        # if planning close to agent, include expected ghost activity
        current_depth_of_search = len(path)

        # we are only concerned about being eaten when we are pacman
        if current_depth_of_search <= self.DEPTH_CUTOFF and game_state.get_agent_state(self.captureAgent.index).is_pacman:
            self.expanded += 1  # track number of states expanded

            # make any nearby enemy ghosts take a step toward you if legal
            for i, next_game_state in enumerate(next_game_states):
                # get enemys
                current_agent_index = self.captureAgent.index
                enemy_indexes = self.captureAgent.get_opponents(next_game_state)

                # keep only enemies that are close enough to catch pacman.
                close_enemy_indexes = [enemy_index for enemy_index in enemy_indexes if next_game_state.get_agent_position(enemy_index) is not None]
                
                my_pos = next_game_state.get_agent_state(current_agent_index).get_position()
                
                adjacent_enemy_indexs = list(filter(lambda x: self.captureAgent.get_maze_distance(my_pos, next_game_state.get_agent_state(x).get_position()) <= 1, close_enemy_indexes))

                # check in enemies are in the right state
                adjacent_ghost_indexs = list(filter(lambda x: (not next_game_state.get_agent_state(x).is_pacman) and (next_game_state.get_agent_state(x).scared_timer <= 0), adjacent_enemy_indexs))

                # move enemies to the pacman position
                ghost_kill_directions = []
                for index in adjacent_ghost_indexs:
                    position = next_game_state.get_agent_state(index).get_position()
                    for action in game_state.get_legal_actions(self.captureAgent.index):
                        new_pos = Actions.get_successor(position, action)
                        if new_pos == my_pos:
                            ghost_kill_directions.append(action)
                            break

                # update state:
                for enemy_index, direction in zip(adjacent_ghost_indexs, ghost_kill_directions):
                    self.expanded += 1
                    next_game_state = next_game_state.generate_successor(
                        enemy_index, direction)

                # make the update
                next_game_states[i] = next_game_state
                # if they are next to pacman, move ghost to pacman possiton

        successors = [((next_game_state,), action, 1)
                      for action, next_game_state in zip(actions, next_game_states)]

        return successors




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
        actions = aStarSearch(problem, heuristic=self.defensiveHeuristic)

        if actions != None:
            return actions[0]
        else:
            return random.choice(game_state.get_legal_actions(self.index))

    def defensiveHeuristic(self, state, problem=None):
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

    def getViableBoundaryPositions(self):
        myPos = self.startingGameState.get_agent_position(self.captureAgent.index)
        b = self.boundary
        boundaryPositions = []
        enemyEntryPositions = []

        for h in range(0, self.gridHeight):
            if self.captureAgent.red:
                if not(self.walls[b][h]) and not(self.walls[b+1][h]):
                    if (b, h) != myPos:
                        boundaryPositions.append((b, h))
                    enemyEntryPositions.append((b+1, h))

            else:
                if not(self.walls[b][h]) and not(self.walls[b-1][h]):
                    if (b, h) != myPos:
                        boundaryPositions.append((b, h))
                    enemyEntryPositions.append((b-1, h))

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