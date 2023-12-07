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
from contest.util import nearestPoint

# Standard imports
from contest.captureAgents import CaptureAgent
import random
from contest.game import Directions, Actions, AgentState  # basically a class to store data
from contest.capture import GameState

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
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_distances = [self.get_maze_distance(my_pos, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        
        are_scared = [a.scared_timer > 8 for a in enemies if not a.is_pacman and a.get_position() != None]
        #exist_threat = thereIsAThreat(game_state, self)
        prev_game_state = self.get_previous_observation()
        #if prev_game_state != None:
        #    used_to_be_threat = thereIsAThreat(prev_game_state, self)
        #else: 
        #    used_to_be_threat = False
        

        # Food
        food_list = self.get_food(game_state).as_list()
        is_carrying_food = agent_state.num_carrying > 0
        
        # Capsule
        capsules = self.get_capsules(game_state)
        if capsules:
            capsule_distances = [self.get_maze_distance(my_pos, cap) for cap in capsules]
            min_capsule_distance = min(capsule_distances)

        problem = FoodOffense(startingGameState=game_state, captureAgent=self)
        

        if len(food_list) > 0:
            nearest_food = min(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
            nearest_food_distance = self.get_maze_distance(my_pos, nearest_food)
        
        is_scared = False

        
        if len(ghost_distances) > 0:
            min_ghost_distance = min(ghost_distances)
            min_ghost_distance_index = ghost_distances.index(min_ghost_distance)
            is_scared = are_scared[min_ghost_distance_index]
        else:
            min_ghost_distance = 1000

        if capsules and (len(ghost_distances) and min_ghost_distance < 5 or min_capsule_distance < 2):   # Adjust distance threshold as needed
            best_capsule = min(capsules, key=lambda cap: self.get_maze_distance(my_pos, cap))
            path_to_food = aStarSearch(problem, self.heuristicToPos, goal=problem.isGoalStatePosition, target= best_capsule)
            return path_to_food[0] if path_to_food else Directions.STOP
        
        carrying_food = game_state.get_agent_state(self.index).num_carrying
        food_limit = 5
        
        if is_scared: 
            food_limit = 10
        
        ispac = game_state.get_agent_state(self.index).is_pacman
        
        if game_state.data.timeleft > 50 and carrying_food <= food_limit:
            if len(food_list) > 2 and (not ispac or is_scared or (nearest_food_distance < min_ghost_distance and abs(nearest_food_distance - min_ghost_distance) > 3)):
                path_to_food = aStarSearch(problem, self.heuristicToPos, goal=problem.isGoalStatePosition, target=nearest_food)
                return path_to_food[0] if path_to_food else Directions.STOP
            else:
                path_to_home = aStarSearch(problem, self.heuristicToPos, goal=problem.isGoalStatePosition, target=self.start)
                return path_to_home[0] if path_to_home else Directions.STOP
        else:       
            path_to_home = aStarSearch(problem, self.heuristicToPos, goal=problem.isGoalStatePosition, target=self.start)
            return path_to_home[0] if path_to_home else Directions.STOP
    
    

################# Heuristics ###################################

    def heuristicToPos(self, data):
        game_state = data['game']
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        num_carrying = game_state.get_agent_state(self.index).num_carrying
        ghost_distances = [self.get_maze_distance(self.start, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        are_scared = [enemy.scared_timer > 5 for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        min_ghost_distance = 100
        if len(ghost_distances) > 0:
            min_ghost_distance = min(ghost_distances)
            min_ghost_distance_index = ghost_distances.index(min_ghost_distance)
            is_scared = are_scared[min_ghost_distance_index]
            if is_scared:
                min_ghost_distance = 0
        return self.get_maze_distance(data['pos'], data['target']) + min_ghost_distance * num_carrying

    def eatingFoodHeuristic(self, data):
        # MIN HEURISTIC:
        # -11 if there was food in the current pos
        # ascending following the distance to the food
        game_state = data['game']
        DISTANCE_FOOD_MUL = 10

        offence_food_pos = self.get_food(game_state).as_list()
        current_pos = game_state.get_agent_position(self.index)
        
        prev_game_state = self.get_previous_observation()
        if prev_game_state != None:
            if prev_game_state.has_food(current_pos[0], current_pos[1]):
                return - (DISTANCE_FOOD_MUL + 1)

        distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
        return - 1 / (distanceClosestFood+1) * DISTANCE_FOOD_MUL
    
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
        

    def getStartState(self):
        # This needs to return the state information to being with

        return self.startingGameState.get_agent_position(self.index)
        
    def getStartData(self, target = None):
        # This needs to return the state information to being with
        data = util.Counter()
        data['prev_game'] = None
        data['game'] = self.startingGameState
        data['action'] = None
        data['cost'] = 0
        data['pos'] = self.startingGameState.get_agent_position(self.index)
        data['agent'] = self.captureAgent
        data['target'] = target

        return data

    def isGoalStatePosition(self, data):
        return data['pos'] == data['target']
            

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
                data['game'] = game_state.generate_successor(self.index, next_action)
                data['action'] = next_action
                data['cost'] = 1
                data['pos'] = fut_pos
                data['agent'] = self.captureAgent
                data['target'] = old_data['target']
                successors.append(data)

        return successors

################# Search Algorithems ###################




def aStarSearch(problem, heuristic=None, goal = None, target = None):
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
    start_node = (problem.getStartData(target),  [], 0)

    distributions = [util.Counter()]
    
    # as a priority there will be the total cost + heuristic
    # as first node this is irrelevant
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        (data, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        if goal(data):
            #updateDistributions(distributions, data, heuristic)
            return path

        # if not the algorithm search in his successor and insert them in the frontier to be expanded
        for new_data in problem.get_successors(data):
            fut_cost = cost + new_data['cost']
            new_pos = new_data['pos']

            if new_pos not in expanded_nodes or fut_cost < min_cost[new_pos]:
                min_cost[new_pos] = fut_cost
                expanded_nodes.add(new_pos)
                
                # the priority is the fut cost + heuristic
                priority = fut_cost + heuristic(new_data)
                
                next_action = new_data['action']
                total_path = path + [next_action]
                
                #updateDistributions(distributions, data, heuristic, show = False)
                    
                frontier.push((new_data, total_path, fut_cost), priority)

def showAndNormalize(distributions, agent):
    distributions[0].incrementAll(distributions[0].keys(), -min(distributions[0].values()))
    distributions[0].normalize()
    agent.display_distributions_over_positions(distributions)

def updateDistributions(distributions, data, heuristic, show = True):
    distributions[0][data['pos']] = heuristic(data)
    if show:
        showAndNormalize([distributions[0].copy()], data['agent'])

########################## DEFENCEEEE ##################

class defensiveAgent(agentBase):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        self.mode = DefenceModes.Default
        # Missing food location tell us the direction of the pacman
        # and where we should head. I will by default try to intercept
        # him from the front
        self.prevMissingFoodLocation = None
        self.missingFoodLocation = None

        # A counter used to determine if the missingFoodPrevTarget
        # is valid. This is to reuse it in case 1 turn ago a food was
        # eaten and we still want to continue going towards the next food
        self.missingFoodCounter = 0
        # The below value is the initial value given to the counter 
        # when missing food is found
        self.initMissingFoodCounter = 4
        # This is the current target and actions
        self.target = None
        self.actions = None

        self.possibleEnemyEntryPositions = None
        self.boundaryPositions = None

        self.enemyEntered = False
        self.boundaryGoalPosition = None

    def getMissingFoodPosition(self, startingGameState):

        prevFood = self.get_food_you_are_defending(
                self.get_previous_observation()).as_list() \
            if self.get_previous_observation() is not None else list()

        currFood = self.get_food_you_are_defending(startingGameState).as_list()

        if prevFood:
            if len(prevFood) > len(currFood):
                foodEaten = list(set(prevFood) - set(currFood))
                if foodEaten:
                    return foodEaten[0]
        return None

    def closestPositionWithPriorityQueue(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        while not positions.isEmpty():
            toPos = positions.pop()
            positionsSorted.push(
                toPos, self.get_maze_distance(toPos, fromPos))
        return positionsSorted

    def closestPosition(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        for toPos in positions:
            positionsSorted.push(
                toPos, self.get_maze_distance(toPos, fromPos))
        return positionsSorted

    def goalReached(self):
        return self.actions == None or self.actions ==[]

    def isFoodMissing(self, gameState):
        return self.getMissingFoodPosition(gameState) is not None

    def enemyVisible(self, enemy_idx, gameState):
        return gameState.get_agent_position(enemy_idx) is not None
    
    def anyEnemyVisible(self, game_state):
        enemies = self.get_opponents(game_state)
        return any([self.enemyVisible(idx, game_state) for idx in enemies])
    
    def agentDied(self, game_state):
        prev_game_state = self.get_previous_observation()
        prevPos = prev_game_state.get_agent_position(self.index)
        currPos = game_state.get_agent_position(self.index)
        distance = self.get_maze_distance(
            prevPos, currPos)

        return distance > 2
    
    def capsuleEaten(self, game_state):
        prev_game_state = self.get_previous_observation()
        prev_capsule = self.get_capsules_you_are_defending(prev_game_state)
        capsulesNow = self.get_capsules_you_are_defending(game_state)
        
        return len(capsulesNow) != len(prev_capsule)
        

    def choose_action(self, game_state):
        # If we see an enemy or some food goes mising or a goal is reached we need to 
        # recalculate
        change = self.anyEnemyVisible(game_state) or self.isFoodMissing(game_state) or \
            self.actions == None or self.goalReached() or self.agentDied(game_state) or \
            self.capsuleEaten(game_state)


        if change:
            problem = defendTerritoryProblem(startingGameState=game_state,
                                        captureAgent=self)
            self.actions = aStarSearch(problem, heuristic=self.defensiveHeuristic, goal=problem.isGoalStatePosition)
        
        # I reduce the counter here so that it is reduced even if a change has
        # not occured
        self.missingFoodCounter -= 1

        if self.actions != None and self.actions != []:
            action = self.actions[0]
            self.actions = self.actions[1:]
            return action
        else:
            return random.choice(game_state.get_legal_actions(self.index))

    def isAtMissingFoodLocation(self, data):
        missingFoodExists = data['agent'].missingFoodLocation != None
        return missingFoodExists and data['pos'] ==\
            data['agent'].missingFoodLocation

    def defensiveHeuristic(self, data):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
        BOUNDRY_DISTANCE_MUL = 10
        GOAL_DISTANCE_MUL = 5
        SCARED_GOAL_DISTANCE_MUL = 10
        MISSING_FOOD_POSITION_PENALTY = 100

        game_state = data['game']
        currGoalDistance = data['goal_distance']
        agentPos = data['pos']
        succGoalDistance = data['agent'].get_maze_distance(
            data['target'], agentPos)

        closestBoundary = self.closestPosition(agentPos, self.boundaryPositions).pop()
        
        boundaryDistance = data['agent'].get_maze_distance(
            closestBoundary, agentPos)
        
        heuristic = 0

        # If we are clyde we don't want to reach the goal
        # Just get close
        if self.mode == DefenceModes.Clyde and succGoalDistance != 0:
            heuristic += 10 / (succGoalDistance * SCARED_GOAL_DISTANCE_MUL)
        else:
            heuristic += succGoalDistance * GOAL_DISTANCE_MUL

        heuristic +=  self.isAtMissingFoodLocation(data) *\
              MISSING_FOOD_POSITION_PENALTY

        heuristic += boundaryDistance * BOUNDRY_DISTANCE_MUL 

        if succGoalDistance < currGoalDistance:
            return heuristic
        else:
            return float('inf')

class DefenceModes(Enum):
    # Default mode is set to Pinky
    Default = 2
    Clyde = 1
    Pinky = 2
    Blinky = 3


class defendTerritoryProblem():
    def __init__(self, startingGameState, captureAgent):
        self.NthClosestFood = 1
        self.startingGameState = startingGameState
        self.captureAgent = captureAgent
        self.enemies = self.captureAgent.get_opponents(startingGameState)
        self.walls = startingGameState.get_walls()
        self.agentPosition = self.startingGameState.get_agent_position(
            self.captureAgent.index)
        self.gridWidth = self.captureAgent.get_food(startingGameState).width
        self.gridHeight = self.captureAgent.get_food(startingGameState).height
        if self.captureAgent.red:
            self.boundary = int(self.gridWidth / 2) - 1
        else:
            self.boundary = int(self.gridWidth / 2)
            
        self.myPreciousFood = self.captureAgent.get_food_you_are_defending(startingGameState)

        (self.captureAgent.boundaryPositions,
         self.captureAgent.possibleEnemyEntryPositions) = self.getViableBoundaryPositions()


        self.GOAL_POSITION = self.getGoalPosition()
        self.goalDistance = self.captureAgent.get_maze_distance(self.GOAL_POSITION,
                                                             self.agentPosition)

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

    def getTargetBasedOnEnemyPosition(self, enemyPosition):
        """
        The idea of this method is to find the closest food to the
        enemy position. Then from that position find the next
        closest food excluding the one we already found and so on until
        the NthClosestFood which we return. If we reach the 
        final food before that we return it
        """
        closestFoodPositions = self.captureAgent.closestPosition(
            enemyPosition, self.myPreciousFood.as_list())
        closestFoodPosition = closestFoodPositions.pop()
        for _ in range(self.NthClosestFood - 1):
            if closestFoodPositions.isEmpty():
                return closestFoodPosition
            closestFoodPositions =  self.captureAgent.closestPositionWithPriorityQueue(
                                                closestFoodPosition,
                                                closestFoodPositions)
            closestFoodPosition = closestFoodPositions.pop()
        
        return closestFoodPosition

    def getClosestFood(self):
        """Return minimum distance to food and its position"""
        game_state = self.startingGameState
        food_list = self.captureAgent.get_food(game_state).as_list()
        agentPosition = self.agentPosition

        queue = util.PriorityQueue()
        for food in food_list:
            dist = self.captureAgent.get_maze_distance(agentPosition, food)
            queue.push(food, dist)

        randomFoodChoice = 0
        if len(food_list) > 2:
            randomFoodChoice = random.randint(0, 2)

        for _ in range(randomFoodChoice):
            queue.pop()

        return queue.pop()

    def getRandomBorderPos(self):
        return random.choice(self.captureAgent.boundaryPositions)

    def getGoalPositionIfEnemyLocationUnknown(self):
        missingFoodPosition = self.captureAgent.getMissingFoodPosition(self.startingGameState)

        # If we have a previously missing food try to figure out a path pacman
        # might take and intercept him 
        if missingFoodPosition != None:
            self.captureAgent.prevMissingFoodLocation = self.captureAgent.missingFoodLocation
            self.captureAgent.missingFoodLocation = missingFoodPosition
            self.captureAgent.missingFoodCounter = self.captureAgent.initMissingFoodCounter

            if self.captureAgent.mode == DefenceModes.Pinky:
                # missing food position is the possible enemy position
                self.captureAgent.prevMissingFoodTarget = self.getTargetBasedOnEnemyPosition(missingFoodPosition)
            else:
                self.captureAgent.prevMissingFoodTarget = missingFoodPosition

        # If this turn no new missing food is eaten try to reuse the old target
        if self.captureAgent.missingFoodCounter > 0:
            return self.captureAgent.prevMissingFoodTarget

        return self.getProbableEnemyEntryPointBasedOnFood()

    def getClosestBoundry(self):
        boundaries = self.captureAgent.boundaryPositions
        q = util.PriorityQueue()
        for boundary in boundaries:
            dist = self.captureAgent.get_maze_distance(self.agentPosition, boundary)
            q.push(boundary,  dist)
        
        return q.pop()

    def getGoalPosition(self):
        agentState = self.startingGameState.get_agent_state(
            self.captureAgent.index)

        if agentState.is_pacman:
            return self.getClosestBoundry()

        isScared = agentState.scared_timer > 0
        if isScared:
            self.captureAgent.mode = DefenceModes.Clyde
        else:
            self.captureAgent.mode = DefenceModes.Default

        # If an enemy agents position is know calculate target
        # If not approximate possible targets 
        for enemy in self.enemies:
            if self.startingGameState.get_agent_state(enemy).is_pacman:
                if self.startingGameState.get_agent_position(enemy) != None:
                    enemyPosition = self.startingGameState.get_agent_position(enemy)
                    # We decided to do blinky if we see an agent.
                    # Follow him
                    return enemyPosition
                else:
                    return self.getGoalPositionIfEnemyLocationUnknown()
            else:
                self.captureAgent.enemyEntered = False

        # If no enemy agent is a pacman try to take some food
        return self.getRandomBorderPos()


    def getProbableEnemyEntryPoint(self):
        positionsSorted = util.PriorityQueue()
        positionsSorted = self.captureAgent.closestPosition(
            self.agentPosition, self.captureAgent.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.get_maze_distance(self.agentPosition, possibleEntry) > 5:
                return possibleEntry
        return random.choice(self.captureAgent.possibleEnemyEntryPositions)

    def getProbableEnemyEntryPointBasedOnFood(self):
        positionsSorted = util.PriorityQueue()
        bestEnemyPosition = util.PriorityQueue()
        # Which entry point are we closest to
        positionsSorted = self.captureAgent.closestPosition(
            self.agentPosition, self.captureAgent.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.get_maze_distance(self.agentPosition, possibleEntry) > 5:
                closestFoodPosition = self.captureAgent.closestPosition(
                    possibleEntry, self.myPreciousFood.as_list()).pop()
                distancetoToClosestFoodFromPosition = self.captureAgent.get_maze_distance(
                    possibleEntry, closestFoodPosition)
                bestEnemyPosition.push(
                    possibleEntry, distancetoToClosestFoodFromPosition)

        bestEnemyEntryPosition = bestEnemyPosition.pop()

        if bestEnemyEntryPosition:
            return bestEnemyEntryPosition
        else:
            return random.choice(self.captureAgent.possibleEnemyEntryPositions)


    def getStartState(self):
        # This needs to return the state information to being with

        return self.startingGameState.get_agent_position(self.captureAgent.index)
        
    def getStartData(self, target = None):
        # This needs to return the state information to being with
        data = util.Counter()
        data['prev_game'] = None
        data['game'] = self.startingGameState
        data['action'] = None
        data['cost'] = 0
        data['pos'] = self.startingGameState.get_agent_position(self.captureAgent.index)
        data['agent'] = self.captureAgent
        data['target'] = self.GOAL_POSITION

        return data

    def isGoalStatePosition(self, data):
        return data['pos'] == data['target']
        

    def get_successors(self, old_data):
        game_state = old_data['game']

        actions = game_state.get_legal_actions(self.captureAgent.index)
        agent = game_state.get_agent_state(self.captureAgent.index)

        agentPos = game_state.get_agent_position(self.captureAgent.index)
        goalDistance = self.captureAgent.get_maze_distance(old_data['target'],
                         agentPos)
        
        successors_all = []
        for action in actions:
            data = util.Counter()
            data['prev_game'] = game_state
            data['game'] = game_state.generate_successor(self.captureAgent.index, action)
            data['goal_distance'] = goalDistance
            data['pos'] = data['game'].get_agent_position(self.captureAgent.index)
            data['action'] = action
            data['cost'] = 1
            data['target'] = old_data['target']
            data['agent'] = self.captureAgent
            successors_all.append(data)

        if agent.is_pacman:
            return successors_all

        successors = []

        for successor in successors_all:
            nextPos = successor['pos']
            xs = nextPos[0]

            distance = self.captureAgent.get_maze_distance(
                nextPos, agentPos)
            agentDied = distance > 2

            if agentDied:
                continue
            
            if agent.is_pacman:
                successors.append(successor)
                continue

            if self.captureAgent.red and xs <= self.boundary:
                successors.append(successor)
            elif xs >= self.boundary:
                successors.append(successor)

        return successors