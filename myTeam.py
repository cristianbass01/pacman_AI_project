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

from functools import wraps
import time
from collections import defaultdict
import numpy as np

#times = defaultdict(list)

"""
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        N = 1 if func.__name__ in ['_init_', 'register_initial_state'] else 100
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        times[func.__name__].append(total_time)
        if len(times[func.__name__]) % N == 0:
           print(f'Function {func.__name__} took {np.mean(times[func.__name__]):.4f} seconds on average over the last {len(times[func.__name__])} calls')
            print(f'Function {func.__name__} took {np.max(times[func.__name__]):.4f} seconds on maximum over the last {len(times[func.__name__])} calls')
        return result
    return timeit_wrapper
"""
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

        self.actions = None

        self.last_seen_enemy_pos = None

        self.dead_ends = None
        self.nearest_exit_from_ends = None

        self.past_pos = None

        self.missingFoodLocation = None

        self.missingFoodCounter = 0

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

        self.possibleEnemyEntryPositions = None
        self.boundaryPositions = None
        self.defencePositions = 0

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

    def capsuleEaten(self, game_state):
        prev_game_state = self.get_previous_observation()
        if prev_game_state == None:
            return False
        prev_capsule = self.get_capsules_you_are_defending(prev_game_state)
        capsulesNow = self.get_capsules_you_are_defending(game_state)
        
        return len(capsulesNow) != len(prev_capsule)

    def goalReached(self):
        return self.actions == None or self.actions ==[]

    def isFoodMissing(self, gameState):
        return self.getMissingFoodPosition(gameState) is not None

    def enemyVisible(self, enemy_idx, gameState):
        return gameState.get_agent_position(enemy_idx) is not None
    
    def anyEnemyVisible(self, game_state):
        enemies = self.get_opponents(game_state)
        return any([self.enemyVisible(idx, game_state) for idx in enemies if game_state.get_agent_state(idx).is_pacman])
    
    def agentDied(self, game_state):
        prev_game_state = self.get_previous_observation()
        if prev_game_state == None:
            return False
        prevPos = prev_game_state.get_agent_position(self.index)
        currPos = game_state.get_agent_position(self.index)
        distance = self.get_maze_distance(
            prevPos, currPos)

        return distance > 2
    
    def isBecomeScared(self, game_state):
        prev_game_state = self.get_previous_observation()
        if prev_game_state != None:
            return prev_game_state.get_agent_state(self.index).scared_timer == 0 and game_state.get_agent_state(self.index).scared_timer > 0
        else:
            return False


    def choose_action_defensive(self, game_state):
        # If we see an enemy or some food goes missing or a goal is reached we need to 
        # recalculate
        change = self.anyEnemyVisible(game_state) or self.isFoodMissing(game_state) or \
            self.actions == None or self.goalReached() or self.agentDied(game_state) or \
            self.capsuleEaten(game_state) or self.isBecomeScared(game_state)

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
    
    def closestBoundaryPos(self, game_state):
        positionsSorted = util.PriorityQueue()
        my_pos = game_state.get_agent_position(self.index)

        for boundary_pos in self.boundary_pos:
            positionsSorted.push(
                boundary_pos, distance(game_state, self, boundary_pos))

        closestBoundary = positionsSorted.pop()
        return closestBoundary
    
    def distanceClosestBoundaryPos(self, game_state):
        return distance(game_state, self, self.closestBoundaryPos(game_state))
    
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
        MISSING_FOOD_POSITION_PENALTY = 100

        game_state = data['game']
        currGoalDistance = data['goal_distance']
        agentPos = data['pos']
        succGoalDistance = data['agent'].get_maze_distance(
            data['target'], agentPos)

        closestBoundary = self.closestPosition(agentPos, self.boundaryPositions).pop()
        
        boundaryDistance = data['agent'].get_maze_distance(
            closestBoundary, agentPos)
        
        my_capsules = self.get_capsules_you_are_defending(game_state)

        prev_game_state = self.get_previous_observation()

        heuristic = 0
        enemies = self.get_opponents(game_state)
        enemy_dist = 100
        my_dist_to_caps = 0
        # The distance between my agent and the capsule closest to the
        # enemy should be less then the one to the enemy/I can intercept
        # him

        notChangedDistance = False
        for enemy in enemies:
            enemyPos = game_state.get_agent_position(enemy)
            
            enemy_state = game_state.get_agent_state(enemy)
            if enemy_state.is_pacman and prev_game_state != None:
                prevEnemyPos = prev_game_state.get_agent_position(enemy)
                if prevEnemyPos != None and enemyPos != None:
                    prevPos = prev_game_state.get_agent_position(self.index)
                    prevDistance = data['agent'].get_maze_distance(prevPos, prevEnemyPos)
                    curDistance = data['agent'].get_maze_distance(agentPos, enemyPos)
                    notChangedDistance = curDistance == prevDistance 
           

            for capsule in my_capsules:
                if enemyPos != None:
                    dist = data['agent'].get_maze_distance(
                        enemyPos, capsule)
                    if dist < enemy_dist:
                        my_dist_to_caps = data['agent'].get_maze_distance(data['pos'], capsule)
                        enemy_dist = dist
        
        heuristic += notChangedDistance * 100
        heuristic += self.isFoodMissing(game_state) * 50
        heuristic -= (enemy_dist - my_dist_to_caps) * 100
        heuristic += succGoalDistance * GOAL_DISTANCE_MUL
   
        heuristic += boundaryDistance * BOUNDRY_DISTANCE_MUL 

        if succGoalDistance < currGoalDistance:
            return heuristic
        else:
            return float('inf')

    def choose_action_aggressive(self, game_state):
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action
        # General
        agent_state = game_state.get_agent_state(self.index)
        start = self.start == game_state.get_agent_position(self.index)
        no_moves_left = self.actions == None or len(self.actions) == 0
        my_pos = game_state.get_agent_state(self.index).get_position()
        self.past_pos.append(my_pos)
        if len(self.past_pos) > 5:
            self.past_pos.pop(0)
        is_stuck = len(set(self.past_pos)) < 3 and len(self.past_pos) == 5
        min_safe_pos_distance = min([self.get_maze_distance(my_pos, safe_pos) for safe_pos in self.boundary_pos])

        # Enemy
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_distances = [self.get_maze_distance(my_pos, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        
        exist_threat = thereIsAThreat(game_state, self, 2)
        prev_game_state = self.get_previous_observation()

        # Food
        food_list = self.get_food(game_state).as_list()
        food_carrying = agent_state.num_carrying
            
        # Capsule
        capsules = self.get_capsules(game_state)
        min_capsule_distance = 1000
        if capsules:
            capsule_distances = [self.get_maze_distance(my_pos, cap) for cap in capsules]
            min_capsule_distance = min(capsule_distances)

        if needToCalculate(game_state, prev_game_state, self):
            # Problem
            problem = FoodOffense(startingGameState=game_state, captureAgent=self)

            #best_path_to_safe, _ = self.calculatePathsTo(problem, self.boundary_pos, is_stuck, n_choose = 3)
            best_path_to_safe, _ = self.calculatePath(problem, self.returnSafeHeuristic, problem.isGoalStateReturnSafe)

            food_carrying = game_state.get_agent_state(self.index).num_carrying
            food_limit = 5

            min_safe_pos_distance = 100
            if best_path_to_safe != None:
                min_safe_pos_distance = len(best_path_to_safe)

            is_enemy_scared_enough = False

            closer_enemy_index = None

            if len(ghost_distances) > 0:
                min_ghost_distance = min(ghost_distances)
                closer_enemy_index = ghost_distances.index(min_ghost_distance)
                scared_less_capsule_dist = scaredTimerByIndex(game_state, closer_enemy_index) < (min_capsule_distance + 1)
                is_enemy_scared_enough = scaredTimerByIndex(game_state, closer_enemy_index) > (min_safe_pos_distance + 1)
                enemy_pos = game_state.get_agent_position(closer_enemy_index)
                if enemy_pos != None:
                    is_enemy_closer_to_capsule = any([distance(game_state, self, capsule) > self.get_maze_distance(enemy_pos, capsule) for capsule in capsules])
                else:
                    is_enemy_closer_to_capsule = False

                if capsules and (min_ghost_distance < 4 or min_capsule_distance < 2) and scared_less_capsule_dist and not is_enemy_scared_enough and not is_enemy_closer_to_capsule:
                    self.actions, _ = self.calculatePathsTo(problem, capsules, is_stuck, n_choose=1)

                    if self.actions == None or self.actions == []:
                        self.actions = best_path_to_safe
                    
                    if self.actions == None or self.actions == []:
                        return getRandomSafeAction(game_state, self)
                    
                    return self.actions.pop(0)
            else:
                min_ghost_distance = 100

            if len(food_list) > 0:
                best_path_to_food, chosen_food = self.calculatePath(problem, self.eatingFoodHeuristic, problem.isGoalStateEatingFood)
            else:
                best_path_to_food = []
                chosen_food = None

            if is_enemy_scared_enough: 
                current_score = game_state.data.score
                if self.red and current_score < 0:
                    food_limit = min(-current_score + 2, len(food_list) - 2) 
                elif not self.red and current_score > 0:
                    food_limit = min(current_score + 2, len(food_list) - 2) 
                else:
                    food_limit = min(8, len(food_list) - 2)
            
            im_ghost = isGhostByIndex(game_state, self.index)
            
            if (len(food_list) > 2) and chosen_food in self.dead_ends and closer_enemy_index != None and (game_state.data.timeleft > (min_safe_pos_distance + 5)  and food_carrying <= food_limit):
                distance_to_food = distance(game_state, self, chosen_food)
                distance_food_to_exit = self.get_maze_distance(my_pos, self.nearest_exit_from_ends[chosen_food])
                enemy_pos = game_state.get_agent_position(closer_enemy_index)
                if enemy_pos != None:
                    distance_ghost_to_exit = self.get_maze_distance(enemy_pos, self.nearest_exit_from_ends[chosen_food])
                else:
                    distance_ghost_to_exit = 1000

                if distance_to_food + distance_food_to_exit + 1 < distance_ghost_to_exit:
                    self.actions = best_path_to_food
                else:
                    # Search for other food that is not in a dead_end
                    new_food = [food for food in food_list if food not in self.dead_ends]
                    
                    if new_food == []:
                        self.actions = best_path_to_safe
                    else:
                        chosen_food = [getClosestFood(game_state, self)]
                        self.actions, _ = self.calculatePathsTo(problem, chosen_food, is_stuck, n_choose = 1)
            
                    
            elif (im_ghost or is_enemy_scared_enough or food_carrying <= food_limit) and (len(food_list) > 2) and (game_state.data.timeleft > (min_safe_pos_distance + 5)):
                self.actions = best_path_to_food
                
                if self.actions == None or self.actions == []:
                    self.actions = best_path_to_safe
            
            else: 

                self.actions = best_path_to_safe

                if self.actions == None or self.actions == []:
                    self.actions = best_path_to_food
                
    
        if self.actions == None or self.actions == [] or is_stuck or not self.actions[0] in game_state.get_legal_actions(self.index):
            #print('Random action taken')
            self.actions = []
            return getRandomSafeAction(game_state, self)
        
        return self.actions.pop(0)



    def calculatePathsTo(self, problem, positions, is_stuck, n_choose = 1):
        my_pos = self.past_pos[-1]

        if len(positions) > n_choose:
            boundary_distances = [self.get_maze_distance(my_pos, pos) for pos in self.boundary_pos]
            weights = [1 / dist**2 if dist > 0 else 1 for dist in boundary_distances]

            # Choose a random food based on the weights
            new_positions = random.choices(self.boundary_pos, weights=weights, k=n_choose)
        else:
            new_positions = positions

        paths = util.PriorityQueue()
        for pos in new_positions:
            if is_stuck:
                path = aStarSearch(problem, self.scaredHeuristicToPos, goal=problem.isGoalStatePosition, target= pos)
            else:
                path = aStarSearch(problem, self.heuristicToPos, goal=problem.isGoalStatePosition, target= pos)
            if path != None:
                paths.push(path, len(path))

        
        if n_choose == 1: 
            if paths.isEmpty(): return None, new_positions[0]
            return paths.pop(), new_positions[0]
        else:
            if paths.isEmpty(): return None, new_positions
            return paths.pop(), new_positions
        
    def calculatePath(self, problem, heuristic, goal):
        return aStarSearch(problem, heuristic, goal=goal), problem.goal

################# Heuristics ###################################

    def heuristicToPos(self, data):
        game_state = data['game']
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        food_carrying = game_state.get_agent_state(self.index).num_carrying
        ghost_distances = [self.get_maze_distance(self.start, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        min_ghost_distance = 100
        if len(ghost_distances) > 0:
            min_ghost_distance = min(ghost_distances)
            closer_enemy_index = ghost_distances.index(min_ghost_distance)
            is_scared = scaredTimerByIndex(game_state, closer_enemy_index) < 5 if isGhostByIndex(game_state, closer_enemy_index) and game_state.get_agent_position(closer_enemy_index) != None else True
            if is_scared:
                min_ghost_distance = 100
        return self.get_maze_distance(data['pos'], data['target']) - min_ghost_distance * (food_carrying + 1)
    
    def scaredHeuristicToPos(self, data):
        game_state = data['game']
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        food_carrying = game_state.get_agent_state(self.index).num_carrying
        ghost_distances = [self.get_maze_distance(self.start, enemy.get_position()) for enemy in enemies if not enemy.is_pacman and enemy.get_position() != None]
        min_ghost_distance = 1000
        if len(ghost_distances) > 0:
            min_ghost_distance = min(ghost_distances)
            closer_enemy_index = ghost_distances.index(min_ghost_distance)
            is_scared = scaredTimerByIndex(game_state, closer_enemy_index) < 5 if isGhostByIndex(game_state, closer_enemy_index) and game_state.get_agent_position(closer_enemy_index) != None else True
            if is_scared:
                min_ghost_distance = 1000
        return self.get_maze_distance(data['pos'], data['target']) - min_ghost_distance * (food_carrying + random.randint(1,5))


    def eatingFoodHeuristic(self, data):
        # MIN HEURISTIC:
        # -11 if there was food in the current pos
        # ascending following the distance to the food
        game_state = data['game']
        DISTANCE_FOOD_MUL = 10

        offence_food_pos = self.get_food(game_state).as_list()
        current_pos = game_state.get_agent_position(data['agent'].index)
        
        prev_game_state = self.get_previous_observation()
        if prev_game_state != None:
            if prev_game_state.has_food(current_pos[0], current_pos[1]):
                return - (DISTANCE_FOOD_MUL + 1)
            
        DISTANCE_ENEMY_MUL = 1000
        GO_BOUNDARY_POS = 4
        
        heur = 0

        for enemy in self.get_opponents(game_state):
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos != None:
                distance_to_enemy = distance(game_state, self, enemy_pos)
                enemy_state = game_state.get_agent_state(enemy)
                is_dangerous = distance_to_enemy < 3 and not isPacmanByIndex(game_state, enemy) 
                is_dangerous = is_dangerous and enemy_state.scared_timer < (distance(game_state, self, enemy_pos) + 1)
                if is_dangerous:
                    heur += 1/ (distance_to_enemy+1) * DISTANCE_ENEMY_MUL

        if len(offence_food_pos) > 0:
            distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
        else:
            return heur
        
        return - 1 / (distanceClosestFood+1) * DISTANCE_FOOD_MUL + heur
    
    def returnSafeHeuristic(self, data):
        game_state = data['game']
        DISTANCE_ENEMY_MUL = 1000
        GO_BOUNDARY_POS = 4
        
        heur = 0

        for enemy in self.get_opponents(game_state):
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos != None:
                distance_to_enemy = distance(game_state, self, enemy_pos)
                enemy_state = game_state.get_agent_state(enemy)
                is_dangerous = distance_to_enemy < 3 and not isPacmanByIndex(game_state, enemy) 
                is_dangerous = is_dangerous and enemy_state.scared_timer < (distance(game_state, self, enemy_pos) + 1)
                if is_dangerous:
                    heur += 1/ (distance_to_enemy+1) * DISTANCE_ENEMY_MUL

        heur -= 1/(self.distanceClosestBoundaryPos(game_state)+1) * GO_BOUNDARY_POS
                    
        return heur
    
#################  problems and heuristics  ####################

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        self.actions = None
        CaptureAgent.final(self, state)
        # print(self.weights)
        # did we finish training?

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)

        self.missingFoodLocation = None

        self.missingFoodCounter = 0

        self.mode = DefenceModes.Default

        # Missing food location tell us the direction of the pacman
        # and where we should head. I will by default try to intercept
        # him from the front
        self.prevMissingFoodLocation = []
        self.missingFoodLocation = None

        # A counter used to determine if the missingFoodPrevTarget
        # is valid. This is to reuse it in case 1 turn ago a food was
        # eaten and we still want to continue going towards the next food
        self.missingFoodCounter = 0
        # The below value is the initial value given to the counter 
        # when missing food is found
        self.initMissingFoodCounter = 4

        self.possibleEnemyEntryPositions = None
        self.boundaryPositions = None

        self.boundaryGoalPosition = None

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

        self.past_pos = []

        self.actions = []

        self.last_seen_enemy_pos = None



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
    #@timeit
    def choose_action(self, game_state):
        for enemy in self.get_opponents(game_state):
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos != None and isPacmanByIndex(game_state, enemy) and distance(game_state, self, enemy_pos) <= 3:
                my_teammate_pos = game_state.get_agent_position(self.get_team(game_state)[0])
                if my_teammate_pos != None:
                    distance_teammate_to_enemy = self.get_maze_distance(my_teammate_pos, enemy_pos)
                else: 
                    distance_teammate_to_enemy = 10

                if distance(game_state, self, enemy_pos) < distance_teammate_to_enemy or distance(game_state, self, enemy_pos) <3:
                    return self.choose_action_defensive(game_state)
            
            
        return self.choose_action_aggressive(game_state)


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
        self.goal = None
        

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
    
    def isGoalStateEatingFood(self, data):
        game_state = data['game']

        currentAgent = game_state.get_agent_state(data['agent'].index)
        
        prev_game_state = data['prev_game']
        if prev_game_state != None:
            prev_agent_state = prev_game_state.get_agent_state(data['agent'].index)
            return didAgentEatFood(prev_agent_state, currentAgent)
        return False
            
    def isGoalStateReturnSafe(self, data):
        return data['pos'] in data['agent'].boundary_pos

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
            problem.goal = data['pos']
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
    #print('No path found')

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

    def choose_action(self, game_state):
        # If we see an enemy or some food goes mising or a goal is reached we need to 
        # recalculate
        agentState = game_state.get_agent_state(self.index)
        isScared = agentState.scared_timer > 0

        if isScared:
            return self.choose_action_aggressive(game_state)
        
        return self.choose_action_defensive(game_state)
    

class DefenceModes(Enum):
    # Default mode is set to Pinky
    Default = 1
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
        self.myPreciousFood = self.captureAgent.get_food_you_are_defending(startingGameState)
        self.myPreciousCapsule = self.captureAgent.get_capsules_you_are_defending(startingGameState)

        if self.captureAgent.red:
            self.boundary = int(self.gridWidth / 2) - 1
            closestBorderFood = max(self.myPreciousFood, key=lambda x: x[0])[0]
        else:
            self.boundary = int(self.gridWidth / 2)
            closestBorderFood = min(self.myPreciousFood, key=lambda x: x[0])[0]
            
        (self.captureAgent.boundaryPositions,
         self.captureAgent.possibleEnemyEntryPositions) = self.getViableBoundaryPositions(0)
        self.captureAgent.defencePositions = self.captureAgent.boundaryPositions
        offset = 1

        if self.captureAgent.red:
            while len(self.captureAgent.defencePositions) > self.gridWidth/2 and \
                self.gridWidth - offset >= closestBorderFood:
                offset += 1
                (self.captureAgent.defencePositions,
                self.captureAgent.possibleEnemyEntryPositions) = \
                      self.getViableBoundaryPositions(offset)
        else:
            while len(self.captureAgent.defencePositions) > self.gridWidth/2 and \
                self.gridWidth + offset <= closestBorderFood:
                offset += 1
                (self.captureAgent.defencePositions,
                self.captureAgent.possibleEnemyEntryPositions) = \
                      self.getViableBoundaryPositions(offset)


        self.GOAL_POSITION = self.getGoalPosition()
        self.goalDistance = self.captureAgent.get_maze_distance(self.GOAL_POSITION,
                                                             self.agentPosition)

    def boundaryIsViable(self, boundryWidth, offset, nextOffset, boundryHeight):
        """Checks if a boundry can be used to enter our territory. In other words
         there are no walls stopping an enemy pacman from entering or exiting """
        positionNotAWall = not(self.walls[boundryWidth + offset][boundryHeight]) 
        adjacentPositionNotAWall = not(self.walls[boundryWidth + nextOffset][boundryHeight])
        return positionNotAWall and adjacentPositionNotAWall

    # offset is used to check not only boundry positions
    def getViableBoundaryPositions(self, offset):
        myPos = self.startingGameState.get_agent_position(self.captureAgent.index)
        boundary = self.boundary
        boundaryPositions = []
        enemyEntryPositions = []

        for boundaryHeight in range(0, self.gridHeight):
            isRed = self.captureAgent.red
            # for red is +1 and for blue -1
            # We want to get previous or next cell depending on the team
            nextBoundryWidthOffset = isRed - (not isRed)
            # multiplied by nextBoundryWidhtOffset and (-1) because we want 
            # the offset to go to our territory
            if self.boundaryIsViable(boundary, (-1) * nextBoundryWidthOffset * offset,
                                nextBoundryWidthOffset,
                                boundaryHeight):
                if (boundary, boundaryHeight) != myPos:
                    boundaryPositions.append((boundary, boundaryHeight))
                enemyEntryPositions.append((boundary+nextBoundryWidthOffset,
                                        boundaryHeight))

        return (boundaryPositions, enemyEntryPositions)

    def getTargetBasedOnEnemyPosition(self, enemyPosition, see_enemy = True):
        """
        The idea of this method is to find the closest food to the
        enemy position. Then from that position find the next
        closest food excluding the one we already found and so on until
        the NthClosestFood which we return. If we reach the 
        final food before that we return it
        """
        if see_enemy and len(self.myPreciousCapsule) > 0:
            # probable next position is capsule
            closestCapsulePositions = self.captureAgent.closestPosition(
            enemyPosition, self.myPreciousCapsule)
            return closestCapsulePositions.pop()
            
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

    def getRandomDefencePos(self):
        return random.choice(self.captureAgent.defencePositions)

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

    def getClosestBoundary(self):
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
            return self.getClosestBoundary()

        # If an enemy agents position is know calculate target
        # If not approximate possible targets 
        for enemy in self.enemies:
            if self.startingGameState.get_agent_state(enemy).is_pacman:
                if self.startingGameState.get_agent_position(enemy) != None:
                    enemyPosition = self.startingGameState.get_agent_position(enemy)
                    return enemyPosition
                else:
                    return self.getGoalPositionIfEnemyLocationUnknown()

        # If no enemy agent is a pacman try to take some food
        return self.getRandomDefencePos()

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

        return self.getProbableEnemyEntryPointBasedOnFood()

    
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
    
    ########################## Other Usefull methods #######################à

def is_threat(game_state, my_agent, enemy_idx, threat_distance):
    enemy_pos = game_state.get_agent_position(enemy_idx)

    # Also possible to use self.get_agent_distances for fuzzy estimate of
    # distance without knowing the enemy_pos
    if enemy_pos == None:
        return False
    isGhost = not game_state.get_agent_state(enemy_idx).is_pacman
    scaredTimer = game_state.get_agent_state(enemy_idx).scared_timer

    # Control if enemy is a ghost which we can not kill
    if isGhost and scaredTimer < distance(game_state, my_agent, enemy_pos):
        return distance(game_state, my_agent, enemy_pos) < threat_distance
    
    return False

def closestEnemy(game_state, my_agent):
    """
    Return the index of the closest enemy
    """
    positionsSorted = util.PriorityQueue()

    for enemy in my_agent.get_opponents(game_state):
        
        enemy_pos = game_state.get_agent_position(enemy)
        if enemy_pos != None:
            positionsSorted.push(
                enemy, distance(game_state, my_agent, enemy_pos))

    if positionsSorted.isEmpty():
        return None
    
    closestEnemy = positionsSorted.pop()
    return closestEnemy


def distance(game_state, agent, toPos):
    my_pos = game_state.get_agent_position(agent.index)
    return agent.get_maze_distance(my_pos, toPos)

def distanceFromEnemy(game_state, agent, enemy):
    my_pos = game_state.get_agent_position(agent.index)
    enemy_pos = game_state.get_agent_position(enemy)
    if enemy_pos != None:
        return agent.get_maze_distance(my_pos, enemy_pos)
    
def closeThreatEnemies(game_state, my_agent):
    """
    Return the priority queue with enemies
    """
    positionsSorted = util.PriorityQueue()

    for enemy in my_agent.get_opponents(game_state):
        if is_threat(game_state, my_agent, enemy):
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos != None:
                positionsSorted.push(
                    enemy, distance(game_state, my_agent, enemy_pos))
            
    if positionsSorted.isEmpty(): positionsSorted.push(None, 0)
    return positionsSorted


def getPos(game_state, agent):
    return game_state.get_agent_position(agent.index)

def isPacman(game_state, agent):
    return game_state.get_agent_state(agent.index).is_pacman

def isPacmanByIndex(game_state, agentIndex):
    return game_state.get_agent_state(agentIndex).is_pacman

def isGhost(game_state, agent):
    return not game_state.get_agent_state(agent.index).is_pacman

def isGhostByIndex(game_state, agentIndex):
    return not game_state.get_agent_state(agentIndex).is_pacman

def scaredTimerByIndex(game_state, agentIndex):
    return game_state.get_agent_state(agentIndex).scared_timer

def thereIsAThreat(game_state, my_agent, distance):
        if game_state != None:

            agent_state = game_state.get_agent_state(my_agent.index)
    
            return agent_state != None and any(is_threat(game_state, my_agent,
                                 enemy, distance) for enemy in my_agent.get_opponents(game_state))
    
def didAgentReturnFood(prev_agent_state, agent_state):
    return prev_agent_state != None and prev_agent_state.num_returned < agent_state.num_returned
    
def didAgentEatFood(prev_agent_state, agent_state):
    return prev_agent_state != None and prev_agent_state.num_carrying < agent_state.num_carrying
    
def isEnemyCloserToExitOfDeadEnd(game_state, agent):
    currentPos = game_state.get_agent_position(agent.index)
    enemy = closeThreatEnemies(game_state, agent).pop()
    if enemy != None:
        enemy_pos = game_state.get_agent_position(enemy)
        closer_exit_from_dead_end = agent.nearest_exit_from_ends[currentPos]
        distance_enemy_to_exit_from_dead_end = agent.get_maze_distance(enemy_pos, closer_exit_from_dead_end)
        distance_to_exit_from_dead_end = agent.get_maze_distance(currentPos, closer_exit_from_dead_end)
        if distance_enemy_to_exit_from_dead_end > (distance_to_exit_from_dead_end + 1):
            return False
        return True
    return False

def getClosestFood(game_state, agent):
    """Return minimum distance to food and its position"""
    food_list = agent.get_food(game_state).as_list()
    my_pos = game_state.get_agent_position(agent.index)
    return min(food_list, key=lambda food: agent.get_maze_distance(my_pos, food))

def getClosestCapsule(agent):
    capsules = agent.get_capsules()
    my_pos = agent.get_position()
    return min(capsules, key=lambda cap: agent.get_maze_distance(my_pos, cap))

def needToCalculate(game_state, prev_game_state, agent):
    my_pos = game_state.get_agent_position(agent.index)
    if prev_game_state == None:
        return True
    
    if agent.actions == None or agent.actions == []:
        return True
    
    if thereIsAThreat(game_state, agent, 3):
        return True
    
    closest_food = getClosestFood(game_state, agent)
    if closest_food in agent.dead_ends:
        distance_to_food = distance(game_state, agent, closest_food)
        distance_food_to_exit = agent.get_maze_distance(my_pos, agent.nearest_exit_from_ends[closest_food])
        closer_enemy_index = closestEnemy(game_state, agent)
        if closer_enemy_index != None:
            enemy_pos = game_state.get_agent_position(closer_enemy_index)
            distance_ghost_to_exit = agent.get_maze_distance(enemy_pos, agent.nearest_exit_from_ends[closest_food])
            if distance_to_food + distance_food_to_exit + 1 < distance_ghost_to_exit:
                return True
        
    
    if len(agent.get_food(game_state).as_list()) != len(agent.get_food(prev_game_state).as_list()):
        return True
    
    if len(agent.get_capsules(game_state)) != len(agent.get_capsules(prev_game_state)):
        return True
    
    if my_pos == agent.start:
        return True
    
    if game_state.data.timeleft < 40:
        return True
    
    return False

def getRandomSafeAction(game_state, agent):
    my_actions = game_state.get_legal_actions(agent.index)
    current_pos = game_state.get_agent_position(agent.index)
    secure_actions = []
    safe_actions = []

    enemy_positions = [game_state.get_agent_position(enemy) for enemy in agent.get_opponents(game_state) if game_state.get_agent_position(enemy) != None]
        
    for next_action in my_actions:
        fut_pos = Actions.get_successor(current_pos, next_action)
            
        if not fut_pos in enemy_positions:
            safe_actions.append(next_action)
            if all([agent.get_maze_distance(current_pos, enemy_pos) < agent.get_maze_distance(fut_pos, enemy_pos) for enemy_pos in enemy_positions]):
                secure_actions.append(next_action)
        
    
    if safe_actions == [] and secure_actions == []:
        return random.choice(game_state.get_legal_actions(agent.index))
    
    if safe_actions == []:
        return random.choice(secure_actions)
    
    return random.choice(safe_actions)
