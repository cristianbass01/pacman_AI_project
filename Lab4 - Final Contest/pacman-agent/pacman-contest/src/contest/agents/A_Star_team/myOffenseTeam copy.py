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
from numpy import mean
from contest.util import nearestPoint

# Standard imports
from contest.captureAgents import CaptureAgent
import random
from contest.game import Directions, Actions  # basically a class to store data


# this is the entry points to instanciate you agents
def create_team(first_index, second_index, is_red,
                first='offensiveAgent', second='DefensiveReflexAgent', num_training=0):

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
        agent_state = game_state.get_agent_state(self.index)

        exist_threat = thereIsAThreat(game_state, self)
        prev_game_state = self.get_previous_observation()
        if prev_game_state != None:
            used_to_be_threat = thereIsAThreat(prev_game_state, self)
        else: 
            used_to_be_threat = False

        current_food = len(self.get_food(game_state).as_list())
        food_changed = self.prev_food > current_food
        is_carrying_food = agent_state.num_carrying > 0
        start = self.start == game_state.get_agent_position(self.index)
        no_moves = self.next_moves == None or (self.next_moves != None and len(self.next_moves) == 0)
        
        problem = FoodOffense(startingGameState=game_state, captureAgent=self)
        actions = None

        if exist_threat and isPacman(game_state, self):
            if is_carrying_food:
                print('Using returnSafeHeuristic with isGoalStateReturnSafeOrEatingCapsule')
                actions = aStarSearch(problem, heuristic=self.returnSafeHeuristic, agent=self, goal=problem.isGoalStateReturnSafeOrEatingCapsule)
            else: 
                print('Using eatingFoodHeuristicWithAwareness with isGoalStateEatingFoodWithAwareness')
                actions = aStarSearch(problem, heuristic=self.eatingFoodHeuristicWithAwareness, agent=self, goal=problem.isGoalStateEatingFoodWithAwareness)
        elif start or used_to_be_threat:
            print('Using eatingFoodHeuristic with isGoalStateEatingFood')
            actions = aStarSearch(problem, heuristic=self.eatingFoodHeuristic, agent=self, goal=problem.isGoalStateEatingFood)

        elif no_moves or food_changed:
            if food_changed:
                self.prev_food = len(self.get_food(game_state).as_list())

            print('Using eatingFoodHeuristic with isGoalStateEatingFood')
            actions = aStarSearch(problem, heuristic=self.eatingFoodHeuristic, agent=self, goal=problem.isGoalStateEatingFood)
        
        if actions != None:
            self.next_moves = actions

        # this can occure if start in the goal state. In this case do not want to perform any action.
        if self.next_moves == [] or self.next_moves == None:
            self.next_moves = ["Stop"]

        next_action = self.next_moves[0]
        self.next_moves = self.next_moves[1:]
        return next_action
    
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
    
    def eatingFoodHeuristic(self, state):
        # MIN HEURISTIC:
        # -11 if there was food in the current pos
        # ascending following the distance to the food
        game_state = state[0]
        DISTANCE_FOOD_MUL = 10

        offence_food_pos = self.get_food(game_state).as_list()
        current_pos = game_state.get_agent_position(self.index)
        
        prev_game_state = self.get_previous_observation()
        if prev_game_state != None:
            if prev_game_state.has_food(current_pos[0], current_pos[1]):
                return - (DISTANCE_FOOD_MUL + 1)

        distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
        return - 1 / (distanceClosestFood+1) * DISTANCE_FOOD_MUL
    
    
    def eatingFoodHeuristicWithAwareness(self, state):
        game_state = state[0]
        
        heur = self.eatingFoodHeuristic(state)

        heur += self.returnSafeHeuristic(state)

        currentPos = game_state.get_agent_position(self.index)
        
        is_dead_end = currentPos in self.dead_ends

        if is_dead_end:
            if isEnemyCloserToExitOfDeadEnd(game_state, self):
                heur += 100

        return heur

    def returnSafeHeuristic(self, state):
        game_state = state[0]
        DISTANCE_ENEMY_MUL = 10
        EATING_MORE_FOOD_MUL = 2
        GO_BOUNDARY_POS = 4
        
        heur = 0

        for enemy in self.get_opponents(game_state):
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos != None:
                distance_to_enemy = distance(game_state, self, enemy_pos)
                enemy_state = game_state.get_agent_state(enemy)
                is_dangerous = distance_to_enemy < 3 and not isPacmanByIndex(game_state, enemy) 
                is_dangerous = is_dangerous and enemy_state.scared_timer < (distance(game_state, self, enemy_pos) + 2)
                if is_dangerous:
                    heur += 1/distance_to_enemy * DISTANCE_ENEMY_MUL

        heur -= self.distanceClosestBoundaryPos(game_state) * GO_BOUNDARY_POS
                    
        return heur
    
    def offensiveHeuristic(self, state):
        game_state = state[0]

        my_pos = game_state.get_agent_position(self.index)
        prev_game_state = self.get_previous_observation()

        offence_food_pos = self.get_food(game_state).as_list()
        defence_food_pos = self.get_food_you_are_defending(game_state).as_list()

        if prev_game_state != None:
            prev_offence_food_pos = self.get_food(prev_game_state).as_list()
            prev_defence_food_pos = self.get_food_you_are_defending(prev_game_state).as_list()
        else: 
            prev_defence_food_pos = defence_food_pos
            prev_offence_food_pos = offence_food_pos

        offence_capsule_pos = self.get_capsules(game_state)
        defence_capsule_pos = self.get_capsules_you_are_defending(game_state)
        
        current_score = self.get_score(game_state)
        
        #Usefull
        # self.is_pacman
        # self.scared_timer
        # self.num_carrying
        # self.num_returned
        ### TODO ADD DISTRIBUTIONS
        
        # Pacman
        FOOD_CARRYING_MUL = 2
        FOOD_RETURNED_MUL = 1
        EATING_CAPSULE_MUL = 10
        RETURN_HOME_ALL_FOOD_MUL = 10
        RETURN_HOME_CHASED_MUL = 1
        EAT_ENEMY_MUL = 10
        RETURN_HOME_ENOUGH_FOOD_MUL = 1
        ENEMY_PENALTY_MUL = 1
        TEAMMATE_PENALTY_MUL = 5
        SCORE_MUL = 30
        DEAD_MUL = 100
        
        # Not pacman
        DISTANCE_FOOD_MUL = 5
        DISTANCE_CLOSER_ENEMY_MUL = 1
        CLOSE_TO_DANGER_POS_MUL = 20
        DISTANCE_TO_DANGER_POS_MUL = 5

        # Worst if it dies
        
        agent_state = game_state.get_agent_state(self.index)

        if my_pos == self.start:
            return -10 * DEAD_MUL
        
        if isPacman(game_state, self):

            # If being chased the heuristic is based on the distance of the capsule or home    
            reward_chased = 0
            if any(is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)):
                # TODO check if we can make this part faster cause when I see
                # enemies the game slows down due to recalculating constantly

                # So I am being chased: 
                # The reward should be defined by the distance to the capsules, distance to home and distance to enemy
                distToClosestEnemy = distanceClosestEnemy(game_state, self)

                for capsule in offence_capsule_pos:  
                    distToCapsule = distance(game_state, self, capsule)
                    canKill = distToCapsule < distToClosestEnemy
                    reward_chased -= canKill * EATING_CAPSULE_MUL * distToCapsule # Adjust the reward value as needed
                
                # Distance home. For some reason crashes here. idk why TODO FIX THIS
                # reward_chased -= min([distance(game_state, self, border_pos) for border_pos in self.boundary_pos]) * RETURN_HOME_CHASED_MUL

                # Distance closest enemy
                # TODO This I think should be reworked based on if we want to kill the enemy
                reward_chased += distToClosestEnemy * DISTANCE_CLOSER_ENEMY_MUL

                # Mean distance to all enemies TODO maybe add this??
                # We are slow currently so maybe we should simplify
                # reward_chased

                # If in danger the only scope is to return home or eat a capsule, no more food
                return reward_chased

            # If the enemy is scared and the time to reach him is enough I can try also to kill him (but it is dangerous to follow him)
            reward_enemy_scared = 0
            closestEnemyAgent, closestEnemyDistance = closestEnemy(game_state, self)
            isScared = closestEnemyAgent.scared_timer != 0
            if isScared and closestEnemyAgent.scared_timer <= closestEnemyDistance:
                reward_enemy_scared -= closestEnemyDistance * EAT_ENEMY_MUL

            # Reward for eating food
            #food_reward = agent_state.num_carrying * FOOD_CARRYING_MUL

            # Reward for returning food
            #food_reward += agent_state.num_returned * FOOD_RETURNED_MUL

            # I want to be less distant from food
            distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
            food_reward = -distanceClosestFood * DISTANCE_FOOD_MUL

            # If I am carrying a lot of food, maybe I should do something else (returning home)
            food_reward -= agent_state.num_carrying * FOOD_CARRYING_MUL

            # Reward for returning home once having all the food for win
            home_reward = 0
            if len(offence_food_pos) <= 2:
                home_reward -= min([distance(game_state, self, border_pos) for border_pos in self.boundary_pos]) * RETURN_HOME_ALL_FOOD_MUL

            teammate_penalty = 0
            ## Penalty for being closer to a teammate while on offense
            for teammate in self.get_team(game_state):
                if game_state.get_agent_state(teammate).is_pacman:
                    teammate_pos = game_state.get_agent_position(teammate)
                    teammate_penalty += distance(game_state, self, teammate_pos) * TEAMMATE_PENALTY_MUL

            # Reward based on the score MAYBE DON'T NEEDED
            #score_reward = current_score * SCORE_MUL

            # Penalty for being closer to an enemy while on offense ALREADY IN BEING CHASED
            #enemy_penalty = sum(self.get_maze_distance(my_pos, enemy) for enemy in enemy_positions) * ENEMY_PENALTY_MUL

            # Combine rewards and penalties to form the heuristic
            heuristic = food_reward + home_reward + teammate_penalty + reward_enemy_scared
        else:
            reward_not_pacman = 0  

            # If not pacman means that it is in the safe zone
            # If it is in the safe zone, the scope will be to return to the enemy zone and eat!
            distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
            reward_not_pacman -= distanceClosestFood * DISTANCE_FOOD_MUL
              
            # Before returning to the enemy zone I need to see if there are other ghosts near me that can kill me or chased me
            if any(is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)):
                # Distance closest enemy
                reward_not_pacman -= distanceClosestEnemy(game_state, self) * DISTANCE_CLOSER_ENEMY_MUL
            
            teammate_penalty = 0
            ## Penalty for being closer to a teammate while on defence
            for teammate in self.get_team(game_state):
                if not game_state.get_agent_state(teammate).is_pacman:
                    teammate_pos = game_state.get_agent_position(teammate)
                    teammate_penalty += distance(game_state, self, teammate_pos) * TEAMMATE_PENALTY_MUL

            # Better if I will stay closer to the border in order to pass (but not necessary in order to 'go after the hill' if necessary)
            # Below currently not working TODO FIX
            #distanceToClosestDangerPos = min([distance(game_state, self, danger_pos) for danger_pos in self.danger_boundary])
            # reward_not_pacman -= distanceToClosestDangerPos * DISTANCE_TO_DANGER_POS_MUL
            
            heuristic = reward_not_pacman + teammate_penalty
        
        # I want to be less distant from food
        #distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offence_food_pos])
        #heuristic = -distanceClosestFood * 0.1
        return heuristic
    
def is_threat(game_state, my_agent, enemy_idx):
    enemy_pos = game_state.get_agent_position(enemy_idx)
    # Also possible to use self.get_agent_distances for fuzzy estimate of
    # distance without knowing the enemy_pos (| NIKI) TODO
    if enemy_pos == None:
        return False
    isGhost = not game_state.get_agent_state(enemy_idx).is_pacman
    #isScared = game_state.get_agent_state(enemy_agent).is_scared
    scaredTimer = game_state.get_agent_state(enemy_idx).scared_timer

    # Control if enemy is a ghost which we can not kill
    MIN_DISTANCE_CONSIDERED_THREAT = 3
    if isGhost and scaredTimer <= (distance(game_state, my_agent, enemy_pos) + 2):
        if distance(game_state, my_agent, enemy_pos) < MIN_DISTANCE_CONSIDERED_THREAT:
            return False
        return True
    
    return False

def getCost(my_agent, my_pos, enemy_pos, maxCost = 30):
    if enemy_pos != None:
        return my_agent.get_maze_distance(my_pos, enemy_pos)
    return maxCost

def closestEnemy(game_state, my_agent):
    """
    Return the state of the closest enemy
    """
    positionsSorted = util.PriorityQueue()
    my_pos = game_state.get_agent_position(my_agent.index)

    for enemy in my_agent.get_opponents(game_state):
        
        enemy_pos = game_state.get_agent_position(enemy)
        positionsSorted.push(
            enemy, getCost(my_agent, my_pos, enemy_pos))

    closestEnemy = positionsSorted.pop()
    enemy_pos = game_state.get_agent_position(closestEnemy)
    enemy_state = game_state.get_agent_state(closestEnemy)
    return enemy_state, getCost(my_agent, my_pos, enemy_pos)


def closeThreatEnemies(game_state, my_agent):
    """
    Return the priority queue with enemies
    """
    positionsSorted = util.PriorityQueue()
    my_pos = game_state.get_agent_position(my_agent.index)

    for enemy in my_agent.get_opponents(game_state):
        if is_threat(game_state, my_agent, enemy):
            enemy_pos = game_state.get_agent_position(enemy)
            positionsSorted.push(
                enemy, getCost(my_agent, my_pos, enemy_pos))
    if positionsSorted.isEmpty(): positionsSorted.push(None, 0)
    return positionsSorted

def getPos(game_state, agent):
    return game_state.get_agent_position(agent.index)

def isPacman(game_state, agent):
    return game_state.get_agent_state(agent.index).is_pacman

def isPacmanByIndex(game_state, agentIndex):
    return game_state.get_agent_state(agentIndex).is_pacman

def distance(game_state, agent, toPos):
    my_pos = game_state.get_agent_position(agent.index)
    return agent.get_maze_distance(my_pos, toPos)

def distanceClosestEnemy(game_state, my_agent):
    _, closestEnemyDistance = closestEnemy(game_state, my_agent)
    return closestEnemyDistance

def thereIsAThreat(game_state, my_agent):
        if game_state != None:

            agent_state = game_state.get_agent_state(my_agent.index)
    
            return agent_state != None and any(is_threat(game_state, my_agent,
                                 enemy) for enemy in my_agent.get_opponents(game_state))
    
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
        self.startingGameState = startingGameState
        # Need to ignore previous score change, as everything should be considered relative to this state
        self.startingGameState.data.score_change = 0
        self.MINIMUM_IMPROVEMENT = 1
        self.DEPTH_CUTOFF = 20
        # WARNING: Capture agent doesn't update with new state, this should only be used for non state dependant utils (e.g distancer)
        self.my_agent = captureAgent
        self.goal_state_found = None

    def getStartState(self):
        # This needs to return the state information to being with
        return (self.startingGameState, self.startingGameState.get_score(), None)
        
    
    def isGoalStateEatingFood(self, state, agent = None):
        game_state = state[0]

        currentAgent = game_state.get_agent_state(agent.index)
        
        prev_game_state = state[2]
        if prev_game_state != None:
            prev_agent_state = prev_game_state.get_agent_state(agent.index)
            return didAgentEatFood(prev_agent_state, currentAgent)
        return False
    
    def isGoalStateEatingFoodWithAwareness(self, state, agent = None):
        game_state = state[0]

        currentAgent = game_state.get_agent_state(agent.index)
        currentPos = game_state.get_agent_position(agent.index)
        
        prev_game_state = state[2]
        if prev_game_state != None:
            prev_agent_state = prev_game_state.get_agent_state(agent.index)
            is_dead_end = currentPos in agent.dead_ends
            is_chased = thereIsAThreat(game_state, agent)

            if not is_chased:
                return didAgentEatFood(prev_agent_state, currentAgent)
            elif is_dead_end:
                if isEnemyCloserToExitOfDeadEnd(game_state, agent):
                    return didAgentEatFood(prev_agent_state, currentAgent)
        return False
    
    def isGoalStateReturnSafeOrEatingCapsule(self, state, agent = None):
        current_game_state = state[0]
        current_pos = current_game_state.get_agent_position(agent.index)

        if self.startingGameState.get_score() < current_game_state.get_score():
            return True
        elif current_pos in agent.boundary_pos:
            return True
        
        prevCapsules = agent.get_capsules(self.startingGameState)

        hasEatenCapsule = current_pos in prevCapsules
        if hasEatenCapsule:          
            return True
        
        return False


    def isGoalState(self, state, agent = None):
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        # Goal state when:
        # - Pacman is in our territory
        # - has eaten x food: This comes from the score changing
        # these are both captured by the score changing by a certain amount

        # Note: can't use CaptureAgent, at it doesn't update with game state
        game_state = state[0]

        agent_state = game_state.get_agent_state(self.my_agent.index)
        currentPos = game_state.get_agent_position(self.my_agent.index)

        # Try to get previous game_state (if exists)
        prev_game_state = state[2]
        prev_agent_state = None
        if prev_game_state != None:
            prev_agent_state = prev_game_state.get_agent_state(self.my_agent.index)

        # Being eaten is always not a goal state
        # Calculating the distance from the past position and the current position to see if I am being killed
        if prev_agent_state != None:
            pastPos = prev_game_state.get_agent_position(self.my_agent.index)
            distancePastToCurrentPos = self.my_agent.get_maze_distance(pastPos, currentPos)
            # I can cover a distance more than 1 only if I have being eaten
            if distancePastToCurrentPos > 2:
                return False


        if isPacman(game_state, self.my_agent):

            closestEnemyAgent, closestEnemyDistance = closestEnemy(game_state, self.my_agent)
            isScared = closestEnemyAgent.scared_timer != 0

            # If being chased my goal is not being chased anymore (so eating a capsule or return home)
            # So if there is a threat, I'm not in the goal state    
            if thereIsAThreat(game_state, self.my_agent):
                if not closestEnemyDistance > 5:
                    # If the enemy is closer than 5 squares mean that he can follow me
                    # This is not a goal state
                    return False
                    

            # If the enemy is scared and the time to reach him is enough I can try also to kill him (but it is dangerous to follow him)
            if isScared and closestEnemyAgent.scared_timer <= closestEnemyDistance:
                # If it is scared and the distance is not enough to kill him it is not a goal
                return False

            # If there is no threat my goal is to eat or return home to collect the food that I am carrying
            # TODO I think these depends on how far we are from the boundary, so it will depend on the heuristic
            # But carrying more food or having returned more food is a goal state
            if didAgentEatFood(prev_agent_state, agent_state):
                return True
                
            if didAgentReturnFood(prev_agent_state, agent_state):
                return True
        else:
            # If not pacman means that it is in the safe zone
            # The only reason to return to the safe zone if to escape from an agent or to collect food
            if thereIsAThreat(prev_game_state, self.my_agent):
                return True
            
            if didAgentReturnFood(prev_agent_state, agent_state):
                return True
            
        return False
        
    
    def enemyVisible(self, enemy_idx, gameState):
        return gameState.get_agent_position(enemy_idx) is not None

    def enemyAdjacent(self, agentPos, enemyPos, game_state):
        return self.captureAgent.get_maze_distance(agentPos,
                         game_state.get_agent_state(enemyPos).get_position()) <= 1
    
    def enemyDangerous(self, enemyIndex, game_state):
        enemyState = game_state.get_agent_state(enemyIndex)
        enemyNotScared = enemyState.scared_timer <= 0
        return (not isPacmanByIndex(game_state, enemyIndex)) and enemyNotScared
    
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
        current_pos = game_state.get_agent_position(self.captureAgent.index)
        my_actions = game_state.get_legal_actions(self.captureAgent.index)
        # not interested in exploring the stop action as the state will be the same as our
        # current one.
        if Directions.STOP in my_actions:
            my_actions.remove(Directions.STOP)
        
        next_game_states = [game_state.generate_successor(self.captureAgent.index, action)      
                                for action in my_actions]
        
        # we are only concerned about being eaten when we are pacman
        if isPacman(game_state, self.my_agent):
            # If in the next state you are next to a ghost assume it will eat you
            # if legal
            for i, next_game_state in enumerate(next_game_states):
                
                next_pos = next_game_state.get_agent_position(self.captureAgent.index)
                isKilled = self.my_agent.get_maze_distance(current_pos, next_pos) > 1
                
                if isKilled:
                    next_game_states.remove(next_game_state)
                    my_actions.remove(my_actions[i])
                    continue
                
                for enemy in self.my_agent.get_opponents(next_game_state):
                    if is_threat(next_game_state, self.my_agent, enemy):
                        min_enemy_distance = float('inf')
                        best_enemy_action = None

                        for enemy_action in next_game_state.get_legal_actions(enemy):
                            next_enemy_game_state = next_game_state.generate_successor(enemy, enemy_action)

                            next_enemy_pos = next_enemy_game_state.get_agent_position(enemy)
                        
                            if next_enemy_pos != None:
                                # Enemy can still see me
                                current_enemy_distance = distance(next_enemy_game_state, self.my_agent, next_enemy_pos)

                                if current_enemy_distance < min_enemy_distance:
                                    min_enemy_distance = current_enemy_distance
                                    best_enemy_action
                        
                        if best_enemy_action != None:
                            next_game_states[i] = next_game_state.generate_successor(enemy, best_enemy_action)
        
        # Why do we do (next_game_state,) and not just (next_game_state)?
        successors = [((next_game_state, 0, game_state), action, 1)
                      for action, next_game_state in zip(my_actions, next_game_states)]

        return successors

################# Search Algorithems ###################




def aStarSearch(problem, heuristic=None, agent = None, goal = None):
    """Search the node that has the lowest combined cost and heuristic first."""
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []
    #MAX_DEPTH = 25
    # use a priority queue to store states
    frontier = util.PriorityQueue()

    if goal == None: goal = problem.isGoalState
    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(),  [], 0)

    distributions = [util.Counter()]
    
    # as a priority there will be the total cost + heuristic
    # as first node this is irrelevant
    frontier.push(start_node, 0)
    while not frontier.isEmpty():
        (node, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        if goal(node, agent = agent) :
            # Last heuristic
            heur = heuristic(node)
            my_pos = node[0].get_agent_position(agent.index)
            distributions[0][my_pos] = heur

            showAndNormalize([distributions[0].copy()], agent)

            return path

        # the algorithm control if the node is being expanded before
        if node[:1] not in expanded_nodes:
            expanded_nodes.append(node[:1])

            # if not the algorithm search in his successor and insert them in the frontier to be expanded
            for (child, n_action, n_cost) in problem.get_successors(node, path):
                if child[:1] not in expanded_nodes:
                    # fut_cost must be passed and represent the cost to reach that position
                    fut_cost = cost + n_cost

                    # total cost is the fut cost + heuristic and is passed as the priority
                    heur = heuristic(child)
                    total_cost = cost + n_cost + heur
                    total_path = path + [n_action]

                    my_pos = child[0].get_agent_position(agent.index)
                    distributions[0][my_pos] = heur
                    showAndNormalize([distributions[0].copy()], agent)
                    
                    frontier.push((child, total_path, fut_cost), total_cost)

def showAndNormalize(distributions, agent):
    distributions[0].incrementAll(distributions[0].keys(), -min(distributions[0].values()))
    distributions[0].normalize()
    agent.display_distributions_over_positions(distributions)

########################## DEFENCEEEE ##################

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
