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
        
        self.gridHeight = None
        self.gridWidth = None
        self.start = None

        self.safe_boundary = None
        self.danger_boundary = None

        self.safe_pos = None
        self.danger_pos = None

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

        self.safe_pos = []
        for y in range(self.gridHeight):
            if not game_state.has_wall(self.safe_boundary, y):
                self.safe_pos += (self.safe_boundary, y)

        self.danger_pos = []
        for y in range(self.gridHeight):
            if not game_state.has_wall(self.danger_boundary, y):
                self.danger_pos += (self.danger_boundary, y)

class offensiveAgent(agentBase):

    def choose_action(self, game_state):
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action

        problem = FoodOffense(startingGameState=game_state, captureAgent=self)


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

        #prev_difend_food_pos = self.get_food(prev_game_state).as_list()
        #prev_offens_food_pos = self.get_food_you_are_defending(prev_game_state).as_list()

        offence_capsule_pos = self.get_capsules(game_state)
        defence_capsule_pos = self.get_capsules_you_are_defending(game_state)

        teammate_positions = [game_state.get_agent_position(agent) for agent in self.get_team(game_state) if agent != self.index]
        enemy_positions = [game_state.get_agent_position(agent) for agent in self.get_opponents(game_state)]

        
        current_score = self.get_score(game_state)
        
        #Usefull
        # self.is_pacman
        # self.scared_timer
        # self.num_carrying
        # self.num_returned
        ### TODO ADD DISTRIBUTIONS

        # Pacman
        FOOD_CARRYING_MUL = 1
        FOOD_RETURNED_MUL = 1
        EATING_CAPSULE_MUL = 10
        RETURN_HOME_ALL_FOOD_MUL = 10
        RETURN_HOME_CHASED_MUL = 1
        EAT_ENEMY_MUL = 10
        RETURN_HOME_ENOUGH_FOOD_MUL = 1
        ENEMY_PENALTY_MUL = 1
        TEAMMATE_PENALTY_MUL = 5
        SCORE_MUL = 30
        
        # Not pacman
        DISTANCE_FOOD_MUL = 5
        DISTANCE_CLOSER_ENEMY_MUL = 10
        CLOSE_TO_DANGER_POS_MUL = 20
        DISTANCE_TO_DANGER_POS_MUL = 5


        agent_state = game_state.get_agent_state(self.index)

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
                # reward_chased -= min([distance(game_state, self, border_pos) for border_pos in self.safe_pos]) * RETURN_HOME_CHASED_MUL

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
            food_reward = agent_state.num_carrying * FOOD_CARRYING_MUL

            # Reward for returning food
            food_reward += agent_state.num_returned * FOOD_RETURNED_MUL

            # Reward for returning home once having all the food for win
            home_reward = 0
            if len(offens_food_pos) <= 2:
                home_reward -= min([distance(game_state, self, border_pos) for border_pos in self.safe_pos]) * RETURN_HOME_ALL_FOOD_MUL

            teammate_penalty = 0
            ## Penalty for being closer to a teammate while on offense
            teammate_penalty = sum([distance(game_state, self, teammate) for teammate in teammate_positions]) * TEAMMATE_PENALTY_MUL

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
            distanceClosestFood = min([distance(game_state, self, food_pos) for food_pos in offens_food_pos])
            reward_not_pacman -= distanceClosestFood * DISTANCE_FOOD_MUL
              
            # Before returning to the enemy zone I need to see if there are other ghosts near me that can kill me or chased me
            if any(is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)):
                # Distance closest enemy
                reward_not_pacman -= distanceClosestEnemy(game_state, self) * DISTANCE_CLOSER_ENEMY_MUL
            
            # Better if I will stay closer to the border in order to pass (but not necessary in order to 'go after the hill' if necessary)
            # Below currently not working TODO FIX
            #distanceToClosestDangerPos = min([distance(game_state, self, danger_pos) for danger_pos in self.danger_boundary])
            # reward_not_pacman -= distanceToClosestDangerPos * DISTANCE_TO_DANGER_POS_MUL
            
            heuristic = reward_not_pacman

        return heuristic
    
def is_threat(game_state, my_agent, enemy_agent):
    enemy_pos = game_state.get_agent_position(enemy_agent)
    # Also possible to use self.get_agent_distances for fuzzy estimate of
    # distance without knowing the enemy_pos (| NIKI) TODO
    if enemy_pos == None:
        return False
    isGhost = not game_state.get_agent_state(enemy_agent).is_pacman
    #isScared = game_state.get_agent_state(enemy_agent).is_scared
    scaredTimer = game_state.get_agent_state(enemy_agent).scared_timer

    # Control if enemy is a ghost which we can not kill
    if isGhost and not scaredTimer <= distance(game_state, my_agent, enemy_pos):
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

def getPos(game_state, agent):
    return game_state.get_agent_position(agent.index)

def isPacman(game_state, agent):
    return game_state.get_agent_state(agent.index).is_pacman

def distance(game_state, agent, toPos):
    my_pos = game_state.get_agent_position(agent.index)
    return agent.get_maze_distance(my_pos, toPos)

def distanceClosestEnemy(game_state, my_agent):
    _, closestEnemyDistance = closestEnemy(game_state, my_agent)
    return closestEnemyDistance

    
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
        return (self.startingGameState, self.startingGameState.get_score())

    def isGoalState(self, state):
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        # Goal state when:
        # - Pacman is in our territory
        # - has eaten x food: This comes from the score changing
        # these are both captured by the score changing by a certain amount
 
        agent_state = game_state.get_agent_state(self.my_agent.index)

        # Note: can't use CaptureAgent, at it doesn't update with game state
        game_state = state[0]
        
        # Try to get previous game_state (if exists)
        prev_game_state = agent_state.get_previous_observation()
        prev_agent_state = None
        if prev_game_state != None:
            prev_agent_state = prev_game_state.get_agent_state(self.my_agent.index)

        # Being eaten is always not a goal state
        # Calculating the distance from the past position and the current position to see if I am being killed
        
        if prev_agent_state != None:
            pastPos = prev_agent_state.get_agent_position(self.my_agent.index)
            currentPos = game_state.get_agent_position(self.my_agent.index)
            distancePastToCurrentPos = distance()

        # If red team, want scores to go up
        if self.captureAgent.red == True:

            if isPacman(game_state, self):

                closestEnemyAgent, closestEnemyDistance = closestEnemy(game_state, self)
                isScared = closestEnemyAgent.scared_timer != 0

                # If being chased my goal is not being chased anymore (so eating a capsule or return home)
                # So if there is a threat, I'm not in the goal state    
                if any(is_threat(game_state, self, enemy) for enemy in self.get_opponents(game_state)):
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
                if prev_agent_state != None and prev_agent_state.num_carrying < agent_state.num_carrying:
                    return True
                
                if prev_agent_state != None and prev_agent_state.num_returned < agent_state.num_returned:
                    return True
            else:
                # If not pacman means that it is in the safe zone
                # If it is in the safe zone, the scope will be to return to the enemy zone and eat!
                # So my future goal is to increase the number of food that I am carrying
                

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

        actions = game_state.get_legal_actions(self.my_agent.index)

        # not interested in exploring the stop action as the state will be the same as out current one.
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        next_game_states = [game_state.generate_successor(self.my_agent.index, action) for action in actions]

        # if planning close to agent, include expected ghost activity
        current_depth_of_search = len(path)

        # we are only concerned about being eaten when we are pacman
        if current_depth_of_search <= self.DEPTH_CUTOFF and game_state.get_agent_state(self.my_agent.index).is_pacman:
            self.expanded += 1  # track number of states expanded

            # make any nearby enemy ghosts take a step toward you if legal
            for i, next_game_state in enumerate(next_game_states):
                # get enemys
                current_agent_index = self.my_agent.index
                enemy_indexes = self.my_agent.get_opponents(next_game_state)

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
        # This is the target gotten the previous time we found missing food
        self.missingFoodPrevTarget = None

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

class DefenceModes(Enum):
    # Default mode is set to Pinky
    Default = 2
    Clyde = 1
    Pinky = 2
    Blinky = 3


class defendTerritoryProblem():
    def __init__(self, startingGameState, captureAgent):
        self.NthClosestFood = 1
        self.expanded = 0
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

        (self.viableBoundaryPositions,
         self.possibleEnemyEntryPositions) = self.getViableBoundaryPositions()

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

    #def clyde(self):
    #    boundaryGoalPositions = self.closestPosition(
    #        self.intialPosition, self.viableBoundaryPositions)

    #    if self.captureAgent.boundaryGoalPosition == None:
    #        boundaryGoalPosition = boundaryGoalPositions.pop()
    #        self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
    #    else:
    #        if self.captureAgent.boundaryGoalPosition == self.intialPosition:
    #            boundaryGoalPosition = boundaryGoalPositions.pop()
    #            self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
    #        else:
    #            boundaryGoalPosition = self.captureAgent.boundaryGoalPosition
    #    return boundaryGoalPosition

    def getTargetBasedOnEnemyPosition(self, enemyPosition):
        """
        The idea of this method is to find the closest food to the
        enemy position. Then from that position find the next
        closest food excluding the one we already found and so on until
        the NthClosestFood which we return. If we reach the 
        final food before that we return it
        """
        closestFoodPositions = self.closestPosition(
            enemyPosition, self.myPreciousFood.as_list())
        closestFoodPosition = closestFoodPositions.pop()
        for _ in range(self.NthClosestFood - 1):
            if closestFoodPositions.isEmpty():
                return closestFoodPosition
            closestFoodPositions =  self.closestPositionWithPriorityQueue(
                                                closestFoodPosition,
                                                closestFoodPositions)
            closestFoodPosition = closestFoodPositions.pop()
        
        return closestFoodPosition

    def getClosestFood(self):
        """Return minimum distance to food and its position"""
        game_state = self.startingGameState
        food_list = self.captureAgent.get_food(game_state).as_list()
        agentPosition = self.agentPosition

        min_pos = None
        min_dist = 99999999
        for food in food_list:
            dist = self.captureAgent.get_maze_distance(agentPosition, food)
            if dist < min_dist:
                min_pos = food
                min_dist = dist

        return min_pos

    def getGoalPositionIfEnemyLocationUnknown(self):
        missingFoodPosition = self.getMissingFoodPosition()

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
            self.captureAgent.missingFoodCounter -= 1
            return self.captureAgent.prevMissingFoodTarget

        return self.getProbableEnemyEntryPointBasedOnFood()

    def getGoalPosition(self):
        isScared = self.startingGameState.get_agent_state(
            self.captureAgent.index).scared_timer > 0

        if isScared:
            self.captureAgent.mode = DefenceModes.Clyde
        else:
            self.captureAgent.mode = DefenceModes.Default

        # If an enemy agents position is know calculate target
        # If not approximate possible targets 
        for enemy in self.enemies:
            if self.startingGameState.get_agent_state(enemy).is_pacman:
                self.captureAgent.enemyEntered = True
                if self.startingGameState.get_agent_position(enemy) != None:
                    enemyPosition = self.startingGameState.get_agent_position(enemy)
                    # We decided to do blinky if we see an agent.
                    # Follow him
                    self.captureAgent.mode = DefenceModes.Blinky
                    return enemyPosition
                else:
                    return self.getGoalPositionIfEnemyLocationUnknown()
            else:
                self.captureAgent.enemyEntered = False

        # If no enemy agent is a pacman try to take some food
        return self.getClosestFood()

    def closestPositionWithPriorityQueue(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        while not positions.isEmpty():
            toPos = positions.pop()
            positionsSorted.push(
                toPos, self.captureAgent.get_maze_distance(toPos, fromPos))
        return positionsSorted

    def closestPosition(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        for toPos in positions:
            positionsSorted.push(
                toPos, self.captureAgent.get_maze_distance(toPos, fromPos))
        return positionsSorted

    def getProbableEnemyEntryPoint(self):
        positionsSorted = util.PriorityQueue()
        positionsSorted = self.closestPosition(
            self.agentPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.get_maze_distance(self.agentPosition, possibleEntry) > 5:
                return possibleEntry
        return random.choice(self.possibleEnemyEntryPositions)

    def getProbableEnemyEntryPointBasedOnFood(self):
        positionsSorted = util.PriorityQueue()
        bestEnemyPosition = util.PriorityQueue()
        # Which entry point are we closest to
        positionsSorted = self.closestPosition(
            self.agentPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.get_maze_distance(self.agentPosition, possibleEntry) > 5:
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

        prevFood = self.captureAgent.get_food_you_are_defending(
                self.captureAgent.get_previous_observation()).as_list() \
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

        myPos = game_state.get_agent_position(self.captureAgent.index)

        if myPos == self.GOAL_POSITION:
            return True
        else:
            return False

    def get_successors(self, state, path = None):
        self.expanded += 1

        game_state = state[0]

        actions = game_state.get_legal_actions(self.captureAgent.index)

        goalDistance = self.captureAgent.get_maze_distance(self.GOAL_POSITION,
                         game_state.get_agent_position(self.captureAgent.index))

        successors_all = [((game_state.generate_successor(self.captureAgent.index, action),
                    goalDistance), action, 1) for action in actions]

        successors = successors_all

        # Add if you want the agent to not be able to leave boundry
        # for successor in successors_all:
        #     (xs, ys) = successor[0][0].get_agent_position(
        #         self.captureAgent.index)
        #     if self.captureAgent.red:
        #         if xs <= self.boundary:
        #             successors.append(successor)
        #     else:
        #         if xs >= self.boundary:
        #             successors.append(successor)

        return successors




################# Search Algorithems ###################




def aStarSearch(problem, heuristic=None):
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