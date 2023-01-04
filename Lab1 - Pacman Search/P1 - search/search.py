# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** MY CODE***"
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []

    # use a stack to implement the dfs
    frontier = util.Stack()

    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(), [], 0)
    frontier.push(start_node)

    # continue until there are no more node to expand
    while not frontier.isEmpty():

        #take the next node
        (node, path, cost) = frontier.pop()
        expanded_nodes.append(node)

        # if this node is in goal state it return the path to reach that state
        if problem.isGoalState(node):
            return path

        # if not the algorithm search in his successor and insert them in the frontier to be expanded
        for (child, n_action, n_cost) in problem.getSuccessors(node):
            if child not in expanded_nodes: #also a node from the frontier could be insert again
                total_cost = cost + n_cost
                total_path = path + [n_action]
                frontier.push((child, total_path, total_cost))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** MY CODE ***"
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []

    # use a queue to implement the bfs
    frontier = util.Queue()

    # a list that will contain only the position of the frontier
    frontier_nodes = []

    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(), [], 0)
    frontier.push(start_node)
    frontier_nodes.append(start_node[0])
    while not frontier.isEmpty():

        #take the next node
        (node, path, cost) = frontier.pop()
        frontier_nodes.remove(node)
        expanded_nodes.append(node)

        # if this node is in goal state it return the path to reach that state
        if problem.isGoalState(node):
            return path

        # if not the algorithm search in his successor and insert them in the frontier to be expanded
        for (child, n_action, n_cost) in problem.getSuccessors(node):
            if child not in frontier_nodes and child not in expanded_nodes: #node from the frontier does not be insert again
                total_cost = cost + n_cost
                total_path = path + [n_action]
                frontier.push((child, total_path, total_cost))
                frontier_nodes.append(child)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** MY CODE ***"
    # create a list to store the expanded nodes that do not need to be expanded again
    expanded_nodes = []

    # use a priority queue to store states
    frontier = util.PriorityQueue()

    # A node in the stack is composed by :
    # his position
    # his path from the initial state
    # his total cost to reach that position
    start_node = (problem.getStartState(), [], 0)

    # as priority is inserted the total cost to reach that position
    frontier.push(start_node, start_node[2])
    while not frontier.isEmpty():
        (node, path, cost) = frontier.pop()

        # if this node is in goal state it return the path to reach that state
        if problem.isGoalState(node):
            return path

        # the algorithm control if the node is being expanded before
        if node not in expanded_nodes:
            expanded_nodes.append(node)

            # if not the algorithm search in his successor and insert them in the frontier to be expanded
            for (child, n_action, n_cost) in problem.getSuccessors(node):
                if child not in expanded_nodes:
                    total_cost = cost + n_cost
                    total_path = path + [n_action]
                    frontier.push((child, total_path, total_cost), total_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** MY CODE ***"
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
            for (child, n_action, n_cost) in problem.getSuccessors(node):
                if child not in expanded_nodes:
                    # fut_cost must be passed and represent the cost to reach that position
                    fut_cost = cost + n_cost

                    #total cost is the fut cost + heuristic and is passed as the priority
                    total_cost = cost + n_cost + heuristic(child, problem)
                    total_path = path + [n_action]
                    frontier.push((child, total_path, fut_cost), total_cost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
