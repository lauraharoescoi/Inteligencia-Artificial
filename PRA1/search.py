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
import node

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
    initial_state = problem.getStartState()
    n = node.Node(initial_state)
    fringe = util.Stack()
    fringe.push(n)
    expanded = []
	
    while True:
        if fringe.isEmpty():
            return None
        n = fringe.pop()
        expanded.append(n.state)

        if problem.isGoalState(n.state):
            return n.total_path()
			
        for successor in problem.getSuccessors(n.state):
            (state, action, cost) = successor
            if state not in expanded and state not in (node.state for node in fringe.list):
                fringe.push(node.Node(state, n, action, n.cost + cost))


def breadthFirstSearch(problem):
    initial_state = problem.getStartState()
    n = node.Node(initial_state)
    fringe = util.Queue()
    fringe.push(n)
    expanded = []
	
    while True:
        if fringe.isEmpty():
            return None
        n = fringe.pop()
        expanded.append(n.state)

        if problem.isGoalState(n.state):
            return n.total_path()
			
        for successor in problem.getSuccessors(n.state):
            (state, action, cost) = successor
            if state not in expanded and state not in (node.state for node in fringe.list):
                fringe.push(node.Node(state, n, action, n.cost + cost))


def uniformCostSearch(problem):
    from node import Node

    initial_state = problem.getStartState()

    n = Node(initial_state)

    fringe = util.PriorityQueue()
    fringe.push(n,0)
    expanded = []
	
    while True:
        if fringe.isEmpty():
            return None

        n = fringe.pop()
        expanded.append(n.state)

        if problem.isGoalState(n.state):
            return n.total_path()
			
        for successor in problem.getSuccessors(n.state):
            (state, action, cost) = successor
            priority = n.cost + cost
            if state not in expanded and state not in (node[2].state for node in fringe.heap):
                fringe.push(Node(state, n, action, n.cost + cost), priority)
            elif state not in expanded:
                for node in fringe.heap:
                    if state == node[2].state and priority < node[2].cost:
                        fringe.update(Node(state, n, action, n.cost + cost), priority)



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    from node import Node

    initial_state = problem.getStartState()

    n = Node(initial_state)

    fringe = util.PriorityQueue()
    fringe.push(n,0 + heuristic(n.state, problem))
    expanded = []
	
    while True:
        if fringe.isEmpty():
            return None

        n = fringe.pop()
        expanded.append(n.state)

        if problem.isGoalState(n.state):
            return n.total_path()
			
        for successor in problem.getSuccessors(n.state):
            (state, action, cost) = successor
            priority = n.cost + cost + heuristic(state, problem)
            if state not in expanded and state not in (node[2].state for node in fringe.heap):
                fringe.push(Node(state, n, action, n.cost + cost), priority)
            elif state not in expanded:
                for node in fringe.heap:
                    if state == node[2].state and priority < node[2].cost:
                        fringe.update(Node(state, n, action, n.cost + cost), priority)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
