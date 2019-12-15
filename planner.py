from domain_tools import *
import heapq
from time import time

def heuristic(node):
    return 0

def planner(domain,init_node,goals):
    states_explored = 0
    closed = set()
    fringe = [(heuristic(init_node), -init_node.cost, init_node)]
    heapq.heapify(fringe)
    start = time()
    #a-star search
    while True:
        if len(fringe) == 0:
            print('States Explored: %d'%(states_explored))
            return None

        # Get node with minimum evaluation function from heap
        h, _, node = heapq.heappop(fringe)
        states_explored += 1

        # Goal test
        if node.contains(goals):
            print('\nGoal Reached at level: %d'%(node.cost))
            for a in node.path:
                print(a)
            break

        # Expand node if we haven't seen it before
        if node not in closed:
            closed.add(node)
            # Apply all applicable actions to get successors

            successors = set(node.apply(action)
                             for action in domain.actions.values()
                             if node.can_apply(action))

            # Compute heuristic and add to fringe
            for successor in successors:
                if successor not in closed:
                    f = successor.cost + heuristic(successor)
                    heapq.heappush(fringe, (f, -successor.cost, successor))


problem=Problem('./domain/diaper_domain.pddl','./problem/diaper_story.pddl')
domain=problem.domain
init_node=problem.init_state
goals=problem.goals

planner(domain,init_node,goals)