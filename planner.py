from domain_tools import *
import heapq
from time import time

class Planner():
    def __init__(self,domain_file,problem_file):
        problem=Problem(domain_file,problem_file)
        domain=problem.domain
        init_node=problem.init_state
        goals=problem.goals
        self.solve(domain,init_node,goals)

    def heuristic(self,node):
        return 0

    def solve(self,domain,init_node,goals):
        states_explored = 0
        closed = set()
        opened = [(self.heuristic(init_node), -init_node.cost, init_node)]
        heapq.heapify(opened) #min heap
        start = time()
        #a-star search
        while True:
            if len(opened) == 0:
                print('\nStates Explored: %d'%(states_explored))
                return None

            # Get node with minimum evaluation function from heap
            h, _, node = heapq.heappop(opened)
            states_explored += 1

            # Goal test
            if node.contains(goals):
                print('\nGoal Reached at level: %d'%(node.cost))
                for a in node.path:
                    print(a)
                break

            # Expand node if not in closed
            if node not in closed:
                closed.add(node)

                #apply action by checking precondition
                applied=set()
                for action in domain.actions.values():
                    if node.can_apply(action):
                        applied.add(node.apply(action))

                # Compute heuristic and add to fringe
                for successor in applied:
                    if successor not in closed:
                        e = successor.cost + self.heuristic(successor)
                        heapq.heappush(opened, (e, successor.cost, successor))

Planner('./domain/diaper_domain.pddl','./problem/diaper_story.pddl')
