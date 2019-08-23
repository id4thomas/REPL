
import numpy as np

# namedtuple 생성
from checker import Checker
import operator as ops
import itertools #for combination

NUM_OPS = {
    '>' : ops.gt,
    '<' : ops.lt,
    '=' : ops.eq,
    '>=': ops.ge,
    '<=': ops.le
}

class Grounder():
    def __init__(self,env):
        #get grounded
        self.env=env
        self.env.ground_actions()
        #problem=self.env.problem
        #domain=self.env.domain

    def get_arg_list(self,predicate,arg_dict):
        arg_list=[]
        #arglist = ', '.join(map(str, self.sig[1:]))
        for i in range(1,len(predicate)):
            arg_list.append(arg_dict[predicate[i]])
        #print(predicate,'arg_list',arg_list)
        return predicate[0],arg_list

    def get_predicate_types(self,problem,domain):
        predicate_types=dict()
        #get possible actions
        actions=domain.actions

        for action in actions:
            #init argument parameters for this action
            arg_dict=dict()
            for i in range(len(action.types)):
                arg_dict[action.arg_names[i]]=action.types[i]

            #get predicate types from preconds and effects
            for pre in action.preconditions:
                if pre[0] in NUM_OPS:
                    pass
                else:
                    k,v=self.get_arg_list(pre,arg_dict)
                    predicate_types[k]=v

            for effect in action.effects:
                if effect[0] == -1:
                    k,v=self.get_arg_list(effect[1],arg_dict)
                    predicate_types[k]=v
                elif effect[0] == '+=':
                    pass
                elif effect[0] == '-=':
                    pass
                else:
                    k,v=self.get_arg_list(effect,arg_dict)
                    predicate_types[k]=v

        print('predicate types',predicate_types)
        return predicate_types

    def _GroundedPredicate(self,predicate,params):
        grounded=[predicate]

        return grounded

    def get_predicate_combination(self,problem,domain):
        types=self.get_predicate_types(problem,domain)

        combination=[]
        objects=problem.objects
        actions=domain.actions
        print(objects)

        #iterate over predicates
        for k in types.keys():
            print(k,types[k])
            if len(types[k])==0:#0 parameters
                param_comb=[]
            elif len(types[k])==1: #1 parameter
                param_comb=objects[types[k][0]]
            else: #at least two
                param_comb=list(itertools.product([k], objects[types[k][0]]))
                #param_comb=list(itertools.product(objects[types[k][0]], objects[types[k][1]]))
                #print(param_comb)
                for i in range(1,len(types[k])):
                    product=list(itertools.product(param_comb,objects[types[k][i]]))
                    param_comb=[]
                    for comb in product:
                        front=list(comb[0])
                        front.append(comb[1])
                        param_comb.append(front)
                        #print(list(comb[0]),comb[1],test)
                    #param_comb=[list(comb[0]).append(comb[1]) for comb in param_comb]
                    #param_comb=list(zip(param_comb[0],param_comb[1]))
                combination+=param_comb
                #print(param_comb)
        #print(combination)
        print('Total %d combinations'%(len(combination)))
        #list to tuple
        combination=[tuple(c) for c in combination]
        #print(combination)
        return combination
