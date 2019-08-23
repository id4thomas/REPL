from __future__ import print_function
from pyddl import Domain, Problem, Action, neg, planner
#import sqlite3
#import timeit
#import time
from Grounder import Grounder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from pyddl.planner import monotone_heuristic
class Planner_env():
	#define problem & domain in init
	def __init__(self):
		#self.con = sqlite3.connect("story_demo.db")
		self.domain = Domain((
			Action(
				'go',
				parameters=(
					('character', 'p'),
					('location', 'l1'),
					('location', 'l2'),
				),
				preconditions=(
					('at', 'p', 'l1'),
				),
				effects = (
					('at', 'p', 'l2'),
					neg(('at', 'p', 'l1')),
				),
			),
			Action(
				'purchase',
				parameters=(
					('character', 'p'),
					('product', 'pr'),
					('location', 'l'),
				),
				preconditions=(
					('at', 'p', 'l'),
					('instock','pr','l'),
					('want','p','pr'),
					('need','p','pr'),
				),
				effects=(
					('purchased', 'p', 'pr'),
					('have', 'p', 'pr'),
					neg(('instock', 'pr', 'l')),
					('outstock', 'pr', 'l'),
					neg(('need','p','pr')),
				),
			),
			Action(
				'fail_buy',
				parameters=(
					('character', 'p'),
					('product', 'pr'),
					('location', 'l'),
				),
				preconditions=(
					('at', 'p', 'l'),
					('outstock', 'pr', 'l'),
					('want','p','pr'),
					('need','p','pr'),
				),
				effects=(
					('failedbuy', 'p', 'pr'),
				),
			),
			Action(
				'refund',
				parameters=(
					('character', 'p'),
					('product', 'pr'),
					('location', 'l'),
				),
				preconditions=(
					('at', 'p', 'l'),
					('have','p','pr'),
					('outstock', 'pr', 'l'),
				),
				effects=(
					('instock', 'pr', 'l'),
					neg(('have','p','pr')),
					neg(('outstock', 'pr', 'l')),
					neg(('purchased','p','pr')),
				),
			),
			Action(
				'change_mind',
				parameters=(
					('character', 'p'),
					('product', 'pr'),
				),
				preconditions=(
					('have','p','pr'),
					('want', 'p', 'pr'),
				),
				effects=(
					neg(('want', 'p', 'pr')),
				),
			)
		))
		self.problem = Problem(
				self.domain,
				{
					'location': ('Ahome','Shop','Bhome'),
					'character': ('A', 'B'),
					'product': ('D'),
				},
				init=(
					('at', 'A', 'Ahome'),
					('at', 'B', 'Bhome'),
					('want','A','D'),
					('want','B','D'),
					('need','A','D'),
					('need','B','D'),
					('instock', 'D', 'Shop'),
				),
				goal=(
					('failedbuy', 'A', 'D'),
					('at', 'B', 'Shop'),
					('instock', 'D', 'Shop'),
					('at', 'A', 'Ahome'),
					('need','A','D'),
				),
		)
		#self.actions={}
		#self.ground_actions()
		grounder=Grounder(self)
		self.predicate_comb=grounder.get_predicate_combination(self.problem,self.domain)
		self.action_comb=[str(action) for action in self.problem.grounded_actions]

		self.num_states=len(self.predicate_comb)
		self.num_actions=len(self.action_comb)

		self.init_state=self.problem.initial_state
		print(self.predicate_comb)
		#print(self.action_comb)
		#print(self.actions)
		#self.state_binarizer = OneHotEncoder(categories=self.predicate_comb)
		#self.state_binarizer.fit(self.predicate_comb)
		self.goal = (self.problem.goals, self.problem.num_goals)
		self.goal_set=set(self.problem.goals)
		#print('goals',self.problem.goals,type(self.problem.goals))
	def ground_actions(self):
		#self.initialize_problem()
		#print("problem initiated")
		self.actions={}
		for action in self.problem.grounded_actions:
			#print("name",action.name,"prec",action.preconditions)
			if(action.name not in self.actions):
				self.actions[action.name]=[]
			self.actions[action.name].append(action)
		return

	def reset(self):
		#return to inital state
		return self.init_state

	def state_to_onehot(self,state):
		#print('state',state.predicates)
		onehot=np.zeros(self.num_states)
		for i in range(self.num_states):
			if self.predicate_comb[i] in state.predicates:
				onehot[i]=1
		return onehot

	def apply_action(self,node,action_idx):
		applied=False
		goal_reached=False
		target_action=self.action_comb[action_idx]
		action_name=target_action[:target_action.find('(')]
		action_params=target_action[target_action.find('(')+1:-1]
		#print(action_name)
		#print(action_params)
		#print(tuple((str(action_name)+", "+str(action_params)).split(', ')))
		for action in self.actions[action_name]:
			#print(action.sig)
			if action.sig==tuple((str(action_name)+", "+str(action_params)).split(', ')):
				#print("found action:",action.sig)
				#print("my action",(str(row[1])+","+str(row[2])),end=",")
				if node.is_true(action.preconditions,action.num_preconditions):
					node=node.apply(action)
					applied=True
					#print('Applied')
					#check goal!!!!
					if node.is_true(*self.goal):
						goal_reached=True
					return node,applied,goal_reached
				else:
					#print("Cannot apply action")
					goal_reached=True
					break
		return node,applied,goal_reached

	def simple_heuristic(self,node):
		inter=self.goal_set.intersection(node.predicates)
		return len(inter)
