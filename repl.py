#reinforcement planner
#single agent
import numpy as np

# namedtuple 생성
from checker import Checker

from Planner_env import Planner_env
import operator as ops
from collections import namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

NUM_OPS = {
    '>' : ops.gt,
    '<' : ops.lt,
    '=' : ops.eq,
    '>=': ops.ge,
    '<=': ops.le
}
#Hyper Parameters
GAMMA = 0.9  # Time Discount Value
MAX_STEPS = 100  # Max steps per Episode
NUM_EPISODES = 100000  # Max number of Episodes

BATCH_SIZE = 32
CAPACITY = 1000 #Capacity of Memory Class

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY   # 메모리의 최대 저장 건수
        self.memory = []  # 실제 transition을 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)을 메모리에 저장'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 메모리가 가득차지 않은 경우

        # Transition이라는 namedtuple을 사용하여 키-값 쌍의 형태로 값을 저장
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 다음 저장할 위치를 한 자리 뒤로 수정

    def sample(self, batch_size):
        '''batch_size 갯수 만큼 무작위로 저장된 transition을 추출'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''len 함수로 현재 저장된 transition 갯수를 반환'''
        return len(self.memory)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        # Dueling Network
        self.fc3_adv = nn.Linear(n_mid, n_out)  # Advantage함수쪽 신경망
        self.fc3_v = nn.Linear(n_mid, 1)  # 가치 V쪽 신경망

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h1))
        adv = self.fc3_adv(h3)  # 이 출력은 ReLU를 거치지 않음
        val = self.fc3_v(h2).expand(-1, adv.size(1))  # 이 출력은 ReLU를 거치지 않음
        # val은 adv와 덧셈을 하기 위해 expand 메서드로 크기를 [minibatch*1]에서 [minibatch*2]로 변환
        # adv.size(1)은 2(출력할 행동의 가짓수)

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        # val+adv에서 adv의 평균을 뺀다
        # adv.mean(1, keepdim=True) 으로 열방향(행동의 종류 방향) 평균을 구함 크기는 [minibatch*1]이 됨
        # expand 메서드로 크기를 [minibatch*2]로 늘림

        return output

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class Brain:
    def __init__(self,num_states,num_actions):
        self.num_actions = num_actions

        #make memory instance
        self.memory = ReplayMemory(CAPACITY)

        #make network
        n_in, n_mid, n_out = num_states, 64, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)  # Net 클래스를 사용
        self.target_q_network = Net(n_in, n_mid, n_out)  # Net 클래스를 사용
        print(self.main_q_network)

        #optimizer : adam
        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.01)

    def replay(self):
        '''Experience Replay로 신경망의 결합 가중치 학습'''

        #print('cur memory ',len(self.memory))
        # 1. 저장된 transition의 수를 확인
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 미니배치 생성
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 결합 가중치 수정
        self.update_main_q_network()

    def decide_action(self, state, episode):
        # epsilon greedy
        epsilon = 0.8 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):#select action by q value - exploit
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:#select random action - explore
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])

        #shape of action : torch.LongTensor of size 1*1
        return action

    def make_minibatch(self):
        '''2. 미니배치 생성'''

        # 2.1 메모리 객체에서 미니배치를 추출
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 갯수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
        # 이것을 미니배치로 만들기 위해
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환한다
        batch = Transition(*zip(*transitions))

        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있도록 Variable로 만든다
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 갯수만큼 있는 형태이다
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형한다
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        # cat은 Concatenates(연접)을 의미한다
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        #print(batch.next_state)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states


    def get_expected_state_action_values(self):
        '''정답신호로 사용할 Q(s_t, a_t)를 계산'''

        # 3.1 신경망을 추론 모드로 전환
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며
        # [torch.FloatTensor of size BATCH_SIZEx2] 형태이다
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가
        # 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고, 이에 대한 Q값을 gather 메서드로 모아온다
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}값을 계산한다 이때 다음 상태가 존재하는지에 주의해야 한다

        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 결합 가중치 수정'''

        # 4.1 신경망을 학습 모드로 전환
        self.main_q_network.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_state_action_values은
        # size가 [minibatch]이므로 unsqueeze하여 [minibatch*1]로 만든다
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 결합 가중치를 수정한다
        self.optimizer.zero_grad()  # 경사를 초기화
        loss.backward()  # 역전파 계산
        self.optimizer.step()  # 결합 가중치 수정

    def update_target_q_network(self):  # DDQN에서 추가됨
        '''Target Q-Network을 Main Q-Network와 맞춤'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict() )

class Agent:
    def __init__(self,num_states,num_actions):
        #make brain
        self.brain = Brain(num_states, num_actions)
        #pass

    def update_q_function(self):
        #experience replay
        self.brain.replay()
        #pass

    def get_action(self,state,episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self,state,action,state_next,reward):
        #memorize information to memory instance
        self.brain.memory.push(state, action, state_next, reward)
        #pass

    def update_target_q_function(self):
        #update target q network to match main q
        self.brain.update_target_q_network()
        #pass

class Environment:
    def __init__(self):
        self.env=Planner_env()
        num_states=self.env.num_states
        num_actions=self.env.num_actions
        print('Num states: %d Num actions: %d'%(num_states,num_actions))
        self.agent=Agent(num_states,num_actions)
        self.num_goals=len(self.env.goal_set)
    def run(self):
        episode_10_list = np.zeros(10)  # number of steps survived in recent 10 episodes
        complete_episodes = 0  # number of successful episodes
        episode_final = False  # 마지막 에피소드 여부
        #current_node = self.env.reset()
        #state=self.env.state_to_onehot(current_node)
        #self.env.apply_action(self.env.reset(),0)

        for episode in range(NUM_EPISODES):
            current_node = self.env.reset() #get story init state

            #convert node to onehot
            state=self.env.state_to_onehot(current_node)
            state = torch.from_numpy(state).type(torch.FloatTensor)

            state = torch.unsqueeze(state, 0)#size num_state -> 1*num_state

            for step in range(MAX_STEPS): #one episode
                #get action from agent
                action = self.agent.get_action(state, episode)#episode: episodenum
                #print('step %d applying %d'%(step,action))
                #apply action
                next_node,applied,done=self.env.apply_action(current_node,action)

                #done & applied : goal
                #done & !applied : fail
                max_reached=False
                if step>1000:
                    print('Max Step Reached')
                    done=True
                    applied=False
                    max_reached=True
                h=self.env.simple_heuristic(next_node)
                #print('cost:',h)
                if done:#action failed
                    print("Episode Done")
                    next_node=None
                    state_next=None
                    #save survived step to recent 10 episodes list
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if applied:
                        print('goal reached')
                        reward = torch.FloatTensor([50.0])#reward 1 - successful
                        complete_episodes = complete_episodes + 1
                    else:
                        if max_reached:
                            print('max reached')
                            #reward = torch.FloatTensor([-1.0])#reward is penalty
                            reward = torch.FloatTensor([-1])#reward is penalty
                            complete_episodes = 0
                        else:
                            print('story failed')
                            #reward = torch.FloatTensor([-1.0])#reward is penalty
                            reward = torch.FloatTensor([-5])#reward is penalty
                            complete_episodes = 0

                else:#action successful
                    #print('reward: ',((h/self.num_goals)*10)/(step+1))
                    reward = torch.FloatTensor([((h/self.num_goals)*100)/(step+1)])  #reward 0
                    state_next=self.env.state_to_onehot(next_node)
                    #state_next = observation_next  # 관측 결과를 그대로 상태로 사용
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)

                # Experience Replay로 Q함수를 수정
                self.agent.update_q_function()

                # 관측 결과를 업데이트
                current_node = next_node
                state=state_next

                if done:
                    if applied:
                        print('goal reached')
                    else:
                        print('story fail')
                    print('%d Episode: Finished after %d steps：최근 10 에피소드의 평균 단계 수 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    if(episode % 2 == 0):
                        self.agent.update_target_q_function()
                    break
                pass
            if step==MAX_STEPS:
                print('story fail')
            print("episode %d complete steps %d"%(episode,step))
            #episode complete
            if episode_final is True:
                break

            if complete_episodes >= 10: #10 consecutive episodes
                print('10 에피소드 연속 성공')
                episode_final = True
env=Environment()
env.run()
'''
print('Grounding Complete')
print('Grounded Actions')
print(grounded.action_comb)
print('\nGrounded Predicates')
print(grounded.predicate_comb)
'''
