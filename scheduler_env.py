from copy import deepcopy
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import tensorflow as tf
from ml_scheduler import Scheduler, generate_tasksets, load_tasksets 

class SchedulerEnv(Env):
    def __init__(self, scheduler):
        self.hyper_period = scheduler.hyper_period
        self.ntasks = scheduler.ntasks
        self.msets = scheduler.msets
        self.processor_num = scheduler.processor_num
        self.res_num = scheduler.res_num
        self.c_min = scheduler.c_min
        self.c_max = scheduler.c_max
        self.subset = scheduler.subset
        self.SPORADIC = scheduler.SPORADIC
        self.mod = scheduler.mod
        self.res_cntr = -1

        self.action_shape = self.ntasks * self.msets
        self.observation_shape = tf.convert_to_tensor(scheduler.to_array()).shape
        self.state = scheduler
        while deepcopy(self.state).generate_states()[0][1] == 0:
            self.state = deepcopy(self.state).generate_states()[0][0]
            self.time = self.state.time

    def step(self, action):
        # check if chosen action is a possible action and state to new state
        n_states = self.state.generate_states()
        # print(n_states)
        # print(action)
        possible_action = False
        # print(action)
        # print(action[0])
        for n_state in n_states:
            if n_state[1]-1 == action:
                possible_action = True
                self.state = n_state[0] 
                self.time = self.state.time 
            
        # self.state = action[0]
        # self.time = self.state.time
        # possible_action = True
               
        # while len(deepcopy(self.state).generate_states()) == 1:
        #     self.state = deepcopy(self.state).generate_states()[0][0]
        #     self.time = self.state.time
        # print(deepcopy(self.state).generate_states()[0][1])
        while deepcopy(self.state).generate_states()[0][1] == 0:
            self.state = deepcopy(self.state).generate_states()[0][0]
            self.time = self.state.time

        # calculate reward
        if possible_action:
            reward = self.state.calculate_scores()
            invalid = 0
        else:
            # prob = []
            # for n_state in n_states:
            #     prob.append(action[n_state[1]])
            # high = np.argmax(prob)
            # self.state = random.choice(n_states)[0]
            # self.time = self.state.time
            reward = -1
            invalid = 1
            

        # check if all tasks are schedules or hyperperiod reached
        if self.state.all_tasks_scheduled() or self.state.time >= 10 or reward <= 0:
            done = True
        else:
            done = False

        # set place holder for info
        info = {'invalid': invalid}

        return self.state, reward, done, info

    def render(self):
        print(self.state.to_string())

    def reset(self):
        # generate new taskset with same setting
        # if won:
        self.res_cntr += 1
        res = [1,2,4,8]
        self.res_num = res[self.res_cntr % 4]
        generate_tasksets(self.ntasks, self.msets, self.processor_num, self.res_num, self.c_min, self.c_max, self.subset, self.mod)
            # print('generate new taskets ...')
        # load newly generated taskset
        tasksets = load_tasksets(self.ntasks, self.msets, self.processor_num, self.res_num, self.c_min, self.c_max, self.subset, self.SPORADIC)
        
        settings = {
            'ntasks': self.ntasks, 
            'msets': self.msets, 
            'processors': self.processor_num,
            'res_num': self.res_num,
            'c_min': self.c_min,
            'c_max': self.c_max,
            'subset': self.subset,
            'SPORADIC': self.SPORADIC,
            'mod': self.mod,
        }
        self.state = Scheduler(tasksets, settings)

        while deepcopy(self.state).generate_states()[0][1] == 0:
            self.state = deepcopy(self.state).generate_states()[0][0]
            self.time = self.state.time
    
        return self.state
