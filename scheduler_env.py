from copy import deepcopy
from gym import Env
import tensorflow as tf
from ml_scheduler import Scheduler, generate_tasksets, load_tasksets 

class SchedulerEnv(Env):
    def __init__(self, scheduler):
        # copy the settings of scheduler
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
        # set counter for resource list
        self.res_cntr = -1
        
        # get shape of our action space
        self.action_shape = self.ntasks * self.msets
        # get shape of our observation space
        self.observation_shape = tf.convert_to_tensor(scheduler.to_array()).shape
        # set scheduler as state of environemnt
        self.state = scheduler
        # jump to next check where at least 1 job is ready and can get executed
        while deepcopy(self.state).generate_states()[0][1] == 0:
            self.state = deepcopy(self.state).generate_states()[0][0]
            self.time = self.state.time

    # do step in environment
    def step(self, action):
        # check if chosen action is a possible action and state to new state
        n_states = self.state.generate_states()
        
        possible_action = False
        
        for n_state in n_states:
            if (n_state[1]-1) % self.ntasks == action % self.ntasks:
                possible_action = True
                self.state = n_state[0] 
                self.time = self.state.time
            
        # jump to next check where at least 1 job is ready and can get executed
        while deepcopy(self.state).generate_states()[0][1] == 0:
            self.state = deepcopy(self.state).generate_states()[0][0]
            self.time = self.state.time

        # calculate reward
        if possible_action:
            reward = self.state.calculate_scores()
            invalid = 0
        else:
            reward = -1
            invalid = 1
            

        # check if hyperperiod is reached or deadline was missed
        if self.state.hyper_period_reached() or reward <= 0:
            done = True
        else:
            done = False

        # set place holder for info
        info = {'invalid': invalid}

        return self.state, reward, done, info

    def render(self):
        print(self.state.to_string())
        
    # reset environment by generating and loading new tasksets with similar settings
    def reset(self):
        # increment pointer on for resource list
        self.res_cntr += 1
        res = [1,2,4,8]
        self.res_num = res[self.res_cntr % 4]
        
        # generate and load tasksets
        generate_tasksets(self.ntasks, self.msets, self.processor_num, self.res_num, self.c_min, self.c_max, self.subset, self.mod)
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

        # generate scheduler object
        self.state = Scheduler(tasksets, settings)
        # jump to next check where at least 1 job is ready and can get executed
        while deepcopy(self.state).generate_states()[0][1] == 0:
            self.state = deepcopy(self.state).generate_states()[0][0]
            self.time = self.state.time
    
        return self.state
