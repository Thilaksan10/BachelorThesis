import numpy as np
import taskset_generator.generator as gen
from agent import Agent
import datetime
from ml_scheduler import Task, Scheduler
from scheduler_env import SchedulerEnv
import matplotlib.pyplot as plt
from copy import deepcopy

# generate tasksets
def generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, mod):
    tasksets_name = './experiments/evaluation/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max)
    tasksets = gen.generate(ntasks, msets, processors * utli, res_num, 0.5, c_min, c_max, mod)
    np.save(tasksets_name, tasksets)

# load tasksets
def load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC):
    task_sets = []
    
    tasksets_name = './experiments/evaluation/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max) + '.npy'
    task_sets = np.load(tasksets_name, allow_pickle=True)

    task_id = 1
    tasksets = []
    for task_set in task_sets:
        taskset = []
        for task in task_set:
            taskset.append(Task(period=task[-1], task_id=task_id, segments=task[:-1], SPORADIC=SPORADIC))
            task_id += 1
        tasksets.append(taskset)

    return tasksets

def select_edf(scheduler):
        states = deepcopy(scheduler).generate_states()
        min_arg = 0
        min_deadline = scheduler.hyper_period + 1
        for index, state in enumerate(states):
            for taskset in scheduler.tasksets:
                for task in taskset:
                    if task.task_id == state[1]:
                        if min_deadline > task.released_job.deadline:
                            min_arg = index
                            min_deadline = task.released_job.deadline
                        break

        return states[min_arg][1]

def select_rate_monotonic(scheduler):
        states = deepcopy(scheduler).generate_states()
        min_arg = 0
        min_period = scheduler.hyper_period + 1
        for index, state in enumerate(states):
            for taskset in scheduler.tasksets:
                for task in taskset:
                    if task.task_id == state[1]:
                        if min_period > task.period:
                            min_arg = index
                            min_period = task.period
                        break

        return states[min_arg][1]

# use priority inheritance protocol to select next job to execute 
# priority = period -> the lower the period the lower the priority 
# job with lowest priority gets executed next
def select_pip(self):
    # generate tuple of (possible next state, task id)
    states = deepcopy(self).generate_states()
    # set task with the lowest priority to be the first task in states list 
    min_arg = 0
    # set minimum period bigger than hyper period
    min_period = self.hyper_period + 1
    # job is currently executing
    if self.processors['processor_1'] is not None:
        # set period of executing job as priority
        exec_task_period = self.processors['processor_1'].period
        # executing sub job of job is critical
        if self.processors['processor_1'].get_current_sub_job().is_critical():
            sub_job = self.processors['processor_1'].get_current_sub_job()
            # if job in waitling list with lower period set priority of executing job to priority of waiting job with lowest priority
            for waiting_job in self.waiting_lists[f'resource_{sub_job.resource_id+1}']:
                exec_task_period = min(exec_task_period, waiting_job.period)
    # search for every task in state list
    for index, state in enumerate(states):
        for taskset in self.tasksets:
            for task in taskset:
                if task.task_id == state[1]:
                    # set the priority of task in consideration
                    current_task_period = task.period
                    # next execution of task is critical and released job of task acquired resource
                    if task.released_job.get_current_sub_job().is_critical() and self.resources[f'resource_{task.released_job.get_current_sub_job().resource_id+1}'] == task.released_job:
                        # if job in waitling list with lower period set priority of executing job to priority of waiting job with lowest priority
                        for waiting_job in self.waiting_lists[f'resource_{task.released_job.get_current_sub_job().resource_id+1}']:
                            current_task_period = min(current_task_period, waiting_job.period)
                    # task in consideration has currently lowest period
                    if min_period > current_task_period:
                        # set index for state list and set the current lowest priority  
                        min_arg = index
                        min_period = current_task_period
                    # task in consideration is executing
                    elif self.processors['processor_1']:
                        # task with lowest priortiy is executing
                        if min_period == exec_task_period and self.processors['processor_1'] == task.released_job:
                            # set index for state list
                            min_arg = index
                    break
    # return the task id
    return states[min_arg][1]

# use priority ceiling protocol to select next job to execute 
# priority = period -> the lower the period the lower the priority 
# job with lowest priority gets executed next     
def select_pcp(self):
    # generate tuple of (possible next state, task id)
    states = deepcopy(self).generate_states()
    # set task with the lowest priority to be the first task in states list 
    min_arg = 0
    # set minimum period bigger than hyper period
    min_period = self.hyper_period + 1
    # set the global priority ceiling to the hyper period
    priority_ceiling = self.hyper_period
    # set the global prioriy ceiling by iteratiing through resources list and find acquired resources
    for index, resource in enumerate(self.resources):
        if self.resources[resource] != None:
            priority_ceiling = min(priority_ceiling, self.priority_ceilings[index])
    # job is currently executing
    if self.processors['processor_1'] is not None:
        # set period of executing job as priority
        exec_task_period = self.processors['processor_1'].period
        # executing sub job of job is critical
        if self.processors['processor_1'].get_current_sub_job().is_critical():
            sub_job = self.processors['processor_1'].get_current_sub_job()
            # do priority inheritance
            for waiting_job in self.waiting_lists[f'resource_{sub_job.resource_id+1}']:
                exec_task_period = min(exec_task_period, waiting_job.period)
    # search for every task in state list
    for index, state in enumerate(states):
        for taskset in self.tasksets:
            for task in taskset:
                if task.task_id == state[1]:
                    current_task_period = task.period
                    # next execution of task is critical and released job of task acquired resource
                    if task.released_job.get_current_sub_job().is_critical() and self.resources[f'resource_{task.released_job.get_current_sub_job().resource_id+1}'] == task.released_job:
                        # do priority inheritance
                        for waiting_job in self.waiting_lists[f'resource_{task.released_job.get_current_sub_job().resource_id+1}']:
                            current_task_period = min(current_task_period, waiting_job.period)
                    # next execution of task is critical and released job of task did not acquire resource
                    elif task.released_job.get_current_sub_job().is_critical() and self.resources[f'resource_{task.released_job.get_current_sub_job().resource_id+1}'] == None:
                        # check if tasks period is lower than global periority ceiling
                        if task.period >= priority_ceiling:
                            current_task_period = min_period
                    # tasnk in consideration has currently lowest period
                    if min_period > current_task_period:
                        # set index for state list and set the current lowest priority 
                        min_arg = index
                        min_period = current_task_period
                    # task in consideration is executing
                    elif self.processors['processor_1']:
                        # task with lowest priortiy is executing
                        if min_period == exec_task_period and self.processors['processor_1'] == task.released_job:
                            # set index of state list
                            min_arg = index
                    break
    # return the task id
    return states[min_arg][1]

if __name__ == '__main__':
    # tasks per taskset
    ntasks = 5
    # number of tasksets
    msets = 100
    # number of processors
    processors = 1
    # num of resources
    resources_no = [1, 2, 4, 8]

    # set minimum and maximum utilization for critiacl section
    c_min = 0.05
    c_max = 0.1
    subset = 1

    # sporadic setting 0 = Periodic, 1 = Sporadic
    SPORADIC = 0
    mod = 1

    min_utli = 5
    max_utli = 105

    # generate tasksets for every utilization
    for res_num in resources_no:
        for i in range (min_utli, max_utli, 5):
            utli = float(i/100)
            generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, mod)
            tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC)
            settings = {
                'ntasks': ntasks, 
                'msets': int(msets/msets), 
                'processors': processors,
                'res_num': res_num,
                'c_min': c_min,
                'c_max': c_max,
                'subset': subset,
                'SPORADIC': SPORADIC,
                'mod': mod,
            }

    # load taskset to initialize environment
    res_num = 1
    utli = 0.05
    
    taskset = []
    for i in range(ntasks):
        taskset.append(Task(1, i+1, [[0,-1], [0,0], [0,-1], [0,0]], SPORADIC))
    tasksets = [taskset]
    settings = {
        'ntasks': ntasks, 
        'msets': int(msets/msets), 
        'processors': processors,
        'res_num': res_num,
        'c_min': c_min,
        'c_max': c_max,
        'subset': subset,
        'SPORADIC': SPORADIC,
        'mod': mod,
    }

    scheduler = Scheduler(tasksets, settings)
    # init environment
    env = SchedulerEnv(scheduler)
    # init agent and load weights
    agent = Agent(n_actions=env.action_shape, input_shape=env.observation_shape, alpha=1e-5, n_tasks=ntasks, m_sets=int(msets/msets))
    agent.load_models()

    # evaluate tasksets on agent
    eval_list_agent = []
    overhead_agent = []

    for res_num in resources_no:
        eval_utli = []
        for i in range(min_utli, max_utli, 5):
            utli = float(i/100)
            won = 0
            tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC)
            settings = {
                'ntasks': ntasks, 
                'msets': int(msets/msets), 
                'processors': processors,
                'res_num': res_num,
                'c_min': c_min,
                'c_max': c_max,
                'subset': subset,
                'SPORADIC': SPORADIC,
                'mod': mod,
            }
            for iteration, taskset in enumerate(tasksets):
                scheduler = Scheduler([taskset], settings)
                env = SchedulerEnv(scheduler)
                done = False
                score = 0
                observation = env.state
                steps = 0
                mean_step_time = datetime.timedelta(0)
                while not done:
                    start = datetime.datetime.now()
                    action = agent.choose_action_eval(observation.to_array())
                    end = datetime.datetime.now()
                    time_delta = end - start
                    mean_step_time += time_delta
                    
                    state_, reward, done, info = env.step(action)
                    score += reward
                    steps += 1
                    
                    observation = state_
                overhead_agent.append(round(mean_step_time.total_seconds()/ steps, 3))
                if reward == 1:
                    won += 1
                if info['invalid']:
                    invalid = 1
                else:
                    invalid = 0

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, invalid {invalid}, final_score {reward}, cummulative score {score}')
            eval_utli.append(won/msets)
        eval_list_agent.append(eval_utli)

    # evaluate tasksets on pcp protocol
    eval_list_pcp = []
    overhead_pcp = []

    for res_num in resources_no:
        eval_utli = []
        for i in range(min_utli, max_utli, 5):
            utli = float(i/100)
            won = 0
            tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC)
    
            settings = {
                'ntasks': ntasks, 
                'msets': int(msets/msets), 
                'processors': processors,
                'res_num': res_num,
                'c_min': c_min,
                'c_max': c_max,
                'subset': subset,
                'SPORADIC': SPORADIC,
                'mod': mod,
            }

            for iteration, taskset in enumerate(tasksets):
                
                scheduler = Scheduler([taskset], settings)
                env = SchedulerEnv(scheduler)
               
                
                done = False
                score = 0
                observation = env.state
                states = [observation]
                steps = 0
                mean_step_time = datetime.timedelta(0)
                while not done:
                    start = datetime.datetime.now()
                    action = select_pcp(observation)
                    end = datetime.datetime.now()
                    time_delta = end - start
                    mean_step_time += time_delta
            
                    state_, reward, done, info = env.step(action-1)
                    score += reward
                    steps += 1
                    
                    observation = state_
                    states.append(observation)
                overhead_pcp.append(round(mean_step_time.total_seconds()/ steps, 3))
                if reward == 1:
                    won += 1

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, final_score {reward}, cummulative score {score}')
            eval_utli.append(won/msets)
        eval_list_pcp.append(eval_utli)

    # evaluate tasksets on pip protocol
    eval_list_pip = []
    overhead_pip = []

    for res_num in resources_no:
        eval_utli = []
        for i in range(min_utli, max_utli, 5):
            utli = float(i/100)
            won = 0
            tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC)
            settings = {
                'ntasks': ntasks, 
                'msets': int(msets/msets), 
                'processors': processors,
                'res_num': res_num,
                'c_min': c_min,
                'c_max': c_max,
                'subset': subset,
                'SPORADIC': SPORADIC,
                'mod': mod,
            }

            for iteration, taskset in enumerate(tasksets):
                scheduler = Scheduler([taskset], settings)
                env = SchedulerEnv(scheduler)
                
                done = False
                score = 0
                observation = env.state
                states = [observation]
                steps = 0
                mean_step_time = datetime.timedelta(0)
                while not done:
                    start = datetime.datetime.now()
                    action = select_pip(observation)
                    end = datetime.datetime.now()
                    time_delta = end - start
                    mean_step_time += time_delta
                    
                    state_, reward, done, info = env.step(action-1)
                    score += reward
                    steps += 1
                    
                    observation = state_
                    states.append(observation)
                overhead_pip.append(round(mean_step_time.total_seconds()/ steps, 3))
                if reward == 1:
                    won += 1

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, final_score {reward}, cummulative score {score}')
            eval_utli.append(won/msets)
        eval_list_pip.append(eval_utli)
   

    # plot overhead figure
    plt.figure(11)
    plt.title(f'Average step in milliseconds')
    plt.plot(range(1, len(overhead_agent)+1), overhead_agent, label='Agent')
    plt.plot(range(1, len(overhead_pcp)+1), overhead_pcp, label = 'PCP')
    plt.plot(range(1, len(overhead_pip)+1), overhead_pip, label = 'PIP')
    plt.xlabel('Task Set')
    plt.ylabel('Time in ms')
    plt.legend()
    plt.savefig(f'eval/overhead.png')
    
    plt.show()
    
    # plot acceptance rate figures for every resource count 
    for k, _ in enumerate(resources_no): 
        plt.figure(k)
        plt.plot(range(min_utli, max_utli, 5), eval_list_agent[k], label='Agent')
        plt.plot(range(min_utli, max_utli, 5), eval_list_pcp[k], label='PCP')
        plt.plot(range(min_utli, max_utli, 5), eval_list_pip[k], label='PIP')
        plt.xlabel('Utilization in %')
        plt.ylabel('Acceptance Rate')
        plt.legend()
        if SPORADIC:
            plt.savefig(f'eval/sporadic/eval-sporadic-{2**k}-resources.png')
        elif not mod:
            plt.savefig(f'eval/framebased/eval-framebased-{2**k}-resources.png')
        else:
            plt.savefig(f'eval/periodic/eval_periodic-{2**k}-resources.png')
        plt.show()

    eval_mix_agent = []
    for m, _ in enumerate(eval_list_agent[0]):
        sum = 0
        for eval_utli in eval_list_agent:
            sum += eval_utli[m]
        sum = sum / len(eval_list_agent)
        eval_mix_agent.append(sum)

    eval_mix_pcp = []
    for m, _ in enumerate(eval_list_pcp[0]):
        sum = 0
        for eval_utli in eval_list_pcp:
            sum += eval_utli[m]
        sum = sum / len(eval_list_pcp)
        eval_mix_pcp.append(sum)

    eval_mix_pip = []
    for m, _ in enumerate(eval_list_pip[0]):
        sum = 0
        for eval_utli in eval_list_pip:
            sum += eval_utli[m]
        sum = sum / len(eval_list_pip)
        eval_mix_pip.append(sum)

    # plot average acceptance rate figure over all resource counts 
    plt.figure(10)
    plt.plot(range(min_utli, max_utli, 5), eval_mix_agent, label='Agent')
    plt.plot(range(min_utli, max_utli, 5), eval_mix_pcp, label = 'PCP')
    plt.plot(range(min_utli, max_utli, 5), eval_mix_pip, label = 'PIP')
    plt.xlabel('Utilization in %')
    plt.ylabel('Acceptance Rate')
    plt.legend()
    if SPORADIC:
        plt.savefig(f'eval/sporadic/eval-sporadic-mixed-resources.png')
    elif not mod:
        plt.savefig(f'eval/framebased/eval-framebased-mixed-resources.png')
    else:
        plt.savefig(f'eval/periodic/eval_periodic-mixed-resources.png')
    
    plt.show()
    