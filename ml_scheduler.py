from copy import deepcopy
from operator import mod
from mcts import TreeNode, MCTS
import numpy as np
import taskset_generator.generator as gen
from tqdm import tqdm

class SubJob:
    def __init__(self, utilization, resource_id) -> None:
        self.execution = utilization
        self.resource_id = resource_id

    # checks if sub job is a critical sub job or non-critical job
    def is_critical(self):
        return self.resource_id != -1

    def to_string(self):
        return f'[{self.execution}, {self.resource_id}]'
        

class Job:
    def __init__(self, sub_jobs, period, release_time, deadline, task_id):
        self.sub_jobs = deepcopy(sub_jobs)
        self.period = period
        self.release_time = release_time
        self.deadline = deadline
        self.task_id = task_id
        self.preempt = False
        self.index = 0

    # checks if job is ready to be added to ready list
    def is_ready(self, time):
        return self.release_time <= time

    # check if job is finished
    def is_finished(self):
        return self.index == len(self.sub_jobs)-1

    # returns the current sub job to be executed for this task
    def get_current_sub_job(self):
        return self.sub_jobs[self.index]

    def to_list(self, processor, resource, waiting_priority, time, max_sub_job_count=13):
        list = [[processor], [resource], [waiting_priority], [self.period], [self.release_time], [self.deadline-time]]
        # print(list)
        for sub_job in self.sub_jobs:
            list.append([sub_job.execution])
            list.append([sub_job.resource_id])
            max_sub_job_count -= 1
        for _ in range(max_sub_job_count):
            if list[7] != -1:
                list.insert(6, [-1])
            else:
                list.insert(6,[0])
            list.insert(6, [0])
        # print(list)
        return list

    def to_string(self):
        sub_jobs = f''
        for sub_job in self.sub_jobs:
            sub_jobs += f' {sub_job.to_string()} '
        return f'P: {self.period}, R: {self.release_time}, D: {self.deadline}, ID: {self.task_id} \n Jobs: {sub_jobs}'

class Task:
    def __init__(self, period, task_id, segments):
        self.period = period
        self.task_id = task_id
        self.segments = []
        for segment in segments:
            self.segments.append(SubJob(utilization=segment[0], resource_id=segment[1]))
        self.released_job = Job(self.segments, self.period, 0, self.period, self.task_id)

    # checks if recently released job is ready to be added to the ready list
    def release_job(self, time):
        if self.released_job.is_finished():
            if time >= self.released_job.release_time + self.period:
                release_time = self.released_job.deadline
                deadline = release_time + self.period
                self.released_job = Job(self.segments, self.period, release_time, deadline, self.task_id)
                return True
        return False

    # ckecks if all jobs this task released finished its execution
    def is_finished(self):
        return self.released_job.is_finished() 

    # returns the recently released job 
    def get_current_job(self):
        return self.released_job

    # returns the current sub job to be executed of the recently released task
    def get_current_sub_job(self):
        return self.released_job.get_current_sub_job()

    def to_string(self):
        # print(f'Length: {len(self.tasks)}')
        if self.released_job:
            return f'\n{self.released_job.to_string()}'
        else:
            return None


class Scheduler:
    def __init__(self, tasksets=None, settings=None, scheduler=None):

        self.time = 0
        self.ready_list = []
        self.resources = {}
        self.waiting_lists = {}
        self.processors = {}
        self.missed_deadline = []

        if settings is not None:
            self.ntasks = settings['ntasks']
            self.msets = settings['msets']
            self.processor_num = settings['processors']
            self.res_num = settings['res_num']
            self.c_min = settings['c_min']
            self.c_max = settings['c_max']
            self.subset = settings['subset']
            self.SPORADIC = settings['SPORADIC']
            self.mod = settings['mod']

            for i in range(self.res_num):
                self.resources[f'resource_{i+1}'] = None
                self.waiting_lists[f'resource_{i+1}'] = []
        
            for i in range(self.processor_num):
                self.processors[f'processor_{i+1}'] = None
        
        if tasksets is not None:
            self.tasksets = tasksets
        else:
            self.__dict__ = deepcopy(scheduler.__dict__)

        self.hyper_period = self.calculate_hyper_period()

    def calculate_hyper_period(self):
        max = 1
        for taskset in self.tasksets:
            for task in taskset:
                if task.period > max:
                    max = task.period
        return max

    # check if a new job is released
    def new_job_released(self, previous_state):
        return len(previous_state.ready_list) < len(self.ready_list)

    # recently released job of the given task acquires resource
    def lock_resource(self, job):
        sub_job = job.get_current_sub_job()
        # print(job.resource_id + 1)
        self.resources[f'resource_{sub_job.resource_id + 1}'] = job

    # recently released job of the given task releases resource
    def release_resource(self, job):
        sub_job = job.get_current_sub_job()
        self.resources[f'resource_{sub_job.resource_id + 1}'] = None

    # queues task to the ready list
    def queue(self, task):
        # print(f'Task: {periodic_task.to_string()}')
        # print(f'Ready: {periodic_task.is_ready(self.time)}')
        # print('____________________________________')
        # append task to ready list
        if task.released_job not in self.processors.values() and task.released_job not in self.ready_list and not task.released_job.is_finished():
            self.ready_list.append(task.released_job)
            # job arrived and elligible to preempt current executing job
            task.get_current_job().preempt = True

    # returns the arrival time of the next task 
    def get_next_task_arrival(self):
        next_arrival = float('inf')
        # iterate through 'not ready' list to find the earliest released task
        for taskset in self.tasksets:
            for task in taskset:
                # print(f'Next Arrival: {next_arrival}')
                # print(task.to_string())
                # print(len(task.get_current_job().sub_jobs))
                # print(task.get_current_job().index)
                # print(task.get_current_job().is_finished())
                if task.get_current_job().is_finished():
                    next_release = task.get_current_job().release_time + task.period
                    # print(f'Next Release: {next_release}')
                    if next_release < self.time:
                        # print(f'Next Release2: {next_release}')
                        next_release = float('inf')
                    next_arrival = min(next_arrival, next_release)
        return next_arrival

    def insert_in_waiting_list_edf(self, job):
        # traverse waiting list of needed resource and insert edf
        inserted = False
        for i, waiting_job in enumerate(self.waiting_lists[f'resource_{job.get_current_sub_job().resource_id + 1}']):
            if waiting_job.deadline > job.deadline :
                self.waiting_lists[f'resource_{job.get_current_sub_job().resource_id + 1}'].insert(i, job)
                inserted = True
                break
        if not inserted:
            self.waiting_lists[f'resource_{job.get_current_sub_job().resource_id + 1}'].append(job)

    # appends given job into the waiting list of needed resouce
    def append_to_waiting_list(self, job):
        self.waiting_lists[f'resource_{job.get_current_sub_job().resource_id + 1}'].append(job)

    # executes given job
    def execute(self, job=None):
        # calculate remaining time for next checkpoint
        remaining_time = self.get_next_task_arrival()
        if remaining_time != float('inf'):
            remaining_time -= self.time
        # check if resource needs to be acquired for execution
        added_to_wating_list = 0
        if job:
            sub_job = job.get_current_sub_job()
            # case sub job is critical
            if sub_job.is_critical():
                # check if sub job can acquire resource
                # case resource could be acquired 
                if self.resources[f'resource_{sub_job.resource_id + 1}'] == None:
                    # case job acquires resource
                    if len(self.waiting_lists[f'resource_{sub_job.resource_id + 1}']) == 0:
                        self.lock_resource(job)
                    else:
                        if job in self.waiting_lists[f'resource_{sub_job.resource_id + 1}']:
                            self.waiting_lists[f'resource_{sub_job.resource_id + 1}'].remove(job)
                # case resources is already acquired by another job
                else:
                    if self.resources[f'resource_{sub_job.resource_id + 1}'] != job:
                        self.append_to_waiting_list(job)
                        added_to_wating_list = 1
                        job = None
                    
        # case given job needs to be executed
        if job:
            self.ready_list.remove(job)
            sub_job = job.get_current_sub_job()
            # check if processor is free
            # case processor is free
            if self.processors['processor_1'] is None:
                # job acquires processor
                self.processors['processor_1'] = job
                # calculate time of next checkpoint
                self.time += min(sub_job.execution, remaining_time)
                # calculate remaining time to execute for current sub job
                sub_job.execution -= min(sub_job.execution, remaining_time)
                # case sub job's execution is done
                if sub_job.execution <= 0:
                    # sub job is critical
                    if sub_job.is_critical():
                        # job releases resource
                        self.release_resource(self.processors['processor_1'])
                    # pointer for subjoblist of current job is incremented
                    job.index += 1
                    # case job did not finished all its sub jobs
                    if not job.is_finished():
                        # append job to 'ready' list
                        self.ready_list.append(self.processors['processor_1'])  
                    # clear processor
                    self.processors['processor_1'] = None 
            # case preempt job holding processor 
            else: 
                # append job holding processor to 'ready' list
                self.ready_list.append(self.processors['processor_1'])
                # given job acquires processor
                self.processors['processor_1'] = job
                # calculate time to next checkpoint
                self.time += min(sub_job.execution, remaining_time)
                # calculate remaining execution time of sub job
                sub_job.execution -= min(sub_job.execution, remaining_time)
                # case sub job finished
                if sub_job.execution <= 0:
                    # case sub job is critical
                    if sub_job.is_critical():
                        # job holding the processor releases resource
                        self.release_resource(self.processors['processor_1'])
                    # pointer for subjoblist of current job is incremented
                    job.index += 1
                    # case job is not finished
                    if not job.is_finished():
                        # append job to 'ready' list
                        self.ready_list.append(self.processors['processor_1'])
                    # clear processor
                    self.processors['processor_1'] = None
        # case there is no job given to be executed
        else: 
            # job is already holding the processor
            if self.processors['processor_1'] is not None:
                # get sub job & job which is holding processor
                prev_job = self.processors['processor_1']
                prev_sub_job = prev_job.get_current_sub_job()
                # case sub job to be executed is critical
                if prev_sub_job.is_critical():
                    # case job acquired resource, which is needed to execute critical sub job
                    if self.resources[f'resource_{prev_sub_job.resource_id + 1}'] == self.processors['processor_1']:
                        # calculate time at next checkpoint
                        self.time += min(prev_sub_job.execution, remaining_time)
                        # calculate remaining execution time of executed sub job
                        prev_sub_job.execution -= min(prev_sub_job.execution, remaining_time)
                        # case executed sub job is finished
                        if prev_sub_job.execution <= 0:
                            # case executed sub job is critical
                            if prev_sub_job.is_critical():
                                # job currently holding the processor releases resource 
                                self.release_resource(self.processors['processor_1'])
                            # pointer for subjoblist of current job is incremented 
                            prev_job.index += 1
                            # case job did not finished all its sub job
                            if not prev_job.is_finished():
                                # append job to 'ready' list
                                self.ready_list.append(self.processors['processor_1'])
                            # processor gets cleared
                            self.processors['processor_1'] = None

                # sub job to be executed is not critical
                else:
                    # calculate time for next checkpoint
                    self.time += min(prev_sub_job.execution, remaining_time)
                    # calculate remaining execution time of current sub job
                    prev_sub_job.execution -= min(prev_sub_job.execution, remaining_time)
                    # case executed sub job has finished execution
                    if prev_sub_job.execution <= 0:
                        # case executed sub job is critical
                        if prev_sub_job.is_critical():
                            # job releases its resource
                            self.release_resource(self.processors['processor_1'])
                        # pointer for subjoblist of current job is incremented 
                        prev_job.index += 1
                        # case job finished all its sub jobs
                        if prev_job.is_finished():
                            # pointer for joblist is incremented for task holding the processor
                            self.processors['processor_1'].index += 1
                            # case task finsished all its jobs
                            if self.processors['processor_1'].is_finished():
                                # append task to 'finished tasks' list
                                self.finished_tasks.append(self.processors['processor_1'])
                            # case task did not finish all its jobs
                            else: 
                                # append task to 'not ready' list
                                self.not_ready_list.append(self.processors['processor_1'])
                        # case job did not finish all its sub jobs
                        else:
                            # append task to 'ready' list
                            self.ready_list.append(self.processors['processor_1'])
                            # processor gets cleared
                        self.processors['processor_1'] = None
            # no job is ready for execution
            else:
                if not added_to_wating_list:
                    # jump to next checkpoint
                    self.time += remaining_time 
        # make a copy of the scheduler after execution of job
        scheduler = Scheduler(scheduler=self)

        # update preemption status for all ready jobs
        for job in scheduler.ready_list:
            job.preempt = False

        return scheduler

    # generates all possible next scheduler states after execution of a job
    def generate_states(self):
        states = []
        # queue all newly released jobs to 'ready' list
        for taskset in self.tasksets:
            for task in taskset:
                # release job 
                task.release_job(self.time)
                # queue task to 'ready' list
                if task.released_job.release_time <= self.time:
                    self.queue(task)
        # check for all possible executable jobs
        # case processor is free
        index = 0
        if self.processors['processor_1'] is None:
            # traverse through 'ready' list, since all jobs in 'ready' list can be executed
            for index in range(0, len(self.ready_list)):
                # make copy of the current scheduler state to generate the possible next state on the copy
                temp_state = Scheduler(scheduler=self)

                # get job in copied scheduler and the current sub job of the recently released job 
                job = temp_state.ready_list[index]
                sub_job = job.get_current_sub_job()
                
                # case sub job is critical
                if sub_job.is_critical():

                    # case resource needed for execution of sub job is free and in waiting list or already acquired   
                    if job not in temp_state.waiting_lists[f'resource_{sub_job.resource_id+1}'] or (job in temp_state.waiting_lists[f'resource_{sub_job.resource_id+1}'] and temp_state.resources[f'resource_{sub_job.resource_id+1}'] == None) or temp_state.resources[f'resource_{sub_job.resource_id+1}'] == job:
                        # execute the sub job and append the state to the state list
                        task_id = job.task_id
                        state = temp_state.execute(job)
                        states.append((state, task_id))
                # case sub job is a non-critical sub job
                else:
                    # execute the sub job and append the state to the state list
                    task_id = job.task_id
                    state = temp_state.execute(job)
                    states.append((state, task_id))
        # case a job is holding processor
        else:
            # traverse the 'ready' list to get 
            for index in range(0, len(self.ready_list)):
                # make copy of the current scheduler state to generate the possible next state on the copy
                temp_state = Scheduler(scheduler=self)
                
                # get job in copied scheduler 
                job = temp_state.ready_list[index]
                # check if task arrived and its first job can preempt the job holding the processor
                # case task just arrived and can preempt
                if job.preempt:
                    # get current sub job of job to be executed
                    sub_job = job.get_current_sub_job()
                    # case sub_job is critical
                    if sub_job.is_critical():
                        
                        # case resource needed for sub job's execution is free and is first in waiting list or already acquired
                        if job not in temp_state.waiting_lists[f'resource_{sub_job.resource_id+1}'] or (job in temp_state.waiting_lists[f'resource_{sub_job.resource_id+1}'] and temp_state.resources[f'resource_{sub_job.resource_id+1}'] == None) or temp_state.resources[f'resource_{sub_job.resource_id+1}'] == job:
                            # execute the sub job and append the state to the state list
                            task_id = job.task_id
                            state = temp_state.execute(job)
                            states.append((state, task_id))
                    # case job is a non-critical sub job 
                    else:
                        # execute the sub job and append the state to the state list
                        task_id = job.task_id
                        state = temp_state.execute(job)
                        states.append((state, task_id))
            # generate state, where job which is currently holding the processor continues its execution 
            temp_state = Scheduler(scheduler=self)
            states.append((temp_state.execute(), self.processors['processor_1'].task_id))

        # case no job can be executed
        if len(states) == 0:
            # generate state where scheduler jumps to next checkpoint
            temp_state = Scheduler(scheduler=self)
            states = [(temp_state.execute(), 0)]
        return states

    ''' 
    def select_edf(self):
        if self.ready_list:
            earliest_deadline = self.ready_list[0]
            for periodic_task in self.ready_list:
                if earliest_deadline.tasks[earliest_deadline.index].deadline > periodic_task.tasks[periodic_task.index].deadline:
                    earliest_deadline = periodic_task
            return earliest_deadline
        return None
    '''

    # check if all tasks are scheduled
    def all_tasks_scheduled(self):
        return self.time >= self.hyper_period

    # check if there is an deadlock
    def is_deadlock(self):
        if self.processors['processor_1'] is not None:
            task = self.processors['processor_1'].tasks[self.processors['processor_1'].index]
            job = task.jobs[task.index]
            if job.is_critical():
                return self.resources[f'resource_{job.resource_id+1}'] != self.processors['processor_1'] and self.resources[f'resource_{job.resource_id+1}'] != None
        return False 

    # calculate the score of the current scheduler state
    def calculate_scores(self):
        score = 0

        deadline_miss = False
        for job in self.ready_list:
            if job.deadline < self.time:
                # score += 1000 * task.deadline
                deadline_miss = True
                break
                #score -= self.time - job.deadline
        if self.processors['processor_1']:
            if self.processors['processor_1'].deadline < self.time:
                deadline_miss = True
        
            
        if not deadline_miss:
            score = 1
        else:
            score = 0
        return score


    def schedule_loop(self):
        # print(self.to_string())

        # create MCTS instance
        mcts = MCTS()
        best_move = None
        # inputs = [9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 5, 6, 6, 6, 4, 5, 5, 5, 5, 5, 2, 1, 0, 3, 3, 4, 4, 3, 0, 0, 3, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 0, 5, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 6]
        # i = 0
        steps = 0
        while self.time < self.hyper_period and self.calculate_scores() > 0:
            # mcts selection of next job to execute
            # print(self.time < self.hyper_period)
            # print(best_move)
            states = deepcopy(self).generate_states()
            # print(f'# States : {len(states)}')
            if len(states) == 1:
                self = states[0][0]
                if best_move:
                    if self.to_string() not in best_move.children:
                        # print('Node does not exist')
                        new_node = TreeNode(self, states[0][1], best_move)
                        best_move.children[self.to_string()] = new_node 
                    else:
                        # print('node already exists')
                        new_node = best_move.children[self.to_string()]
                else:
                    # print('first Iteration')
                    new_node = TreeNode(self, states[0][1], best_move)
                best_move = new_node
            else:
                # case not first iteration of schedule loop
                if best_move:
                    # search for best move on already existing mct
                    best_move = mcts.search(self, current_node=best_move)
                    
                # case first iteration of schedule loop
                else: 
                    # search for best move on a new mct
                    best_move = mcts.search(self)
            self = best_move.scheduler
            self.to_array()
            # print(f'Children: {len(best_move.children)}')
            # print('--------------------------')
            # for children in best_move.children:
            #     print(best_move.children[children].scheduler.to_string())
            # print('--------------------------')
            # random selection of next task to execute
            # states = self.generate_states()
            # print(states)
            # self = random.choice(states)[0]

            # edf selection of next task to execute
            # periodic_task = self.select_edf()
            # self.execute_job(periodic_task)

            # choose job to execute section
            # states = self.generate_states()
            # print(states)
            # print()
            # if i < len(inputs):
            #     num = inputs[i]
            #     i += 1
            #     self = states[int(num)][0]
            # else:
            #     conf = 'no'
            #     while conf == 'no':
            #         num = input(f'Enter number from 0 - {len(states)-1}: ')
            #         while num == '' or int(num) >= len(states):
            #             print('inputted number is to high or wrong')
            #             num = input(f'Enter number from 0 - {len(states)-1}: ')

            #         print('_____________________________________________________________________________')
            #         print()
            #         print(states[int(num)][0].to_string())
            #         conf = input('Do you want to execute this state? ')
            #     self = states[int(num)][0]
            #     inputs.append(int(num))
            #     i += 1
            #     print(inputs)
           
            # self.execute(self.ready_list[int(num)])
            # print(f'BEST MOVE SCORE: {best_move.score}')
            # print(f'SCORE: {self.calculate_score()}')
            # input()


            # print(16*'-' + 'End of Timeslot' + 16*'-')
            # print(self.to_string())
            # input()
            print(self.time)

                
            # print(f'Score: {self.calculate_scores()}')
            steps += 1

        # print(f'Score: {self.calculate_scores()}')
        return steps, self

    def to_array(self):
        tasks_array = []

        for taskset in self.tasksets:
            for task in taskset:
                processor_id = 0
                waiting_priority = 0
                resource_id = 0

                
                for index, processor in enumerate(self.processors):
                    if task.released_job == self.processors[processor]:
                        processor_id = index + 1
                
                for index2, resource in enumerate(self.resources):
                    if task.released_job == self.resources[resource]:
                        resource_id = index2 + 1
                    else:
                        if task.released_job in self.waiting_lists[resource]:
                            for index3, job in enumerate(self.waiting_lists[resource]):
                                if task.released_job == job:
                                    resource_id = index2 + 1
                                    waiting_priority = index3 + 1

                tasks_array.append(task.released_job.to_list(processor_id, resource_id, waiting_priority, self.time))
                
        possible_moves_array = []
        states = deepcopy(self).generate_states()

        for _ in range(self.msets * self.ntasks):
            possible_moves_array.append(0)
               
        for state in states:
            if state[1] != 0:
                possible_moves_array[state[1]-1] = 1

        for index, move in enumerate(possible_moves_array):
            tasks_array[index].insert(0, [move])
        

        # for task in tasks_array:
        #     for value in task:
        #         scheduler_array.append(value)
        # taskset_tensor_2d = torch.Tensor(tasks_array)
        # taskset_flatten = torch.flatten(taskset_tensor_2d)
        # time = torch.Tensor([self.time])
        # scheduler_tensor = torch.cat((time, taskset_flatten),-1)
        # scheduler_normalize = scheduler_tensor / 10
        # print(torch.max(taskset_flatten))
        # print(torch.max(taskset_normalize))
        # print(scheduler_array)
        return np.asarray(tasks_array)

    def to_string(self) -> str:
        tasksets = f''
        for index_1, taskset in enumerate(self.tasksets):
            tasksets += f'Taskset {index_1+1}: '
            for index_2, task in enumerate(taskset):
                tasksets += f'Task {index_2+1}: {task.to_string()} '
            tasksets += '\n'

        readylist = f'Ready Tasks: '
        for index, task in enumerate(self.ready_list):
            readylist += f'\nTask {index+1}: {task.to_string()}'

        resources = f'Resources: \n'
        for index, resource in enumerate(self.resources):
            if self.resources[resource] is not None:
                resources += f'Resource {index+1}: {self.resources[resource].to_string()} \n' 
            else:
                resources += f'Resource {index+1}: {None} \n'
            
            resources += f'Waiting: '
            if len(self.waiting_lists[resource]) > 0:
                for job in self.waiting_lists[resource]:
                    resources += job.to_string()
                resources += f'\n'
            else:
                resources += f'None\n'
            
        processors = f'Executed Tasks: \n'
        for index, processor in enumerate(self.processors):
            if self.processors[processor] is not None:
                processors += f'Processor {index+1}: {self.processors[processor].to_string()} \n'
            else:
                processors += f'Processor {index+1}: {None} \n'

        return f'------------------------Schedule------------------------\n\nTasksets:\n{tasksets}\nTime: {self.time}\n{readylist}\n{resources}\n{processors}\n\n'

def generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, mod):
    for i in range(15, 20, 5):
        utli = float(i / 100)
        tasksets_name = './experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(
            utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max)
        tasksets = gen.generate(ntasks, msets, processors * utli, res_num, 0.5, c_min, c_max, mod)
        np.save(tasksets_name, tasksets)

    # for i in range (5, 100, 5):
    #     utli = float(i / 100)
    #     tasksets_periodic_name = './experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n'+str(ntasks)+'_m'+str(msets)+'_p'+str(processors)+ '_u' + str(utli) +'_r'+str(res_num)+'_s'+str(c_min)+'_l'+str(c_max)+'.npy'
    #     tasksets_periodic = np.load(tasksets_periodic_name, allow_pickle=True)
    #     # print(tasksets_periodic)
    #     job_periodic_name = './experiments/inputs/input_job_periodic/' + str(subset) + '/periodic_jobs_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(
    #         utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max) + '.npy'
    #     jobs_set = []
    #     for j in range(0, msets):
    #         jobs = []
    #         # print('\n')
    #         for k in range(0, ntasks):
    #             for b in range(0, 10, tasksets_periodic[j][k][len(tasksets_periodic[j][k])-1]):
    #                 # print(f'({j}, {k})')
    #                 # print(f'B:{b}')
    #                 # print(tasksets_periodic[j][k])
    #                 job = []
    #                 for s in range(len(tasksets_periodic[j][k])-1):
    #                     job.append(tasksets_periodic[j][k][s])
    #                 job.append(int(tasksets_periodic[j][k][len(tasksets_periodic[j][k])-1]))
    #                 job.append(int(b))
    #                 job.append(int(b + tasksets_periodic[j][k][len(tasksets_periodic[j][k])-1]))
    #                 job.append(int(k))
    #                 # now the job structure is [normal_1, critical, normal_2, resource_id, period, release time, deadline, task_id]
    #                 # print(f'Job: {job}')
    #                 jobs.append(job)
    #             # print(type(jobs))
    #         jobs_set.append(jobs)
    #         # print(type(jobs_set))
    #     np.save(job_periodic_name, jobs_set)

def load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, SPORADIC):
    # job_sets = []
    task_sets = []
    for i in range(15, 20, 5):
        utli = float(i / 100)
        tasksets_name = './experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max)  + '.npy'
        task_sets.append(np.load(tasksets_name, allow_pickle=True))

        # job_periodic_name = './experiments/inputs/input_job_periodic/' + str(subset) + '/periodic_jobs_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max) + '.npy'
        # job_sets.append(np.load(job_periodic_name, allow_pickle=True))

    '''
    tasksets = []
    taskset_id = 1
    for jobset in job_sets[0]:
        taskset = []
        periodic_task = []
        task_count = 0
        for job in jobset:
            jobs = []
            task_id = (job[-1]) + taskset_id
            deadline = job[-2]
            release_time = job[-3]
            period = job[-4]
            if SPORADIC:
                release_time += random.uniform(0,period)
            for index, item in enumerate(job):
                if index < len(job)-4:
                    jobs.append(Job(item[0], item[1], period))

            if len(periodic_task) != 0 and periodic_task[-1].task_id != task_id:
                taskset.append(PeriodicTask(periodic_task))
                periodic_task = [Task(jobs, period, release_time, deadline, task_id)]
            else:
                periodic_task.append(Task(jobs, period, release_time, deadline, task_id))
            if deadline == 10:
                task_count += 1
        taskset.append(PeriodicTask(periodic_task))
        tasksets.append(taskset)
        # print(f'Tasks: {task_count}')
        taskset_id += task_count'''

    # print(task_sets[0])
    task_id = 1
    tasksets = []
    for task_set in task_sets[0]:
        taskset = []
        for task in task_set:
            taskset.append(Task(period=task[-1], task_id=task_id, segments=task[:-1]))
            task_id += 1
        tasksets.append(taskset)

    return tasksets

if __name__ == '__main__':
    # tasks per taskset
    ntasks = 10
    # number of tasksets
    msets = 1
    # number of processors
    processors = 1
    # num of resources
    res_num = 1

    c_min = 0.05
    c_max = 0.1
    subset = 1

    # sporadic setting 0 = Periodic, 1 = Sporadic
    SPORADIC = 0
    mod = 0

    # Least common multiple of all Periods in tasksets
    wins = 0
    
    for i in range(10):
        generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, mod)
        tasksets = load_tasksets(ntasks=ntasks, msets=msets, processors=processors, res_num=res_num, c_min=c_min, c_max=c_max, subset=subset, SPORADIC=SPORADIC)
        settings = {
            'ntasks': ntasks, 
            'msets': msets, 
            'processors': processors,
            'res_num': res_num,
            'c_min': c_min,
            'c_max': c_max,
            'subset': subset,
            'SPORADIC': SPORADIC,
            'mod': mod
            }

        scheduler = Scheduler(tasksets, settings)
        steps, final_state = scheduler.schedule_loop()
        if final_state.calculate_scores() == 1:
            wins += 1

        print(f'episode {i}, final score {final_state.calculate_scores()}, steps {steps}')
    print(f'Won Games: {wins}')
    # generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset)
    # tasksets = load_tasksets(ntasks=ntasks, msets=msets, processors=processors, res_num=res_num, c_min=c_min, c_max=c_max, subset=subset, SPORADIC=SPORADIC)
    # settings = {
    #     'hyper_period': hyper_period,
    #     'ntasks': ntasks, 
    #     'msets': msets, 
    #     'processors': processors,
    #     'res_num': res_num,
    #     'c_min': c_min,
    #     'c_max': c_max,
    #     'subset': subset,
    #     'SPORADIC': SPORADIC,
    # }

    # scheduler = Scheduler(tasksets, settings)
    # for taskset in scheduler.tasksets:
    #     for task in taskset:
    #         list = task.released_job.to_list(0, 0, 0, scheduler.time)
    #         print(len(list))

    # # print('\n-------------------------------------\n')
    
    # scheduler.to_array()
    # scheduler.schedule_loop()