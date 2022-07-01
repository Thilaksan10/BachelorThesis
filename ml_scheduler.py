from copy import deepcopy
from mcts import TreeNode, MCTS
from math import gcd
import numpy as np
import random
import taskset_generator.generator as gen

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

    # returns list with important details like period and remaining time till deadline of Job
    def to_list(self, processor, resource, waiting_priority, time, max_sub_job_count=13):
        list = [[processor], [resource], [waiting_priority], [self.period], [self.release_time-time], [self.deadline-time]]
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
        return list

    def to_string(self):
        sub_jobs = f''
        for sub_job in self.sub_jobs:
            sub_jobs += f' {sub_job.to_string()} '
        return f'P: {self.period}, R: {self.release_time}, D: {self.deadline}, ID: {self.task_id} \n Jobs: {sub_jobs}'

class Task:
    def __init__(self, period, task_id, segments, SPORADIC):
        self.period = period
        self.task_id = task_id
        self.segments = []
        self.SPORADIC = SPORADIC
        for segment in segments:
            self.segments.append(SubJob(utilization=segment[0], resource_id=segment[1]))
        # sporadic tasks add random number to release time using uniform distribution
        if SPORADIC:
            release_time = 0 + random.uniform(0, self.period)
            self.released_job = Job(self.segments, self.period, release_time, release_time + self.period, self.task_id)
        else:
            self.released_job = Job(self.segments, self.period, 0, self.period, self.task_id)

    # checks if recently released job is ready to be added to the ready list
    def release_job(self, time):
        if self.released_job.is_finished():
            if time >= self.released_job.release_time + self.period:
                if not self.SPORADIC:
                    release_time = self.released_job.deadline
                else:
                    release_time = self.released_job.deadline + random.uniform(0, self.period)
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

        self.priority_ceilings = self.assign_priority_ceilings()
        self.hyper_period = self.calculate_hyper_period()

    # hyperperiod is defined as least common multiple of all periods
    def calculate_hyper_period(self):
        periods = []
        max = 1
        for taskset in self.tasksets:
            for task in taskset:
                if max < task.period:
                    max = task.period
                periods.append(task.period)
        lcm = 1
        for period in periods:
            lcm = lcm * period // gcd(lcm, period)

        return lcm

    # check if a new job is released
    def new_job_released(self, previous_state):
        return len(previous_state.ready_list) < len(self.ready_list)

    # recently released job of the given task acquires resource
    def lock_resource(self, job):
        sub_job = job.get_current_sub_job()
        self.resources[f'resource_{sub_job.resource_id + 1}'] = job

    # recently released job of the given task releases resource
    def release_resource(self, job):
        sub_job = job.get_current_sub_job()
        self.resources[f'resource_{sub_job.resource_id + 1}'] = None

    # queues task to the ready list
    def queue(self, task):
        # append task to ready list
        if task.released_job not in self.processors.values() and task.released_job not in self.ready_list and not task.released_job.is_finished():
            self.ready_list.append(task.released_job)
            # job arrived and elligible to preempt current executing job
            task.get_current_job().preempt = True

    # returns the arrival time of the next task 
    def get_next_task_arrival(self):
        next_arrival = float('inf')
        # iterate through list of tasksets to find the earliest released task
        for taskset in self.tasksets:
            for task in taskset:
                # released job of task finished execution 
                if task.get_current_job().is_finished():
                    # calculate next job release
                    next_release = task.get_current_job().release_time + task.period
                    if next_release < self.time:
                        next_release = float('inf')
                    next_arrival = min(next_arrival, next_release)
                # task generated a job to release but did not release job
                elif task.get_current_job().release_time > self.time:
                    next_arrival = min(task.get_current_job().release_time, next_arrival)
                # released job of task is not finished
                elif not task.get_current_job().is_finished():
                    next_arrival = min(task.get_current_job().deadline, next_arrival)
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
                    if job in self.waiting_lists[f'resource_{sub_job.resource_id + 1}']:
                        self.waiting_lists[f'resource_{sub_job.resource_id + 1}'].remove(job)
                    self.lock_resource(job)
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
        # return the state, which is the result of executing the task with lowest period
        return states[min_arg][0]

    # assign priority ceilings for all resources
    def assign_priority_ceilings(self):
        priority_ceilings = [10 for _ in range(self.res_num)]

        for id, _ in enumerate(priority_ceilings):
            for taskset in self.tasksets:
                for task in taskset:
                    for segment in task.segments:
                        if segment.resource_id == id:
                            priority_ceilings[id] = min(priority_ceilings[id], task.period)
        return priority_ceilings

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
        # return the state, which is the result of executing the task with lowest period
        return states[min_arg][0]
        
     
    def select_edf(self):
        states = deepcopy(self).generate_states()
        min_arg = 0
        min_deadline = self.hyper_period + 1
        for index, state in enumerate(states):
            for taskset in self.tasksets:
                for task in taskset:
                    if task.task_id == state[1]:
                        if min_deadline > task.released_job.deadline:
                            min_arg = index
                            min_deadline = task.released_job.deadline
                        break

        return states[min_arg][0]

    def select_rate_monotonic(self):
        states = deepcopy(self).generate_states()
        min_arg = 0
        min_period = self.hyper_period + 1
        for index, state in enumerate(states):
            for taskset in self.tasksets:
                for task in taskset:
                    if task.task_id == state[1]:
                        if min_period > task.period:
                            min_arg = index
                            min_period = task.period
                        break

        return states[min_arg][0]
    

    # check if hyperperiod reached
    def hyper_period_reached(self):
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
            if job.deadline <= self.time:
                # score += 1000 * task.deadline
                deadline_miss = True
                break
                #score -= self.time - job.deadline
        if self.processors['processor_1']:
            if self.processors['processor_1'].deadline <= self.time:
                deadline_miss = True
        
        
        if not deadline_miss:
            score = 1
        else:
            score = -2
        return score

    # turn scheduler object in to a numpy array
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
                possible_moves_array[state[1] % self.ntasks -1] = 1

        for index, move in enumerate(possible_moves_array):
            tasks_array[index].insert(0, [move])
        
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

# generate tasksets with given settings
def generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, mod):
    # set the minimum and maximum utilization 
    for i in range(95, 100, 5):
        utli = float(i / 100)
        tasksets_name = './experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(
            utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max)
        tasksets = gen.generate(ntasks, msets, processors * utli, res_num, 0.5, c_min, c_max, mod)
        np.save(tasksets_name, tasksets)

# load tasksets with given settings and covert into 2D list of task objects 
def load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, SPORADIC):
    task_sets = []
    # set the minimum and maximum utilization
    for i in range(95, 100, 5):
        utli = float(i / 100)
        tasksets_name = './experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max)  + '.npy'
        task_sets.append(np.load(tasksets_name, allow_pickle=True))
    
    task_id = 1
    tasksets = []
    for task_set in task_sets[0]:
        taskset = []
        for task in task_set:
            taskset.append(Task(period=task[-1], task_id=task_id, segments=task[:-1], SPORADIC=SPORADIC))
            task_id += 1
        tasksets.append(taskset)

    return tasksets