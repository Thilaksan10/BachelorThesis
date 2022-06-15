import numpy as np
import taskset_generator.generator as gen
from check_schedulability import check_schedulability, scheduling_time
from agent import Agent
from ml_scheduler import Task, Scheduler
from scheduler_env import SchedulerEnv
import matplotlib.pyplot as plt

def generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, mod, iteration):
    tasksets_name = './experiments/evaluation/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(
        utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max) + '_i' + str(iteration)
    tasksets = gen.generate(ntasks, msets, processors * utli, res_num, 0.5, c_min, c_max, mod)
    np.save(tasksets_name, tasksets)

def load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, iteration):
    task_sets = []
    
    tasksets_name = './experiments/evaluation/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max) + '_i' + str(iteration) + '.npy'
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
        states = scheduler.generate_states()
        deadlines = []
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
        states = scheduler.generate_states()
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

if __name__ == '__main__':
    # tasks per taskset
    ntasks = 5
    # number of tasksets
    msets = 1
    # number of processors
    processors = 1
    # num of resources
    resources_no = [1, 2, 4, 8]

    c_min = 0.05
    c_max = 0.1
    subset = 1

    # sporadic setting 0 = Periodic, 1 = Sporadic
    SPORADIC = 0
    mod = 1

    iterations = 10
    min_utli = 5
    max_utli = 105

    generate = 0

    if generate:
        for res_num in resources_no:
            for i in range (min_utli, max_utli, 5):
                utli = float(i/100)
                for iteration in range(iterations):
                    generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, mod, iteration+1)
                    tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, iteration+1)
                    settings = {
                        'ntasks': ntasks, 
                        'msets': msets, 
                        'processors': processors,
                        'res_num': res_num,
                        'c_min': c_min,
                        'c_max': c_max,
                        'subset': subset,
                        'SPORADIC': SPORADIC,
                        'mod': mod,
                    }
                    scheduler = Scheduler(tasksets, settings)
                    print(scheduling_time(scheduler))
                    # while not check_schedulability(scheduler):
                    #     generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, iteration+1)
                    #     tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, iteration+1)
                    #     settings = {
                    #         'hyper_period': hyper_period,
                    #         'ntasks': ntasks, 
                    #         'msets': msets, 
                    #         'processors': processors,
                    #         'res_num': res_num,
                    #         'c_min': c_min,
                    #         'c_max': c_max,
                    #         'subset': subset,
                    #         'SPORADIC': SPORADIC,
                    #     } 
                    #     scheduler = Scheduler(tasksets, settings)
                    #     print(scheduling_time(scheduler))

    res_num = 1
    utli = 0.05
    
    tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, 1)
    settings = {
        'ntasks': ntasks, 
        'msets': msets, 
        'processors': processors,
        'res_num': res_num,
        'c_min': c_min,
        'c_max': c_max,
        'subset': subset,
        'SPORADIC': SPORADIC,
        'mod': mod,
    }

    scheduler = Scheduler(tasksets, settings)
    print(scheduler.to_string())
    env = SchedulerEnv(scheduler)
    agent = Agent(n_actions=env.action_shape, input_shape=env.observation_shape, alpha=1e-5, n_tasks=ntasks, m_sets=msets, policy_layer_dims=[512, 1024, 2048, 1024, 512])
    agent.load_models()
    print(agent.policy_network.checkpoint_dir)

    eval_list_agent = []

    for res_num in resources_no:
        eval_utli = []
        for i in range(min_utli, max_utli, 5):
            utli = float(i/100)
            won = 0
            for iteration in range(iterations):
                tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, iteration+1)
                settings = {
                    'ntasks': ntasks, 
                    'msets': msets, 
                    'processors': processors,
                    'res_num': res_num,
                    'c_min': c_min,
                    'c_max': c_max,
                    'subset': subset,
                    'SPORADIC': SPORADIC,
                    'mod': mod,
                }

                scheduler = Scheduler(tasksets, settings)
                env = SchedulerEnv(scheduler)
                # env.render()
                min_time = scheduling_time(scheduler)
                done = False
                score = 0
                observation = env.state
                while not done:
                    action = agent.choose_action_eval(observation.to_array())
                    # action = select_edf(observation)
                    # action = select_rate_monotonic(observation)
                    # print(action)
                    state_, reward, done, info = env.step(action)
                    score += reward
                    # env.render()
                    # if observation.to_string() == state_.to_string() and not info['invalid']:
                    #     input()
                    observation = state_
                if reward == 1:
                    won += 1
                if info['invalid']:
                    invalid = 1
                else:
                    invalid = 0
                # if min_time > scheduler.hyper_period and reward == 1:
                #     print(observation.to_string())
                #     for job in observation.ready_list:
                #         print(f'Task ID: {job.task_id}')
                #     print(observation.calculate_scores())

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, invalid {invalid}, final_score {reward}, cummulative score {score}, scheduling_time {min_time}')
            eval_utli.append(won/iterations)
        eval_list_agent.append(eval_utli)

    eval_list_edf = []

    for res_num in resources_no:
        eval_utli = []
        for i in range(min_utli, max_utli, 5):
            utli = float(i/100)
            won = 0
            for iteration in range(iterations):
                tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, iteration+1)
                settings = {
                    'ntasks': ntasks, 
                    'msets': msets, 
                    'processors': processors,
                    'res_num': res_num,
                    'c_min': c_min,
                    'c_max': c_max,
                    'subset': subset,
                    'SPORADIC': SPORADIC,
                    'mod': mod,
                }

                scheduler = Scheduler(tasksets, settings)
                env = SchedulerEnv(scheduler)
                
                # env.render()
                min_time = scheduling_time(scheduler)
                done = False
                score = 0
                observation = env.state
                while not done:
                    # action = agent.choose_action_eval(observation.to_array())
                    action = select_edf(observation)
                    # action = select_rate_monotonic(observation)
                    # print(action)
                    state_, reward, done, info = env.step(action-1)
                    score += reward
                    # env.render()
                    # if observation.to_string() == state_.to_string() and not info['invalid']:
                    #     input()
                    
                    observation = state_
                if reward == 1:
                    won += 1
                # if min_time > scheduler.hyper_period and reward == 1:
                #     print(observation.to_string())
                #     for job in observation.ready_list:
                #         print(f'Task ID: {job.task_id}')
                #     print(observation.calculate_scores())

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, final_score {reward}, cummulative score {score}, scheduling_time {min_time}')
            eval_utli.append(won/iterations)
        eval_list_edf.append(eval_utli)

    eval_list_rm = []

    for res_num in resources_no:
        eval_utli = []
        for i in range(min_utli, max_utli, 5):
            utli = float(i/100)
            won = 0
            for iteration in range(iterations):
                tasksets = load_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, utli, SPORADIC, iteration+1)
                settings = {
                    'ntasks': ntasks, 
                    'msets': msets, 
                    'processors': processors,
                    'res_num': res_num,
                    'c_min': c_min,
                    'c_max': c_max,
                    'subset': subset,
                    'SPORADIC': SPORADIC,
                    'mod': mod,
                }

                scheduler = Scheduler(tasksets, settings)
                env = SchedulerEnv(scheduler)
                # env.render()
                min_time = scheduling_time(scheduler)
                done = False
                score = 0
                observation = env.state
                while not done:
                    # action = agent.choose_action_eval(observation.to_array())
                    # action = select_edf(observation)
                    action = select_rate_monotonic(observation)
                    # print(action)
                    state_, reward, done, info = env.step(action-1)
                    score += reward
                    # env.render()
                    # if observation.to_string() == state_.to_string() and not info['invalid']:
                    #     input()
                    observation = state_
                    
                if reward == 1:
                    won += 1
                # if min_time > scheduler.hyper_period and reward == 1:
                #     print(observation.to_string())
                #     for job in observation.ready_list:
                #         print(f'Task ID: {job.task_id}')
                #     print(observation.calculate_scores())

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, final_score {reward}, cummulative score {score}, scheduling_time {min_time}')
            eval_utli.append(won/iterations)
        eval_list_rm.append(eval_utli)
   
    print(eval_list_agent)
    print(eval_list_edf)
    print(eval_list_rm)
    
    for k, _ in enumerate(resources_no): 
        plt.title(f'Acceptance Rate of Agent for {2**k} resources')
        plt.plot(range(min_utli, max_utli, 5), eval_list_agent[k])
        plt.plot(range(min_utli, max_utli, 5), eval_list_edf[k])
        plt.plot(range(min_utli, max_utli, 5), eval_list_rm[k])
        plt.xlabel('Utilization in %')
        plt.ylabel('Acceptance Rate')
        plt.legend(['Agent', 'EDF', 'RM'])
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

    eval_mix_edf = []
    for m, _ in enumerate(eval_list_edf[0]):
        sum = 0
        for eval_utli in eval_list_edf:
            sum += eval_utli[m]
        sum = sum / len(eval_list_edf)
        eval_mix_edf.append(sum)

    eval_mix_rm = []
    for m, _ in enumerate(eval_list_rm[0]):
        sum = 0
        for eval_utli in eval_list_rm:
            sum += eval_utli[m]
        sum = sum / len(eval_list_rm)
        eval_mix_rm.append(sum)

    plt.title(f'Acceptance Rate of Agent for mix resources')
    plt.plot(range(min_utli, max_utli, 5), eval_mix_agent)
    plt.plot(range(min_utli, max_utli, 5), eval_mix_edf)
    plt.plot(range(min_utli, max_utli, 5), eval_mix_rm)
    plt.xlabel('Utilization in %')
    plt.ylabel('Acceptance Rate')
    plt.legend(['Agent', 'EDF', 'RM'])
    if SPORADIC:
        plt.savefig(f'eval/sporadic/eval-sporadic-mixed-resources.png')
    elif not mod:
        plt.savefig(f'eval/framebased/eval-framebased-mixed-resources.png')
    else:
        plt.savefig(f'eval/periodic/eval_periodic-mixed-resources.png')
    
    plt.show()
    