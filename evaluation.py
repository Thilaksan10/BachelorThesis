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
    resources_no = [1, 2, 4, 8]

    c_min = 0.05
    c_max = 0.1
    subset = 2

    # sporadic setting 0 = Periodic, 1 = Sporadic
    SPORADIC = 0
    mod = 0

    iterations = 10
    min_utli = 5
    max_utli = 105

    generate = 1

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
    }

    scheduler = Scheduler(tasksets, settings)
    env = SchedulerEnv(scheduler)
    agent = Agent(n_actions=env.action_shape, input_shape=env.observation_shape, alpha=1e-5, n_tasks=ntasks, m_sets=msets, policy_layer_dims=[512, 1024, 2048, 1024, 512])
    agent.load_models()
    print(agent.policy_network.checkpoint_dir)

    eval_list = []

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
                }

                scheduler = Scheduler(tasksets, settings)
                env = SchedulerEnv(scheduler)
                min_time = scheduling_time(scheduler)
                done = False
                score = 0
                observation = env.state
                while not done:
                    action = agent.choose_action_eval(observation.to_array())
                    state_, reward, done, info = env.step(action)
                    score += reward
                    observation = state_
                if reward == 1:
                    won += 1
                if min_time > scheduler.hyper_period and reward == 1:
                    print(observation.to_string())
                    for job in observation.ready_list:
                        print(f'Task ID: {job.task_id}')
                    print(observation.calculate_scores())
                    # input()

                print(f'res_num {res_num}, utilization {utli}, taskset {iteration}, final_score {reward}, cummulative score {score}, scheduling_time {min_time}')
            eval_utli.append(won/iterations)
        eval_list.append(eval_utli)
   
    agent.save_models()
    print(eval_list)
    for k, eval_utli in enumerate(eval_list): 
        plt.title(f'Acceptance Rate of Agent for n ressources')
        plt.plot(range(5, max_utli, 5), eval_utli)
        plt.xlabel('Utilization in %')
        plt.ylabel('Acceptance Rate')
    plt.legend(resources_no)
    plt.savefig('eval_framebased.png')
    plt.show()