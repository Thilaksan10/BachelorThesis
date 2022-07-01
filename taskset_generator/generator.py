import drs
import numpy as np

# Generates sets of tasks of the form 
# [[normal, -1], [critical, res_id], [normal, -1], [critical, res_id],..., period]

# mod used to control whether to generate the frame based or periodic tasks
# mod == 0: frame-based
# mod == 1: periodic
def generate(nsets, msets, utilization, num_resources, max_task_utli, critical_min, critical_max, mod):
    utilizations = []
    upper_bounds = [max_task_utli] * nsets
    lower_bounds = [0] * nsets
    
    # distribute utilization to every task for every set
    for _ in range(msets):
        utilizations.append(drs.drs(nsets, utilization, upper_bounds, lower_bounds)) 
    

    tasksets = []
    
    # bounds of the critical sections
    c_min = critical_min
    c_max = critical_max
    # number of resources
    num_res = num_resources
    # periods
    periods = [1, 2, 5, 10]

    for i in range(msets):
        taskset = []
        for j in range(nsets):
            task = []
            execution = utilizations[i][j]
            # distribute utilization of critical sections of task
            critical = np.random.uniform(c_min, c_max) * execution
            # calcultae normal execution of task
            normal = execution - critical

            # get number of critical sections
            num_critical = np.random.randint(2, 6)

            normal_sets = []
            critical_sets = []
            # generate 10 lists of non-critical sections and 10 lists of critical sections
            for s in range(10):
                normal_sets.append(drs.drs(n=num_critical, sumu=normal))
                critical_sets.append(drs.drs(n=num_critical, sumu=critical))
            
            # select randomly 1 list of non-critical section and critical section
            normal_set = normal_sets[np.random.randint(0,10)]
            critical_set = critical_sets[np.random.randint(0, 10)]

            # generate list of segments by creating tuples of (utilization, resource_id)
            for k in range(0, num_critical):
                resource_id = np.random.randint(0, num_res)
                task.append([normal_set[k], -1])
                task.append([critical_set[k], resource_id])

            # append segment [0, -1] to end task with a non-critical section
            task.append([0, -1])

            # framebased
            if (mod == 0):
                period = 1
            # else periodic tasks
            else:
                period = periods[np.random.randint(0, 4)]
                # calculate execution of every segment by multiplying utilization with period
                for k in range(0, 2*num_critical+1):
                    task[k][0] = task[k][0] * period
            
            # append period after segments list
            task.append(period)

            taskset.append(task)			
        tasksets.append(taskset)
    return tasksets


