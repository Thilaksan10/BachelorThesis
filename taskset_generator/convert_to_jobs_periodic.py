from __future__ import division
import numpy as np
from io import StringIO
import os
import math
import sys
import getopt
import csv

def main(argv):
    ntasks = 2
    msets = 3
    processors = 1
    # num of resources
    res_num = 1
    c_min = 0.05
    c_max = 0.1
    min_period = 1
    max_period = 10
    subset = 1

    try:
        opts, args = getopt.getopt(argv, "hn:m:p:r:s:l:t:",
                                   ["ntasks=", "msets=", "processors", "res_num=", "c_min=", "c_max=", "subset="])
    except getopt.GetoptError:
        print('tasksets_generater.py -n <n tasks for each set> -m <m tasksets> -p <num of processors> -r <num of resources> -s <min length of the critical section> -l <max length of the critical section>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('tasksets_generater.py -n <n tasks for each set> -m <m tasksets> -p <num of processors> -r <num of resources> -s <min length of the critical section>-l <max length of the critical section>')
            sys.exit()
        elif opt in ("-n", "--ntasks"):
            ntasks = int(arg)
        elif opt in ("-m", "--msets"):
            msets = int(arg)
        elif opt in ("-p", "--processors"):
            processors = int(arg)
        elif opt in ("-r", "--res_num"):
            res_num = int(arg)
        elif opt in ("-s", "--c_min"):
            c_min = float(arg)
        elif opt in ("-l", "--c_max"):
            c_max = float(arg)
        elif opt in ("-t", "--subset"):
            subset = int(arg)

    for i in range (5, 30, 5):
        utli = float(i / 100)
        tasksets_periodic_name = '../experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n'+str(ntasks)+'_m'+str(msets)+'_p'+str(processors)+ '_u' + str(utli) +'_r'+str(res_num)+'_s'+str(c_min)+'_l'+str(c_max)+'.npy'
        tasksets_periodic = np.load(tasksets_periodic_name, allow_pickle=True)
        print(tasksets_periodic)
        job_periodic_name = '../experiments/inputs/input_job_periodic/' + str(subset) + '/periodic_jobs_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(
            utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max) + '.npy'
        jobs_set = []
        for j in range(0, msets):
            jobs = []
            print('\n')
            for k in range(0, ntasks):
                for b in range(0, 10, tasksets_periodic[j][k][len(tasksets_periodic[j][k])-1]):
                    print(f'({j}, {k})')
                    print(f'B:{b}')
                    print(tasksets_periodic[j][k])
                    job = []
                    for s in range(len(tasksets_periodic[j][k])-1):
                        job.append(tasksets_periodic[j][k][s])
                    job.append(int(tasksets_periodic[j][k][len(tasksets_periodic[j][k])-1]))
                    job.append(int(b))
                    job.append(int(b + tasksets_periodic[j][k][len(tasksets_periodic[j][k])-1]))
                    job.append(int(k))
                    # now the job structure is [normal_1, critical, normal_2, resource_id, period, release time, deadline, task_id]
                    print(f'Job: {job}')
                    jobs.append(job)
                # print(type(jobs))
            jobs_set.append(jobs)
            # print(type(jobs_set))
        np.save(job_periodic_name, jobs_set)

if __name__ == "__main__":
    main(sys.argv[1:])
