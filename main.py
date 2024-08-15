# -*- coding: UTF-8 -*-
#对baseline wscnet sentinet spatial channel_wise spatial+channel_wise的IAPS和NAPS做实验
import torch
import common_solver as c_solver
import net as net
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner
import multiprocessing



if __name__ == '__main__':
    # multiprocessing.set_start_method("spawn",True)
    r=runner.runner()

    task_AllData={
    "task_name":"AllData_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"classify_mode":True}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./Data/Alldata","dataset":"AllData","batch_size":64,"epochs":300}},
    "mem_use":[10000,10000]
    }

    task_Person={
    "task_name":"Person_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":140,"classify_mode":True,"constrain_value":10}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./Data/Person","dataset":"Person","batch_size":32,"epochs":150}},
    "mem_use":[10000,10000]
    }

    task_Animal={
    "task_name":"Animal_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":140,"classify_mode":True,"constrain_value":10}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./Data/Animal","dataset":"Animal","batch_size":32,"epochs":150}},
    "mem_use":[10000,10000]
    }

    task_Plant={
    "task_name":"Plant_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":140,"classify_mode":True,"constrain_value":10}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./Data/Plant","dataset":"Plant","batch_size":32,"epochs":150}},
    "mem_use":[10000,10000]
    }

    task_Environment={
    "task_name":"Environment_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":140,"classify_mode":True,"constrain_value":10}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./Data/Environment","dataset":"Environment","batch_size":32,"epochs":150}},
    "mem_use":[10000,10000]
    }

    task_Mix={
    "task_name": "Mix_experiment",
    "solver": {"class": c_solver.common_solver,"params": {}},
    "kernel": {"class": regression_core.regression_processer, "params": {"epoch_lr_decay":140,"classify_mode": True, "constrain_value": 10}},
    "models": [{"class": net.SENet_channel_wise_with_attention, "params": {"C": 3, "activate_type": "sigmoid_res"}}],
    "optimizers": {"function": regression_core.optimizers_producer_classify, "params": {"lr_base": 0.001, "lr_fc": 0.01, "weight_decay": 0.0005, "paral": True}},
    "dataset": {"function": dataset.param_dict_producer, "params": {"path": "./Data/Mix", "dataset": "Mix", "batch_size": 32, "epochs": 150}},
    "mem_use": [10000, 10000]
    }
    tasks=[]

    tasks.append(task_AllData)
    tasks.append(task_Person)
    tasks.append(task_Animal)
    tasks.append(task_Plant)
    tasks.append(task_Environment)
    tasks.append(task_Mix)

    r.generate_tasks(tasks)
    r.main_loop()
