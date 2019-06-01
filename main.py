import numpy as np

import random

from sensor_env import SensorEnv

'''
1.初始化状态state,初始化q表
2.查找每个传感器范围内的目标
3.随机跟踪一个目标（1000m范围内），第一步随机动作
4.根据状态得到回报
5.更新Q表
6.利用Q表选择下一步动作
'''
#这个函数应该生成的是，每个智能体的动作的概率？？？
def epsilon_greedy_policy(Q,epsilon):
    seed = random.random()
    if epsilon>=seed:
        action_index = np.argmax(Q)
    else:
        action_index = random.randint(0,len(Q)-1)
    return action_index



s = SensorEnv()
action_space = s.get_state_start()
targets = s.targets
sensors = s.sensor
#print(action_space)
EPSILON = 0.2#贪婪策略参数
DISCOUNT_FACTOR = 0.6#折扣因子
ALPHA = 0.5#更新步长
N = 1000
Q = np.zeros(len(action_space))#定义Q表，Q表中只有动作空间
count = 10

#policy = epsilon_greedy_policy(Q, EPSILON, len(action_space))#策略是用易普希龙策略在Q表中选值
for i_episode in range(N):#执行步骤迭代信息
    for t in range(count):
        action_index = epsilon_greedy_policy(Q,EPSILON)#动作选择由动作概率得到
        action = action_space[action_index]#在动作空间中选择动作

        dealt = np.zeros(len(targets))#尺寸为目标数的矩阵
        for i in range(len(targets)):#遍历每个目标
            index = i#这个索引应该是从0一直到目标数
            if index in action:
                dealt[i] = 1#应该是目标是否被集中跟踪，被集中跟踪就是1
        if np.sum(dealt) == len(targets):#所有的目标都被集中跟踪到，即每个传感器跟踪到了每个目标
            done = True
            r = +10
        else:
            done = False
            r = -1

        powerNeed = 0

        for i in range(len(action)):
            target_id = action[i]
            sensor_id = sensors[i]
            distance,deviation = s.Observation_deviation(sensor_id,target_id)
            powerNeed += distance*0.01
        r = r + (powerNeed * -1)#根据耗电量修改回报

        best_next_action_index = np.argmax(Q)#利用贪婪策略选择下一步动作
        td_target = r + DISCOUNT_FACTOR * Q[best_next_action_index]
        td_delta = td_target - Q[action_index]
        Q[action_index] += ALPHA * td_delta#利用时许差分法更新Q表

        if done:
            break

    bestActionInd = np.argmax(Q)

print(action_space[bestActionInd])
