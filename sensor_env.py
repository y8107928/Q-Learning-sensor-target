import tensorflow as tf
import numpy as np
import random

class SensorEnv:
    def __init__(self):

        self.sensor = [0,1,2,3,4]#传感器列表
        self.sensor_count = 5


        self.targets = [0,1,2,3,4]

        self.E = np.array([[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,0],[0,1,1,1,1],[0,0,0,1,1]])#传感器连通矩阵

        self.L = 1#网络共识迭代次数

        self.sensor_position = np.array([[1500,3900],[2700,3900],[1500,2700],[2700,2700],[2700,1500]])

        self.target_position_true = np.array([[1900,3100],[3000,3900],[1800,3500],[2200,3300],[2200,2100]])



        self.target_count = 5
        self.distance_limit = 1000#传感器最大检测距离

    #假设已经实现了网络共识，假设上述的真实目标位置就是网络共识后的目标位置
    def Observation_deviation(self,sensor_id,target_id):#target_list是网络共识后的，对应传感器观测到的目标位置
        sensor_p = self.sensor_position[sensor_id]
        target_p = self.target_position_true[target_id]
        distance =((sensor_p[0]-target_p[0])**2 + (sensor_p[1]-target_p[1])**2)**0.5
        deviation = 2 + (distance/100)*2
        return distance,deviation

    def get_or_not(self,sensor_id,target_id):#该函数判断目标是否在传感器的检测范围内
        distance,_ = self.Observation_deviation(sensor_id,target_id)
        if distance <= 1000:
            return True
        else:
            return False

    def target_be_get(self,sensor_id):#该函数得到某个传感器检测范围内的所有目标
        get_list = []
        for target_id in range(self.target_count):
            if self.get_or_not(sensor_id,target_id):
                get_list.append(target_id)
        return get_list

    def reset(self,sensor_id):#初始化状态，为每个传感器随机一个目标(范围内)
        get_list = self.target_be_get(sensor_id)
        random_index = random.randint(0, len(get_list)-1)
        return sensor_id,random_index,get_list[random_index]

    def get_state_start(self):#产生状态矩阵
        state_list = []
        for i in range(self.sensor_count):
            state_sub_list = []
            target_be_get = self.target_be_get(i)
            for j in target_be_get:
                sensor_target_list = j
                state_sub_list.append(sensor_target_list)
            state_list.append(state_sub_list)
        #print(state_list)
        listA = state_list[0]
        listB = state_list[1]
        listC = state_list[2]
        listD = state_list[3]
        listE = state_list[4]
        #print(listA)
        #print(listB)
        #print(listC)
        #print(listD)
        #print(listE)

        state_list = []
        for i in listA:
            for j in listB:
                for k in listC:
                    for l in listD:
                        for m in listE:
                            list_sub = [i,j,k,l,m]
                            state_list.append(list_sub)
        state_array = np.array(state_list)

        return state_array



