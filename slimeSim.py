# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:36:33 2021

@author: jerem
"""
import cv2

from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause

import numpy as np
import math
import random

import time
from numba import njit, jit, cuda

countF, countL, countR = 0,0,0

@njit
def rotate(vx,vy,changeAngle):
    alpha = 0.0174533 * changeAngle
    return vx * math.cos(alpha) - vy * math.sin(alpha), vx * math.sin(alpha) + vy * math.cos(alpha)
def create_circular_mask(h, w, center, radius):

    Y, X = np.ogrid[:h, :w]
    dist_from_center = (X - center[0])*(X - center[0]) + (Y-center[1])*(Y-center[1])

    mask = dist_from_center <= radius*radius
    return mask
@jit
def eight_neighbor_average_convolve2d(x):
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    neighbor_sum = convolve2d(
        x, kernel, mode='same',
        boundary='fill', fillvalue=0)
    num_neighbor = convolve2d(
        np.ones(x.shape), kernel, mode='same',
        boundary='fill', fillvalue=0)
    return neighbor_sum / num_neighbor

#@jit
SIZE = 3.5
def sensor(M, x, y, vx, vy, w, h):

    MUL = 2.2 * 4/SpeedFactor
    Sensor = []
    Sappend = Sensor.append
    nx, ny = x + vx * MUL, y + vy * MUL

    for s in [-1, 1]:
        sx, sy = rotate(vx, vy, - s * 100)

        mask = create_circular_mask(h, w , (nx + sx*MUL, ny + sy*MUL), SIZE)
        Mcopy = M.copy()
        Mcopy[~mask] = 0

        Value = np.sum(Mcopy)
        Sappend(Value)


    mask = create_circular_mask(h, w, (nx, ny), SIZE)


    Mcopy = M.copy()

    t0 = time.perf_counter()
    Mcopy[~mask] = 0
    ValueCenter = np.sum(Mcopy)
    #print("sensor :", (time.perf_counter()-t0)*1000, "ms")


    Sensor0 = Sensor[0]
    Sensor1 = Sensor[1]
    steering = 0
    if ValueCenter >= Sensor1 and ValueCenter >= Sensor0:
        return 0

    elif Sensor[1] > Sensor[0] and Sensor[1] > ValueCenter and Sensor[1] > 0.1:
        #TURN RIGHT
        #print("=>R")
        return -30

    elif Sensor[0] > 0.1:
        #TURN LEFT
        #print("=>L")
        return 30
    return 0




#@njit
def update_M(M, x, y):
    i,j = 0,0
    M[(int(y) + i)%h, (int(x) + j)%w] = 1
    # for i in np.arange(-agentWidth, agentWidth):
    #     for j in np.arange(-agentWidth, agentWidth):
    #         M[(int(y) + i)%h, (int(x) + j)%w] = 1
    #print("up", (time.perf_counter() - t0)*1000,"ms")
    return M
@njit
def find_direction(M, x, y, vx, vy):
    range = 5
    imax, jmax  = int(x), int(y)
    for i in np.arange(-range, range):
        for j in np.arange(-range, range):
            if j == 0 or i == 0: break
            value = M[int(x)+i,int(y)+j]
            if value > 0.1 and value > M[imax, jmax ]:
                imax, jmax = i, j
    angle = math.atan2(jmax - y, imax - x) - math.atan2(vy, vx)
    return angle

def outOfBound(x,y, vx, vy):
    nx, ny = x + vx, y + vy
    if not (nx > 0 and nx < w and ny > 0 and ny < h):
        return True
    else:
        return False

R = 0.75

w,h = int(320*R),int(180*R)
print("RESOLUTION :", w,h)
M = np.zeros((h,w))
#M[:,86:100] = 1
#window = ax.imshow(np.random.random((h,w)),cmap = "gray", origin = 'lower')

CircleCenter = plt.Circle((0,0), SIZE, color='orange', fill=False, alpha = 0.25)
class agent:
    def __init__(self,x, y, vx, vy):
        self.vx, self.vy = vx, vy
        self.x, self.y = x, y
        #self.Vec = plt.quiver(x, y, vx, vy, color = 'g', width = 0.002, alpha = 1)
        #plt.close()



#INIT
"""AGENTS"""

nbAgents = 500
SpeedFactor = 1
agentWidth = 1
agents = []
for _ in range(nbAgents):
    angle = random.random() * 2 * math.pi
    vx, vy = SpeedFactor*math.cos(angle), SpeedFactor*math.sin(angle)

    a = agent(w/2 + random.uniform(-w/2,w/2),
          h/2 + random.uniform(-h/2,h/2),
          vx, vy)
    agents.append(a)


"""Random"""
mu, sigma = 0, 0
random.seed(12)
np.random.seed(10)

"""Simu"""
a0 = 0.95
A = np.full((h,w), a0)
tmax = 2000
dt = tmax / tmax
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', w*3,h*3)

for t in range(tmax):
    t0 = time.perf_counter()

    Sum = 0
    Sum2 = 0
    #M = eight_neighbor_average_convolve2d(M)
    for agent in agents:
        tin = time.perf_counter()

        dtheta = sensor(M, agent.x, agent.y, agent.vx, agent.vy, w, h)
        changeAngle = np.random.normal(mu, sigma) + dtheta


        agent.vx, agent.vy = rotate(agent.vx, agent.vy, changeAngle)

        agent.x += agent.vx
        agent.y += agent.vy
        Sum += (time.perf_counter() - tin)*1000 #FIRST TIME


        if outOfBound(agent.x,agent.y, agent.vx, agent.vy):
            agent.vx, agent.vy = rotate(agent.vx, agent.vy, 45)
            continue

        M = update_M(M, agent.x, agent.y)


    Tmoy = Sum/len(agents)
    M *= A



    # window.set_data(M)
    # draw(), pause(0.001)
    cv2.imshow("image", M)
    cv2.waitKey(1)

    print("tin :", Tmoy, "ms")
    print(countL, countF, countR)
    print(t,"time :", round((time.perf_counter() - t0)*1000,2), "ms")

