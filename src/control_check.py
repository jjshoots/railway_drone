import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from env.aviary import *

class Environment():
    """
    Wrapper for Aviary and Drone Classes with domain randomization
    """
    def __init__(self, rails_dir, drone_dir, plants_dir, max_steps=math.inf, flight_mode=0):
        self.max_steps = max_steps
        self.rails_dir = rails_dir
        self.drone_dir = drone_dir
        self.plants_dir = plants_dir
        self.flight_mode = flight_mode

        self.render = True

        self.reset()


    def reset(self):
        try:
            self.env.disconnect()
        except:
            pass

        self.step_count = 0

        self.env = Aviary(rails_dir=self.rails_dir, drone_dir=self.drone_dir, plants_dir=self.plants_dir, render=self.render)
        self.env.drone.set_mode(self.flight_mode)

        # clear argsv[0] message, I don't know why it works don't ask me why it works
        print ("\033[A                             \033[A")

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()
            self.env.drone.setpoint = np.array([0, 0, 0, 2])


    def get_state(self):
        return self.env.drone.state


    def step(self, target):
        """
        input is drone setpoint depending on flight mode as a np array
        output is whether the env has ended or not as a boolean
        """
        self.env.drone.setpoint = target
        self.env.step()
        self.step_count += 1

        return self.step_count <= self.max_steps



if __name__ == '__main__':
    env = Environment(
        rails_dir='models/rails/',
        drone_dir='models/vehicles/',
        plants_dir='models/plants/',
        max_steps=1000,
        flight_mode=2
        )

    # list holders
    av_states = []
    a_states = []
    lv_states = []
    l_states = []
    setpoints = []

    # initial setpoint and step function
    setpoint = np.array([0., 0., 0., 2.])
    while env.step(setpoint):
        av_states.append(env.get_state()[0])
        a_states.append(env.get_state()[1])
        lv_states.append(env.get_state()[2])
        l_states.append(env.get_state()[3])

        setpoints.append(setpoint.copy())

        if env.step_count >= 100:
            setpoint[3] = 2.

    # stack the states and setpoints
    av_states = np.stack(av_states, axis=0)
    a_states = np.stack(a_states, axis=0)
    lv_states = np.stack(lv_states, axis=0)
    l_states = np.stack(l_states, axis=0)

    setpoints = np.stack(setpoints, axis=0)
    x_axis = np.arange(av_states.shape[0]) / 240.

    fig0, axs0 = plt.subplots(3)
    fig0.suptitle('Angular Velocities')
    axs0[0].plot(x_axis, setpoints[:, 0])
    axs0[0].plot(x_axis, av_states[:, 0])
    axs0[1].plot(x_axis, setpoints[:, 1])
    axs0[1].plot(x_axis, av_states[:, 1])
    axs0[2].plot(x_axis, setpoints[:, 2])
    axs0[2].plot(x_axis, av_states[:, 2])
    plt.show(block=False)

    fig1, axs1 = plt.subplots(3)
    fig1.suptitle('Angular Position')
    axs1[0].plot(x_axis, setpoints[:, 0])
    axs1[0].plot(x_axis, a_states[:, 0])
    axs1[1].plot(x_axis, setpoints[:, 1])
    axs1[1].plot(x_axis, a_states[:, 1])
    axs1[2].plot(x_axis, setpoints[:, 2])
    axs1[2].plot(x_axis, a_states[:, 2])
    plt.show(block=False)

    fig2, axs2 = plt.subplots(3)
    fig2.suptitle('Linear Velocities')
    axs2[0].plot(x_axis, setpoints[:, 0])
    axs2[0].plot(x_axis, lv_states[:, 0])
    axs2[1].plot(x_axis, setpoints[:, 1])
    axs2[1].plot(x_axis, lv_states[:, 1])
    axs2[2].plot(x_axis, setpoints[:, 2])
    axs2[2].plot(x_axis, lv_states[:, 2])
    plt.show(block=False)

    fig3, axs3 = plt.subplots(3)
    fig3.suptitle('Linear Position')
    axs3[0].plot(x_axis, setpoints[:, 0])
    axs3[0].plot(x_axis, l_states[:, 0])
    axs3[1].plot(x_axis, setpoints[:, 1])
    axs3[1].plot(x_axis, l_states[:, 1])
    axs3[2].plot(x_axis, setpoints[:, 3])
    axs3[2].plot(x_axis, l_states[:, 2])
    plt.show(block=False)

    plt.show()
