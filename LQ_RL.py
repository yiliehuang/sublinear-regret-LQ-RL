import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import math
from tqdm import tqdm
import sys
import scipy
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor



class LQ_RL_Simulator:
    def __init__(self, **config):
        self.N = config.get('N')

        self.A = config.get('A')
        self.B = config.get('B')
        self.C = config.get('C')
        self.D = config.get('D')
        self.Q = config.get('Q')
        self.H = config.get('H')
        self.x_0 = config.get('x_0')
        self.T = config.get('T')
        self.dt = config.get('dt')
        self.nt = round(self.T / self.dt)

        self.lr_rate = config.get('lr_rate')
        self.phi_1_rate = config.get('phi_1_rate')
        self.phi_2_rate = config.get('phi_2_rate')
        self.batch_rate = config.get('batch_rate')

        self.initial_n = config.get('initial_n')
        self.initial_phi_1 = config.get('initial_phi_1')
        self.initial_phi_2 = config.get('initial_phi_2')
        self.initial_lr1 = config.get('initial_lr1')
        self.initial_gamma = config.get('initial_gamma')
        self.initial_seed = config.get('initial_seed')
        self.initial_batch = config.get('initial_batch')

        # Initialize a RNG with the initial seed
        self.rng = np.random.default_rng(self.initial_seed)

        self.initialization()

        # keeping record
        self.x_list = []
        self.realized_value_function = []

    def initialization(self):
        self.phi_1 = self.initial_phi_1
        self.phi_2 = self.initial_phi_2
        self._n = self.initial_n  # n-th iteration

        self.phi_1_star = -(self.B + self.C * self.D) / self.D**2
        self.phi_2_star = 0

        self.update_hyper_parameter()

    def update_hyper_parameter(self):
        self.a_n1 = 1 / ((self._n + 1)**(self.lr_rate)) * self.initial_lr1

        self.m_n = self.initial_batch

        self.c_1n = 1.1 * abs(self.phi_1_star)
        self.b_n = max(1, (self._n + 1)**self.phi_2_rate / 1**self.phi_2_rate)

        self.gamma_n = self.initial_gamma

        self.phi_2 = self.initial_phi_2 / \
            (max(1, (self._n + 1)**self.phi_2_rate / 1**self.phi_2_rate))

    def show_parameters(self):
        # Define a list of parameter names you want to show
        parameters_to_show = [
            'phi_1', 'phi_2'
        ]

        print("Current parameters:")
        for param in parameters_to_show:
            print(f"{param}: {getattr(self, param, 'Not set')}")

    def get_value_function(self, x):
        return -0.5 * (x**2)

    def get_action(self, x):
        mean = self.phi_1 * x
        sd = np.sqrt(self.phi_2)
        action = self.rng.normal(mean, sd)  # Use the class's RNG
        return action

    def get_next_state(self, x, u):
        dW = np.sqrt(self.dt) * self.rng.normal()  # Use the class's RNG
        return x + (self.A * x + self.B * u) * self.dt + (self.C * x + self.D * u) * dW

    def get_one_trajectory(self):
        trajectory = []
        x = self.x_0
        for k in range(self.nt):
            u = self.get_action(x)
            trajectory.append((x, u))
            x = self.get_next_state(x, u)
        return trajectory

    def get_entropy(self):
        return 0.5 * np.log(self.phi_2) + 0.5 * np.log(2 * np.pi * np.e)

    def get_partial_pi_partial_phi1(self, x, u):
        return (u - self.phi_1 * x) * x / self.phi_2

    def calculate_phi_gradient(self):
        phi_1_grad_sum = 0
        for _ in range(self.m_n):
            trajectory = self.get_one_trajectory()

            # keeping records
            self.x_list.append([pair[0] for pair in trajectory])

            xs, us = zip(*trajectory)  # Extract states and actions
            xs, us = np.array(xs), np.array(us)
            # Prepare next states array, removing last element
            xs_next = np.roll(xs, -1)[:-1]

            common = self.get_value_function(xs_next) - self.get_value_function(xs[:-1]) -\
                0.5 * self.Q * xs[:-1]**2 * self.dt + \
                self.gamma_n * self.get_entropy() * self.dt

            phi_1_grad = np.sum(self.get_partial_pi_partial_phi1(
                xs[:-1], us[:-1]) * common)

            phi_1_grad_sum += phi_1_grad

        phi_1_grad_avg = phi_1_grad_sum / self.m_n

        return phi_1_grad_avg
    
    def clamp(self, value, a, b):
        return max(a, min(value, b))

    def update_phi_and_project(self):
        phi_1_grad = self.calculate_phi_gradient()
        self.phi_1 = self.clamp(
            self.phi_1 + self.a_n1 * phi_1_grad, -self.c_1n, self.initial_phi_1)

    def run_many_iterations(self):
        phi_1_list = [self.phi_1]
        phi_2_list = [self.phi_2]
        for self._n in range(self.initial_n, self.initial_n + self.N):
            self.update_hyper_parameter()
            self.update_phi_and_project()
            phi_1_list.append(self.phi_1)
            phi_2_list.append(self.phi_2)
        return phi_1_list, phi_2_list

    def get_optimal_value(self):
        return self.phi_1_star, self.phi_2_star

    def j_hat(self, phi_1, phi_2):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        Q = self.Q
        H = self.H
        x_0 = self.x_0
        T = self.T

        def function_a(phi_1):
            return 2*A + 2*phi_1*B + C*C + 2*C*D*phi_1 + D*D*phi_1*phi_1

        def function_f(phi_1):
            a_val = function_a(phi_1)
            common_exp = np.exp(a_val * T)
            res = np.where(np.isclose(a_val, 0, atol=1e-12),
                           x_0 * x_0 / 2 * (-H - Q * T),
                           1 / a_val / 2 * x_0 * x_0 * (Q - common_exp*Q - H*common_exp*a_val))
            return res

        def function_g(phi_1):
            a_val = function_a(phi_1)
            common_exp = np.exp(a_val * T)
            res = np.where(np.isclose(a_val, 0, atol=1e-12),
                           D * D * T / 4 * (-2 * H - Q * T),
                           (Q*T*a_val + Q + H*a_val - common_exp*Q - H*common_exp*a_val) * D * D / 2 / a_val / a_val)
            return res

        return function_f(phi_1) + phi_2 * function_g(phi_1)

    def get_optimal_j_hat(self):
        return self.j_hat(self.phi_1_star, self.phi_2_star)





if __name__ != '__main__':
    print("The following names are defined in this module:")
    for name in dir():
        if not name.startswith("__"):
            print(name)