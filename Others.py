import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import math
from tqdm.auto import tqdm
import sys
import os
import scipy
from scipy.stats import linregress
import statsmodels.api as sm
from scipy.optimize import minimize




class Model_Based_LQ_Simulator:
    def __init__(self, **config):
        self.N = config.get('N')

        self.real_A = config.get('real_A')
        self.real_B = config.get('real_B')
        self.real_C = config.get('real_C')
        self.real_D = config.get('real_D')

        self.A = config.get('initial_A')
        self.B = config.get('initial_B')
        self.C = config.get('initial_C')
        self.D = config.get('initial_D')
        self.Q = config.get('Q')
        self.H = config.get('H')
        self.x_0 = config.get('x_0')
        self.T = config.get('T')

        self.initial_m_n = config.get('initial_m_n')
        self.initial_nt = config.get('initial_nt')
        self.initial_v_n = config.get('initial_v_n')

        self.initial_seed = config.get('initial_seed')
        self.rng = np.random.default_rng(self.initial_seed)

        # record results
        self.A_list = [self.A]
        self.B_list = [self.B]
        self.C_list = [self.C]
        self.D_list = [self.D]
        self.m_n_list = []

        # other misc
        self.phi_1_list = []
        self.mat = np.identity(2)  # 2x2 system for A,B
        self.vec = np.zeros(2)

        # ----- Partial sums for the polynomial objective -----
        self.S_alpha2 = 0.0
        self.S_alphaA = 0.0
        self.S_alphaB = 0.0
        self.S_alphaG = 0.0
        self.S_A2     = 0.0
        self.S_B2     = 0.0
        self.S_G2     = 0.0
        self.S_AB     = 0.0
        self.S_AG     = 0.0
        self.S_BG     = 0.0

        self.initialization()  # your usual initialization steps


    def initialization(self):
        self._n = 0
        self.phi_1_star = -(self.real_C*self.real_D + self.real_B) / self.real_D**2
        #self.alpha = 0.5
        self.phi_1 = -(self.C*self.D + self.B) / self.D**2
        #self.L = np.ceil(np.log2(self.N / self.initial_m_n + 1)) - 1
        
    def update_hyper_parameter(self):
        self._n += 1
        
        self.m_n = 1
        self.v_n = self.initial_v_n / self._n

        self.nt = 100
        self.dt = self.T / self.nt
        
        new_phi_1 = -(self.C*self.D + self.B) / self.D**2
        self.phi_1 = self.ema_update(self.phi_1, new_phi_1)
        self.phi_1 = self.clamp(self.phi_1, -2.2, -0.5)


    def show_parameters(self):
        # Define a list of parameter names you want to show
        parameters_to_show = [
            'A', 'B', 'C', 'D', 'L', '_n', 'm_n', 'nt', 'dt'
        ]

        print("Current parameters:")
        for param in parameters_to_show:
            print(f"{param}: {getattr(self, param, 'Not set')}")

    def get_action(self, x):
        mean = self.phi_1 * x
        sd = np.sqrt(self.v_n)
        action = self.rng.normal(mean, sd)  # Use the class's RNG
        return action

    def get_next_state(self, x, u):
        dW = np.sqrt(self.dt) * self.rng.normal()  # Use the class's RNG
        return x + (self.real_A * x + self.real_B * u) * self.dt + (self.real_C * x + self.real_D * u) * dW
    
    def calculations(self, x_list, u_list):
        x_list = np.array(x_list)
        u_list = np.array(u_list)

        self.phi_1_list.append(self.phi_1)

        # ---------------------
        # 1) Compute A_n, B_n, G_n, alpha_n for the current iteration
        # ---------------------
        diff = np.diff(x_list)
        alpha_n = np.sum(diff**2)  # sum of [x_k+1 - x_k]^2

        A_n = self.dt * np.sum(x_list[:-1]**2)
        B_n = self.dt * np.sum(x_list[:-1]*u_list)
        G_n = self.dt * np.sum(u_list**2)

        # ---------------------
        # 2) Update partial sums
        # ---------------------
        self.S_alpha2 += alpha_n**2
        self.S_alphaA += alpha_n * A_n
        self.S_alphaB += alpha_n * B_n
        self.S_alphaG += alpha_n * G_n

        self.S_A2     += A_n**2
        self.S_B2     += B_n**2
        self.S_G2     += G_n**2
        self.S_AB     += A_n * B_n
        self.S_AG     += A_n * G_n
        self.S_BG     += B_n * G_n

        # ---------------------
        # 3) Update mat, vec for A,B
        # ---------------------
        self.mat += np.array([[A_n, B_n],
                              [B_n, G_n]])
        self.vec += np.array([
            np.sum(x_list[:-1]*diff),
            np.sum(u_list*diff)
        ])

    def estimate_and_update(self):
        # Solve for A,B
        self.A, self.B = np.linalg.inv(self.mat) @ self.vec
        self.A_list.append(self.A)
        self.B_list.append(self.B)

        def objective(params):
            C, D = params
            Sa2  = self.S_alpha2
            SaA  = self.S_alphaA
            SaB  = self.S_alphaB
            SaG  = self.S_alphaG
            SA2  = self.S_A2
            SB2  = self.S_B2
            SG2  = self.S_G2
            SAB  = self.S_AB
            SAG  = self.S_AG
            SBG  = self.S_BG

            val = (
                  Sa2
                - 2*SaA*(C**2)
                - 4*SaB*(C*D)
                - 2*SaG*(D**2)
                + SA2*(C**4)
                + 4*SAB*(C**3*D)
                + (2*SAG + 4*SB2)*(C**2*D**2)
                + 4*SBG*(C*D**3)  
                + SG2*(D**4)
            )
            return val

        initial_guess = [self.C, self.D]
        result = minimize(objective,
                          x0=initial_guess,
                          method='L-BFGS-B')
        self.C, self.D = result.x
        self.C_list.append(self.C)
        self.D_list.append(self.D)
    
    def get_one_trajectory(self):
        x_list = []
        u_list = []
        x = self.x_0
        for k in range(self.nt):
            u = self.get_action(x)
            x_list.append(x)
            u_list.append(u)
            x = self.get_next_state(x, u)
        x_list.append(x)
        
        # calculate, estimate and update
        self.calculations(x_list, u_list)
        self.estimate_and_update()
        return 

    def get_many_trajectories(self):
        for _ in range(self.m_n):
            self.get_one_trajectory()
        return

    def clamp(self, value, a, b):
        return max(a, min(value, b))
    
    def ema_update(self, pre, new):
        w = 0
        return pre * w + new * (1 - w)

    def run_many_iterations(self):
        for self._n in tqdm(range(self.N)):
            self.update_hyper_parameter()
            self.get_many_trajectories()

        self.A_list = np.array(self.A_list)
        self.B_list = np.array(self.B_list)
        self.C_list = np.array(self.C_list)
        self.D_list = np.array(self.D_list)

    def delta(self, A, B, C, D):
        # Calculate Delta using the specific values of A, B, C, and D
        return (B**2 + 2*B*C*D - 2*A*D**2) / D**2

    def k_t(self, t, A, B, C, D):
        # Calculate k(t) using the given values and specific Delta
        Delta = self.delta(A, B, C, D)
        return self.Q / Delta + (self.H - self.Q / Delta) / np.exp(Delta * self.T) * np.exp(Delta * t)

    def value_function(self, A, B, C, D):
        Delta = self.delta(A, B, C, D)
        res = self.Q / Delta + (self.H - self.Q / Delta) / \
            np.exp(Delta * self.T)
        return -0.5 * res * self.x_0**2

    def get_optimal_value_function(self):
        return self.value_function(self.real_A, self.real_B, self.real_C, self.real_D)

    def j_hat(self, phi_1):
        A = self.real_A
        B = self.real_B
        C = self.real_C
        D = self.real_D
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

        return function_f(phi_1)

    def get_all_regrets(self):
        optimal_val = self.get_optimal_value_function()
        curr_vals = self.j_hat(np.array(self.phi_1_list))

        all_regrets = [optimal_val - val for val in curr_vals]
        return all_regrets