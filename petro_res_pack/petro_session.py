# from gym_like_env import Env
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd


def id_tr(x):
    return x


class Session:
    def __init__(self, env, n_iter=None, plot_freq=50, dpi=300):
        self.dpi = dpi
        self.env = env
        self.n_iter = n_iter
        self.plot_freq = plot_freq
        self.times = []
        self.p_well_hist = {}
        self.s_o_well_hist = {}
        self.q_o_hist = {}
        self.q_w_hist = {}
        self.openness = {}

        self.p_well_hist_loc = {}
        self.s_o_well_hist_loc = {}
        self.q_o_hist_loc = {}
        self.q_w_hist_loc = {}
        self.openness_loc = {}
        self.wells_benefit = {}
        self.wells_benefit_loc = {}
        self.i = 0
        for w in self.env.pos_r:
            self.p_well_hist[w] = []
            self.s_o_well_hist[w] = []
            self.q_o_hist[w] = []
            self.q_w_hist[w] = []
            self.openness[w] = []
            self.wells_benefit[w] = []

            self.p_well_hist_loc[w] = []
            self.s_o_well_hist_loc[w] = []
            self.q_o_hist_loc[w] = []
            self.q_w_hist_loc[w] = []
            self.openness_loc[w] = []
            self.wells_benefit_loc[w] = []

    def done(self):
        out = False
        if self.n_iter is not None:
            if self.i < self.n_iter:
                out = False
            else:
                out = True
        return out

    def run(self, policy=None, openity_states=5, obs_trans_func=id_tr, stoch=False, save=False, path=None):

        self.__init__(env=self.env, n_iter=self.n_iter, plot_freq=self.plot_freq)
        n_wells = len(self.env.pos_r)
        env_done = False
        state = self.env.reset()
        state = obs_trans_func(state)

        label_font_size = 16
        title_font_size = 16
        x_tick_size = 14

        nx = self.env.prop.nx
        ny = self.env.prop.ny

        xs = np.linspace(0, self.env.prop.dx * (nx - 1), nx)
        ys = np.linspace(0, self.env.prop.dy * (ny - 1), ny)

        while not env_done:
            state = torch.DoubleTensor(state).unsqueeze(0)

            with torch.no_grad():
                if policy is not None:
                    action = policy.sample_actions(state, stoch=stoch)
                    if action.shape[0] == 1:
                        action = action[0]
                    action = policy.act_to_openness(action)
                    if type(action) == torch.tensor:
                        action = action.numpy()
                else:
                    action = np.ones(len(self.env.pos_r))

            state, reward, env_done, _ = self.env.step(action)
            state = obs_trans_func(state)
            self.i += 1
            self.i += 1
            q_o = self.env.get_q(ph='o')
            q_w = self.env.get_q(ph='w')
            # save local data to average them later
            for _i, w in enumerate(self.env.pos_r):
                self.p_well_hist_loc[w].append(self.env.p[w] / 6894.)
                self.s_o_well_hist_loc[w].append(self.env.s_o[w])
                self.q_o_hist_loc[w].append(q_o[w] * 3600)
                self.q_w_hist_loc[w].append(q_w[w] * 3600)
                self.openness_loc[w].append(action[_i])
                __benefit = self.q_o_hist_loc[w][-1] * self.env.price['o'] * 6.28981
                __benefit -= self.q_w_hist_loc[w][-1] * self.env.price['w'] * 6.28981
                self.wells_benefit_loc[w].append(__benefit)

            # let's plot averaged data and set local hist to []
            if self.i % self.plot_freq == 0:
                self.times.append(self.env.t)
                #
                for _i, w in enumerate(self.env.pos_r):
                    self.p_well_hist[w].append(np.array(self.p_well_hist_loc[w]).mean())
                    self.s_o_well_hist[w].append(np.array(self.s_o_well_hist_loc[w]).mean())
                    self.q_o_hist[w].append(np.array(self.q_o_hist_loc[w]).mean())
                    self.q_w_hist[w].append(np.array(self.q_w_hist_loc[w]).mean())
                    self.openness[w].append(np.array(self.openness_loc[w]).mean())
                    self.wells_benefit[w].append(np.array(self.wells_benefit_loc[w]).mean())

                    self.openness_loc[w] = []
                    self.openness_loc[w] = []
                    self.s_o_well_hist_loc[w] = []
                    self.q_o_hist_loc[w] = []
                    self.q_w_hist_loc[w] = []
                    self.openness_loc[w] = []
                    self.wells_benefit_loc[w] = []

                display.clear_output(wait=True)
                f, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 12))
                f.tight_layout(pad=6.0)

                df = pd.DataFrame(self.env.p.v.reshape((nx, ny)), columns=xs, index=ys)
                sns.heatmap(df / 6894., ax=ax[0][0], cbar=True)
                ax[0][0].set_title(f'Pressure, psi\nt={self.env.t: .1f} days', fontsize=title_font_size)
                ax[0][0].set_xlabel('y, m', fontsize=label_font_size)
                ax[0][0].set_ylabel('x, m', fontsize=label_font_size)
                ax[0][0].tick_params(axis='x', labelsize=x_tick_size)
                ax[0][0].tick_params(axis='y', labelsize=x_tick_size)

                df = pd.DataFrame(self.env.s_o.v.reshape((nx, ny)), columns=xs, index=ys)
                sns.heatmap(df, ax=ax[0][1],
                            cbar=True, fmt=".2f")
                ax[0][1].set_title(f'Saturation, oil\nt={self.env.t: .1f} days', fontsize=title_font_size)
                ax[0][1].set_xlabel('y, m', fontsize=label_font_size)
                ax[0][1].set_ylabel('x, m', fontsize=label_font_size)
                ax[0][1].tick_params(axis='x', labelsize=x_tick_size)
                ax[0][1].tick_params(axis='y', labelsize=x_tick_size)

                for w in self.env.pos_r:
                    ax[1][0].plot(self.times, self.p_well_hist[w], label=f'{w}')
                    ax[1][1].plot(self.times, self.s_o_well_hist[w], label=f'{w}')
                    ax[2][0].plot(self.times, self.q_o_hist[w], label=f'{w}')
                    ax[2][1].plot(self.times, self.q_w_hist[w], label=f'{w}')
                    ax[3][0].plot(self.times, self.openness[w], label=f'{w}')
                    ax[3][1].plot(self.times,  self.wells_benefit[w], label=f'{w}')
                ax[1][0].set_xlabel('time, days', fontsize=label_font_size)
                ax[1][0].set_ylabel('pressure, psi', fontsize=label_font_size)
                ax[1][0].set_title('Pressure in wells', fontsize=title_font_size)
                ax[1][0].legend()
                ax[1][0].tick_params(axis='x', labelsize=x_tick_size)
                ax[1][0].tick_params(axis='y', labelsize=x_tick_size)

                ax[3][0].set_xlabel('time, days', fontsize=label_font_size)
                ax[3][0].set_ylabel('degree of open', fontsize=label_font_size)
                ax[3][0].set_title('Choke openness', fontsize=title_font_size)
                ax[3][0].legend()
                ax[3][0].tick_params(axis='x', labelsize=x_tick_size)
                ax[3][0].tick_params(axis='y', labelsize=x_tick_size)

                ax[3][1].set_xlabel('time, days', fontsize=label_font_size)
                ax[3][1].set_ylabel('USD / h', fontsize=label_font_size)
                ax[3][1].set_title('Well benefit', fontsize=title_font_size)
                ax[3][1].legend()
                ax[3][1].tick_params(axis='x', labelsize=x_tick_size)
                ax[3][1].tick_params(axis='y', labelsize=x_tick_size)

                ax[1][1].set_xlabel('time, days', fontsize=label_font_size)
                ax[1][1].set_ylabel('fraction', fontsize=label_font_size)
                ax[1][1].set_title('Oil fraction in wells', fontsize=title_font_size)
                ax[1][1].legend()
                ax[1][1].tick_params(axis='x', labelsize=x_tick_size)
                ax[1][1].tick_params(axis='y', labelsize=x_tick_size)

                ax[2][0].set_xlabel('time, days', fontsize=label_font_size)
                ax[2][0].set_ylabel('q, m3/h', fontsize=label_font_size)
                ax[2][0].set_title('Oil rate', fontsize=title_font_size)
                ax[2][0].legend()
                ax[2][0].tick_params(axis='x', labelsize=x_tick_size)
                ax[2][0].tick_params(axis='y', labelsize=x_tick_size)

                ax[2][1].set_xlabel('time, days', fontsize=label_font_size)
                ax[2][1].set_ylabel('q, m3/h', fontsize=label_font_size)
                ax[2][1].set_title('Water rate', fontsize=title_font_size)
                ax[2][1].legend()
                ax[2][1].tick_params(axis='x', labelsize=x_tick_size)
                ax[2][1].tick_params(axis='y', labelsize=x_tick_size)

                ax[1][0].set_xscale('log')
                ax[1][1].set_xscale('log')
                ax[2][0].set_xscale('log')
                ax[2][1].set_xscale('log')
                ax[3][0].set_xscale('log')
                ax[3][1].set_xscale('log')

                plt.tight_layout()
                if save:
                    plt.savefig(f'{path}/{self.i:06}.png', dpi=self.dpi)
                plt.show()

    def reset(self):
        self.times = []
        self.p_well_hist = {}
        self.s_o_well_hist = {}
        self.q_o_hist = {}
        self.q_w_hist = {}
        self.i = 0
        self.times = []
        for w in self.env.pos_r:
            self.p_well_hist[w] = []
            self.s_o_well_hist[w] = []
            self.q_o_hist[w] = []
            self.q_w_hist[w] = []
        self.env.reset()
