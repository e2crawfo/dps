import numpy as np


class ValueFunction(object):
    def __init__(self):
        pass

    def fit_and_eval(self, obs, actions, rewards):
        """ Make parameters take new batch of trajectories into account,
            and then provide value estimates for those trajectories. """
        raise Exception("Not implemented.")


class SimpleValueFunction(object):
    """ Form an average based on the timestep, but don't take state into account. """

    def fit_and_eval(self, obs, actions, rewards):
        sum_rewards = np.flipud(np.cumsum(np.flipud(rewards), axis=0))
        return sum_rewards.mean(1, keepdims=True)


class SimpleMemoryValueFunction(object):
    """ Form an average based on the timestep, mixing with results from previous
        batches of data. alpha \in (0, 1) determines mixing rate. """
    def __init__(self, alpha, initial=None):
        self.alpha = alpha
        self.values = initial

    def fit_and_eval(self, obs, actions, rewards):
        sum_rewards = np.flipud(np.cumsum(np.flipud(rewards), axis=0))
        new_values = sum_rewards.mean(1, keepdims=True)

        if self.values is not None:
            self.values = self.alpha * new_values + (1-self.alpha) * self.values
        else:
            self.values = new_values
        return self.values


class FuncApproxValueFunction(object):
    def __init__(self):
        pass

    def fit_and_eval(self, obs, actions, rewards):
        pass
