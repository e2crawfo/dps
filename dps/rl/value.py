import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from dps.rl import DiscretePolicy, AgentHead, RLObject
from dps.utils.tf import masked_mean, tf_roll, RNNCell


class ValueFunction(AgentHead):
    def __init__(self, size, policy, name):
        self.policy = policy
        self._size = size
        super(ValueFunction, self).__init__(name)

    @property
    def n_params(self):
        return self._size

    def generate_signal(self, key, context, **kwargs):
        if key == 'values':
            utils = self.agent.get_utils(self, context)
            values = tf.identity(utils, name="{}_{}_values".format(self.agent.name, self.name))

            mask = context.get_signal('mask')
            label = "{}-estimated_value".format(self.display_name)
            context.add_summary(tf.summary.scalar(label, masked_mean(values, mask)))

            return values

        elif key == 'one_step_td_errors':
            rewards = context.get_signal('rewards')
            gamma = context.get_signal('gamma')
            c = kwargs.get('c', None)
            rho = context.get_signal('rho', self.policy, c=c)
            values = context.get_signal('values', self, gradient=True)

            shifted_values = tf_roll(values, 1, fill=0.0, reverse=True, axis=0)

            one_step_estimate = rho * (rewards + gamma * shifted_values)
            td_errors = one_step_estimate - values

            # if context.truncated_rollouts is True, this is slightly problematic
            # as the td error at the last timestep is not correct as the one-step-estimate
            # includes only the final reward, no value function (since we don't know what state
            # comes next). So just zero-out the corresponding td error.
            if context.truncated_rollouts:
                td_errors = tf.concat([td_errors[:-1, ...], tf.zeros_like(td_errors[-1:, ...])], axis=0)

            mask = context.get_signal('mask')
            label = "{}-one_step_td_error".format(self.display_name)
            context.add_summary(tf.summary.scalar(label, masked_mean(td_errors, mask)))

            return td_errors

        elif key == 'monte_carlo_td_errors':
            if context.truncated_rollouts:
                raise Exceltion("NotImplemented")

            discounted_returns = context.get_signal('discounted_returns', self.policy)
            values = context.get_signal('values', self, gradient=True)

            return discounted_returns[:-1, ...] - values[:-1, ...]

        else:
            raise Exception()


class ActionValueFunction(AgentHead):
    def __init__(self, size, policy, name):
        self.policy = policy
        self._size = size
        super(ActionValueFunction, self).__init__(name)

    @property
    def n_params(self):
        return self._size

    def generate_signal(self, key, context):
        if key == 'action_values_all':
            utils = self.agent.get_utils(self, context)
            values = tf.identity(utils, name="{}_{}_values".format(self.agent.name, self.name))
            return values

        elif key == 'action_values':
            action_values = context.get_signal('action_values_all', self, gradient=True)
            actions = context.get_signal('actions')
            action_values = tf.reduce_sum(actions * action_values, axis=-1, keepdims=True)

            mask = context.get_signal('mask')
            label = "{}-estimated_action_value".format(self.display_name)
            context.add_summary(tf.summary.scalar(label, masked_mean(action_values, mask)))

            return action_values
        elif key == 'one_step_td_errors':
            rewards = context.get_signal('rewards')
            gamma = context.get_signal('gamma')
            action_values = context.get_signal('action_values', self, gradient=True)

            shifted_values = tf_roll(action_values, 1, fill=0.0, reverse=True, axis=0)

            one_step_estimate = rewards + gamma * shifted_values
            td_errors = one_step_estimate - action_values

            # if context.truncated_rollouts is True, this is slightly problematic
            # as the td error at the last timestep is not correct as the one-step-estimate
            # includes only the final reward, no value function (since we don't know what state
            # comes next). So just zero out the corresponding td error.
            if context.truncated_rollouts:
                td_errors = tf.concat([td_errors[:-1, ...], tf.zeros_like(td_errors[-1:, ...])], axis=0)

            mask = context.get_signal('mask')
            label = "{}-one_step_td_error".format(self.display_name)
            context.add_summary(tf.summary.scalar(label, masked_mean(td_errors, mask)))

            return td_errors
        else:
            raise Exception()


class AverageValueEstimator(RLObject):
    """ Value estimation without a value function. """

    def __init__(self, policy, importance_c=None):
        self.policy = policy
        self.importance_c = importance_c

    def generate_signal(self, signal_key, context):
        if signal_key == "values":
            if self.policy is not None:
                return context.get_signal('average_monte_carlo_values', self.policy, importance_c=self.importance_c)
            else:
                return context.get_signal('average_discounted_returns')
        else:
            raise Exception("NotImplemented")


class MonteCarloValueEstimator(RLObject):
    """ Off-policy monte-carlo value estimation. """

    def __init__(self, policy=None, importance_c=None):
        self.policy = policy
        self.importance_c = importance_c

    def generate_signal(self, signal_key, context):
        if signal_key == "values":
            if self.policy is not None:
                return context.get_signal('monte_carlo_values', self.policy, c=self.importance_c)
            else:
                return context.get_signal('discounted_returns')
        elif signal_key == "action_values":
            if self.policy is not None:
                return context.get_signal('monte_carlo_action_values', self.policy, c=self.importance_c)
            else:
                return context.get_signal('discounted_returns')
        else:
            raise Exception("NotImplemented")


class AdvantageEstimator(RLObject):
    """ Create an advantage estimator from a state value estimator paired with a state-action value estimator. """

    def __init__(self, q_estimator, v_estimator, standardize=True):
        self.q_estimator = q_estimator
        self.v_estimator = v_estimator
        self.standardize = standardize

    def post_process(self, advantage, context):
        if self.standardize:
            mask = context.get_signal('mask')
            mean = masked_mean(advantage, mask)
            _advantage = advantage - mean
            variance = masked_mean(_advantage**2, mask)
            std = tf.sqrt(variance)
            _advantage = tf.cond(std <= 0, lambda: _advantage, lambda: _advantage/std)
            advantage = mask * _advantage + (1-mask) * advantage
        return advantage

    def generate_signal(self, signal_key, context):
        if signal_key == "advantage":
            q = context.get_signal('action_values', self.q_estimator)
            v = context.get_signal('values', self.v_estimator)
            advantage = q - v
            advantage = self.post_process(advantage, context)

            mask = context.get_signal("mask")

            mean_advantage = masked_mean(advantage, mask)
            context.add_summary(tf.summary.scalar("advantage", mean_advantage))

            mean_abs_advantage = masked_mean(tf.abs(advantage), mask)
            context.add_summary(tf.summary.scalar("abs_advantage", mean_abs_advantage))

            return advantage
        elif signal_key == "advantage_all":
            q = context.get_signal('action_values_all', self.q_estimator)
            v = context.get_signal('values', self.v_estimator)
            v = v[..., None]
            advantage = q - v
            advantage = self.post_process(advantage, context)
            return advantage
        else:
            raise Exception("NotImplemented")


class BasicAdvantageEstimator(AdvantageEstimator):
    """ Advantage estimation without a value function. """

    def __init__(self, policy=None, standardize=True, q_importance_c=None, v_importance_c=None):
        q_estimator = MonteCarloValueEstimator(policy, importance_c=q_importance_c)
        v_estimator = AverageValueEstimator(policy, importance_c=v_importance_c)
        super(BasicAdvantageEstimator, self).__init__(q_estimator, v_estimator, standardize)


# class NStepValueEstimator(RLObject):
#     def __init__(self, value_function, n=1, standardize=True):
#         self.value_function = value_function
#         self.n = n
#         assert n > 0
#         super(NStepValueEstimator, self).__init__(standardize)
#
#     def generate_signal(self, signal_key, context):
#         if signal_key == "values":
#             values = context.get_signal("values", self.value_function)
#
#             rewards = context.get_signal("rewards")
#             T = tf.shape(rewards)[0]
#             discount_matrix = tf_discount_matrix(gamma * self.lmbda, T, self.n)
#             n_step_rewards = tf.tensordot(discount_matrix, rewards, axes=1)
#
#             shifted_values = tf_roll(values, self.n, fill=0.0, reverse=True)
#             values = n_step_rewards + (gamma**self.n) * shifted_values
#             return values
#         else:
#             raise Exception("NotImplemented")


class RetraceCell(RNNCell):
    def __init__(self, dim, gamma, lmbda, to_action_value):
        self.dim = dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.to_action_value = to_action_value

    def __call__(self, inp, state, scope=None):
        rho, r, v = inp
        prev_retrace = state

        if self.to_action_value:
            one_step_estimate = r + self.gamma * v
            adjustment = self.gamma * self.lmbda * (rho * prev_retrace - v)
        else:
            one_step_estimate = rho * (r + self.gamma * v)
            adjustment = rho * self.gamma * self.lmbda * (prev_retrace - v)

        new_retrace = one_step_estimate + adjustment

        return (new_retrace, one_step_estimate, adjustment), new_retrace

    @property
    def state_size(self):
        return self.dim

    @property
    def output_size(self):
        return (self.dim,) * 3

    def zero_state(self, batch_size, dtype):
        return tf.fill((batch_size, 1), 0.0)


class Retrace(RLObject):
    """ An off-policy (though also works on-policy) returned-based estimate of the value of
        a policy as a function of either a state or a state-action pair. The estimate
        is based on an existing value function or action-value function.

        Parameters
        ----------

    """
    def __init__(
            self, policy, value_function, lmbda=1.0, importance_c=0,
            to_action_value=False, from_action_value=False,
            name=None):

        self.policy = policy
        self.value_function = value_function
        self.lmbda = lmbda
        self.importance_c = importance_c
        self.to_action_value = to_action_value
        self.from_action_value = from_action_value
        self.name = name or self.__class__.__name__

    def generate_signal(self, signal_key, context):
        if signal_key == "action_values" and self.to_action_value:
            pass
        elif signal_key == "values" and not self.to_action_value:
            pass
        else:
            raise Exception("NotImplemented")

        rewards = context.get_signal("rewards")
        rho = context.get_signal("rho", self.policy, c=self.importance_c)

        if self.from_action_value:
            if isinstance(self.policy, DiscretePolicy):
                pi_log_probs_all = context.get_signal("log_probs_all", self.policy)
                pi_probs = tf.exp(pi_log_probs_all)
                action_values = context.get_signal("action_values", self.value_function)
                values = tf.reduce_sum(pi_probs * action_values, axis=-1, keepdims=True)
            else:
                action_values = context.get_signal("action_values", self.value_function)
                values = action_values * rho
        else:
            values = context.get_signal("values", self.value_function)

        R = rewards
        V = tf_roll(values, 1, fill=0.0, reverse=True)
        RHO = rho

        if context.truncated_rollouts:
            R = R[:-1, ...]
            V = V[:-1, ...]
            RHO = RHO[:-1, ...]

        if self.to_action_value:
            RHO = tf_roll(RHO, 1, fill=1.0, reverse=True)

        gamma = context.get_signal("gamma")
        retrace_cell = RetraceCell(
            rewards.shape[-1], gamma, self.lmbda, self.to_action_value)

        retrace_input = (
            tf.reverse(RHO, axis=[0]),
            tf.reverse(R, axis=[0]),
            tf.reverse(V, axis=[0]),
        )

        (retrace, one_step_estimate, adjustment), _ = dynamic_rnn(
            retrace_cell, retrace_input,
            initial_state=V[-1, ...],
            parallel_iterations=1, swap_memory=False, time_major=True)

        one_step_estimate = tf.reverse(one_step_estimate, axis=[0])
        adjustment = tf.reverse(adjustment, axis=[0])
        retrace = tf.reverse(retrace, axis=[0])

        if context.truncated_rollouts:
            retrace = tf.concat([retrace, V[-1, ...]], axis=0)

        mask = context.get_signal("mask")
        label = "{}-one_step_estimate".format(self.name)
        context.add_summary(tf.summary.scalar(label, masked_mean(one_step_estimate, mask)))
        label = "{}-adjustment".format(self.name)
        context.add_summary(tf.summary.scalar(label, masked_mean(adjustment, mask)))
        label = "{}-retrace".format(self.name)
        context.add_summary(tf.summary.scalar(label, masked_mean(retrace, mask)))

        return retrace


class GeneralizedAdvantageEstimator(AdvantageEstimator):
    """
    Parameters
    ----------

    """
    def __init__(
            self, policy, value_function, action_value_function=None,
            lmbda=1.0, standardize=True):

        self.policy = policy
        self.value_function = value_function
        self.action_value_function = action_value_function
        self.lmbda = lmbda

        if action_value_function is not None:
            q_estimator = Retrace(
                policy, lmbda, value_function,
                from_action_value=False, to_action_value=True
            )
        else:
            q_estimator = Retrace(
                policy, lmbda, action_value_function,
                from_action_value=True, to_action_value=True
            )

        super(GeneralizedAdvantageEstimator, self).__init__(
            q_estimator=q_estimator, v_estimator=self.value_function,
            standardize=standardize)
