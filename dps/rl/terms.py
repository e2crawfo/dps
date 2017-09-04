import tensorflow as tf

from dps.rl import ObjectiveFunctionTerm
from dps.utils import masked_mean


class PolicyGradient(ObjectiveFunctionTerm):
    def __init__(self, policy, advantage_estimator, epsilon=None, **kwargs):
        self.policy = policy
        self.advantage_estimator = advantage_estimator
        self.epsilon = epsilon
        super(PolicyGradient, self).__init__(**kwargs)

    def generate_signal(self, signal_key, context):
        if signal_key == "prev_log_probs":
            log_probs = context.get_signal('log_probs', self.policy)
            prev_log_probs = tf.placeholder(tf.float32, shape=log_probs.shape, name="_prev_log_probs")
            return prev_log_probs
        elif signal_key == "adv_times_ratio":
            advantage = context.get_signal('advantage', self.advantage_estimator)
            self.log_probs = context.get_signal('log_probs', self.policy, gradient=True)
            self.prev_log_probs = context.get_signal('prev_log_probs', self)

            ratio = tf.exp(self.log_probs - self.prev_log_probs)

            if self.epsilon is None:
                adv_times_ratio = ratio * advantage
            else:
                adv_times_ratio = tf.minimum(
                    advantage * ratio,
                    advantage * tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon))

            if self.use_weights:
                weights = context.get_signal('weights')
                adv_times_ratio *= weights
            return adv_times_ratio
        else:
            raise Exception("NotImplemented")

    def build_graph(self, context):
        adv_times_ratio = context.get_signal("adv_times_ratio", self, gradient=True)
        objective = tf.reduce_mean(tf.reduce_sum(adv_times_ratio, axis=0))

        label = "{}-policy_gradient_objective".format(self.policy.display_name)
        context.add_summary(tf.summary.scalar(label, objective))

        return objective

    def pre_update(self, feed_dict, context):
        sess = tf.get_default_session()
        prev_log_probs = sess.run(self.log_probs, feed_dict=feed_dict)
        feed_dict[self.prev_log_probs] = prev_log_probs

    def pre_eval(self, feed_dict, context):
        self.pre_update(feed_dict, context)


class PolicyEntropyBonus(ObjectiveFunctionTerm):
    def __init__(self, policy, **kwargs):
        self.policy = policy
        super(PolicyEntropyBonus, self).__init__(**kwargs)

    def generate_signal(self, signal_key, context):
        if signal_key == "entropy":
            entropy = context.get_signal('entropy', self.policy, gradient=True)

            if self.use_weights:
                weights = context.get_signal('weights')
                entropy *= weights

            return entropy
        else:
            raise Exception("NotImplemented")

    def build_graph(self, context):
        entropy = context.get_signal('entropy', self, gradient=True)
        mask = context.get_signal('mask')
        objective = masked_mean(entropy, mask)

        label = "{}-entropy".format(self.policy.display_name)
        context.add_summary(tf.summary.scalar(label, objective))

        return objective


def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class PolicyEvaluation_State(ObjectiveFunctionTerm):
    def __init__(self, value_function, target_generator, clipped_error=False, **kwargs):
        self.value_function = value_function
        self.target_generator = target_generator
        self.clipped_error = clipped_error
        super(PolicyEvaluation_State, self).__init__(**kwargs)

    def generate_signal(self, signal_key, context):
        if signal_key == "td_error":
            estimates = context.get_signal('values', self.value_function, gradient=True)
            targets = context.get_signal('values', self.target_generator)
            return estimates - targets
        elif signal_key == "squared_td_error":
            td_error = context.get_signal('td_error', self, gradient=True)
            squared_td_error = clipped_error(td_error) if self.clipped_error else td_error**2
            return squared_td_error
        else:
            raise Exception("NotImplemented")

    def build_graph(self, context):
        td_error = context.get_signal("td_error", self)
        squared_td_error = context.get_signal("squared_td_error", self, gradient=True)

        mask = context.get_signal('mask')
        weights = context.get_signal('weights')

        if context.truncated_rollouts:
            # When using truncated rollouts, there is no way to get a data-based
            # estimate of the value of the final state, so there is no
            # point in training on it.
            td_error = td_error[:-1, ...]
            squared_td_error = squared_td_error[:-1, ...]
            mask = mask[:-1, ...]
            weights = weights[:-1, ...]

        if self.use_weights:
            squared_td_error *= weights

        mean_td_error = masked_mean(td_error, mask)

        label = "{}-opt-mean_abs_td_error".format(self.value_function.display_name)
        context.add_summary(tf.summary.scalar(label, tf.abs(mean_td_error)))

        mean_squared_td_error = masked_mean(squared_td_error, mask)

        label = "{}-opt-mean_squared_td_error".format(self.value_function.display_name)
        context.add_summary(tf.summary.scalar(label, mean_squared_td_error))

        return -mean_squared_td_error


class PolicyEvaluation_StateAction(PolicyEvaluation_State):
    def generate_signal(self, signal_key, context):
        if signal_key == "td_error":
            estimates = context.get_signal('action_values', self.value_function, gradient=True)
            targets = context.get_signal('action_values', self.target_generator)
            return estimates - targets
        elif signal_key == "squared_td_error":
            td_error = context.get_signal('td_error', self, gradient=True)
            squared_td_error = clipped_error(td_error) if self.clipped_error else td_error**2
            return squared_td_error
        else:
            raise Exception("NotImplemented")
