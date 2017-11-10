import tensorflow as tf

from dps.rl import ObjectiveFunctionTerm
from dps.utils.tf import masked_mean


class PolicyGradient(ObjectiveFunctionTerm):
    def __init__(self, policy, advantage_estimator, epsilon=None, importance_c=0, **kwargs):
        self.policy = policy
        self.advantage_estimator = advantage_estimator
        self.epsilon = epsilon
        self.importance_c = importance_c
        super(PolicyGradient, self).__init__(**kwargs)

    def generate_signal(self, signal_key, context, **kwargs):
        if signal_key == "prev_log_probs":
            self.log_probs = context.get_signal('log_probs', self.policy)
            self.prev_log_probs = tf.placeholder(tf.float32, shape=self.log_probs.shape, name="_prev_log_probs")
            return self.prev_log_probs
        elif signal_key == "prev_advantage":
            self.advantage = context.get_signal('advantage', self.advantage_estimator)
            self.prev_advantage = tf.placeholder(tf.float32, shape=self.advantage.shape, name="_prev_advantage")
            return self.prev_advantage
        elif signal_key == 'importance_weights':
            pi_log_probs = context.get_signal("prev_log_probs", self)
            mu_log_probs = context.get_signal("mu_log_probs")
            importance_weights = tf.exp(pi_log_probs - mu_log_probs)

            label = "{}-mean_importance_weight".format(self.name)
            mask = context.get_signal("mask")
            context.add_summary(tf.summary.scalar(label, masked_mean(importance_weights, mask)))

            return importance_weights
        elif signal_key == "rho":
            c = kwargs.get('c', None)
            importance_weights = context.get_signal("importance_weights", self)

            if c is not None:
                if c <= 0:
                    rho = importance_weights
                else:
                    rho = tf.minimum(importance_weights, c)
            else:
                rho = tf.ones_like(importance_weights)

            label = "{}-mean_rho_c={}".format(self.name, c)
            mask = context.get_signal("mask")
            context.add_summary(tf.summary.scalar(label, masked_mean(rho, mask)))

            return rho

        elif signal_key == "adv_times_ratio":
            log_probs = context.get_signal('log_probs', self.policy, gradient=True)
            prev_log_probs = context.get_signal('prev_log_probs', self)

            ratio = tf.exp(log_probs - prev_log_probs)

            prev_advantage = context.get_signal('prev_advantage', self)

            if self.epsilon is None:
                adv_times_ratio = ratio * prev_advantage
            else:
                adv_times_ratio = tf.minimum(
                    prev_advantage * ratio,
                    prev_advantage * tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon))

            if self.use_weights:
                weights = context.get_signal('weights')
                adv_times_ratio *= weights

            rho = context.get_signal('rho', self, c=self.importance_c)
            adv_times_ratio *= rho

            return adv_times_ratio
        else:
            raise Exception("NotImplemented")

    def build_graph(self, context):
        adv_times_ratio = context.get_signal("adv_times_ratio", self, gradient=True)
        mask = context.get_signal("mask")
        objective = masked_mean(adv_times_ratio, mask)

        label = "{}-policy_gradient_objective".format(self.policy.display_name)
        context.add_summary(tf.summary.scalar(label, objective))

        return objective

    def pre_update(self, feed_dict, context):
        sess = tf.get_default_session()
        prev_log_probs, prev_advantage = sess.run([self.log_probs, self.advantage], feed_dict=feed_dict)
        feed_dict.update({
            self.prev_log_probs: prev_log_probs,
            self.prev_advantage: prev_advantage,
        })

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
            td_error *= weights
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


class ValueFunctionRegularization(ObjectiveFunctionTerm):
    """ Penalize the KL divergence between the new value function and
        the value function as it was at the start of the current update step. """

    def __init__(self, policy_evaluation, **kwargs):
        self.policy_evaluation = policy_evaluation
        self.value_function = self.policy_evaluation.value_function
        super(ValueFunctionRegularization, self).__init__(**kwargs)

    def generate_signal(self, signal_key, context):
        if signal_key == "variance":
            self.variance = tf.placeholder(tf.float32, ())

            squared_error = context.get_signal('squared_td_error', self.policy_evaluation)
            self.variance_computation = tf.reduce_mean(squared_error)

            return self.variance
        elif signal_key == "prev_values":
            self.values = context.get_signal('values', self.value_function)
            self.prev_values = tf.placeholder(tf.float32, shape=self.values.shape, name="_prev_values")
            return self.prev_values
        else:
            raise Exception("NotImplemented")

    def build_graph(self, context):
        prev_values = context.get_signal('prev_values', self)
        values = context.get_signal('values', self.value_function, gradient=True)
        variance = context.get_signal('variance', self)

        mean_kl_divergence = tf.reduce_mean((prev_values - values)**2 / (2 * variance))

        label = "{}-mean_kl_divergence".format(self.value_function.display_name)
        context.add_summary(tf.summary.scalar(label, mean_kl_divergence))

        return -mean_kl_divergence

    def pre_update(self, feed_dict, context):
        sess = tf.get_default_session()
        variance, prev_values = sess.run([self.variance_computation, self.values], feed_dict=feed_dict)
        feed_dict.update({
            self.prev_values: prev_values,
            self.variance: variance,
        })

    def pre_eval(self, feed_dict, context):
        self.pre_update(feed_dict, context)


class ConstrainedPolicyEvaluation_State(ObjectiveFunctionTerm):
    def __init__(self, value_function, target_generator, epsilon, clipped_error=False, n_samples=0, direct=False, **kwargs):
        self.value_function = value_function
        self.target_generator = target_generator
        self.epsilon = epsilon
        self.clipped_error = clipped_error
        self.n_samples = n_samples
        self.direct = direct
        super(ConstrainedPolicyEvaluation_State, self).__init__(**kwargs)

    def generate_signal(self, signal_key, context):
        if signal_key == "td_error":
            estimates = context.get_signal('values', self.value_function, gradient=True)
            targets = context.get_signal('values', self.target_generator)
            return estimates - targets
        elif signal_key == "squared_td_error":
            td_error = context.get_signal('td_error', self, gradient=True)
            squared_td_error = clipped_error(td_error) if self.clipped_error else td_error**2
            return squared_td_error
        elif signal_key == "variance":
            self.variance = tf.placeholder(tf.float32, ())

            squared_error = context.get_signal('squared_td_error', self)
            self.variance_computation = tf.reduce_mean(squared_error)

            return self.variance
        elif signal_key == "prev_values":
            self.values = context.get_signal('values', self.value_function)
            self.prev_values = tf.placeholder(tf.float32, shape=self.values.shape, name="_prev_values")
            return self.prev_values
        else:
            raise Exception("NotImplemented")

    def build_graph(self, context):
        prev_values = context.get_signal('prev_values', self)
        values = context.get_signal('values', self.value_function, gradient=True)
        variance = context.get_signal('variance', self)
        targets = context.get_signal('values', self.target_generator)

        if self.direct:
            std = tf.sqrt(variance)
            constrained_values = tf.clip_by_value(
                values, prev_values - std * self.epsilon, prev_values + std * self.epsilon)
            objective = -(constrained_values - targets)**2
            divergence = tf.abs(constrained_values - prev_values)

            if self.use_weights:
                weights = context.get_signal('weights')
                objective *= weights

            mask = context.get_signal("mask")

            mean_divergence = masked_mean(tf.reduce_mean(divergence, axis=-1, keep_dims=True), mask)
            label = "{}-opt-mean_ve_divergence".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, mean_divergence))

            objective = masked_mean(tf.reduce_mean(objective, axis=-1, keep_dims=True), mask)
            label = "{}-opt-ve_direct_objective".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, objective))

            td_error = context.get_signal("td_error", self)
            squared_td_error = context.get_signal("squared_td_error", self)

            mean_td_error = masked_mean(td_error, mask)
            label = "{}-opt-mean_abs_td_error".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, tf.abs(mean_td_error)))

            mean_squared_td_error = masked_mean(squared_td_error, mask)
            label = "{}-opt-mean_squared_td_error".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, mean_squared_td_error))

            return objective

        else:
            clipped_ratio = None
            if self.n_samples == 0:
                ratio = tf.exp(0.5 * (2 * targets - values - prev_values) * (values - prev_values) / variance)

                # prev_advantage = (values - targets) ** 2 + variance
                prev_advantage = (prev_values - targets) ** 2 + variance

                if self.epsilon is None:
                    adv_times_ratio = ratio * prev_advantage
                else:
                    clipped_ratio = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)
                    adv_times_ratio = tf.minimum(
                        prev_advantage * ratio,
                        prev_advantage * clipped_ratio
                    )
            else:
                T = tf.shape(values)[0]
                batch_size = tf.shape(values)[1]
                samples = tf.random_normal((T, batch_size, self.n_samples)) * tf.sqrt(variance) + prev_values
                ratio = tf.exp(0.5 * (2 * samples - values - prev_values) * (values - prev_values) / variance)
                prev_advantage = (prev_values - targets) ** 2 + variance - (samples - targets)**2

                if self.epsilon is None:
                    adv_times_ratio = ratio * prev_advantage
                else:
                    clipped_ratio = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)
                    adv_times_ratio = tf.minimum(
                        prev_advantage * ratio,
                        prev_advantage * clipped_ratio
                    )

            if self.use_weights:
                weights = context.get_signal('weights')
                adv_times_ratio *= weights

            mask = context.get_signal("mask")

            mean_ratio = masked_mean(tf.reduce_mean(ratio, axis=-1, keep_dims=True), mask)
            label = "{}-opt-mean_ve_ratio".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, mean_ratio))

            if clipped_ratio is not None:
                mean_clipped_ratio = masked_mean(tf.reduce_mean(clipped_ratio, axis=-1, keep_dims=True), mask)
                label = "{}-opt-mean_ve_clipped_ratio".format(self.value_function.display_name)
                context.add_summary(tf.summary.scalar(label, mean_clipped_ratio))

            mean_advantage = masked_mean(tf.reduce_mean(prev_advantage, axis=-1, keep_dims=True), mask)
            label = "{}-opt-mean_ve_advantage".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, mean_advantage))

            objective = masked_mean(tf.reduce_mean(adv_times_ratio, axis=-1, keep_dims=True), mask)
            label = "{}-opt-ve_objective".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, objective))

            td_error = context.get_signal("td_error", self)
            squared_td_error = context.get_signal("squared_td_error", self)

            mean_td_error = masked_mean(td_error, mask)
            label = "{}-opt-mean_abs_td_error".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, tf.abs(mean_td_error)))

            mean_squared_td_error = masked_mean(squared_td_error, mask)
            label = "{}-opt-mean_squared_td_error".format(self.value_function.display_name)
            context.add_summary(tf.summary.scalar(label, mean_squared_td_error))

            return objective

    def pre_update(self, feed_dict, context):
        sess = tf.get_default_session()
        variance, prev_values = sess.run([self.variance_computation, self.values], feed_dict=feed_dict)
        feed_dict.update({
            self.prev_values: prev_values,
            self.variance: variance,
        })

    def pre_eval(self, feed_dict, context):
        self.pre_update(feed_dict, context)
