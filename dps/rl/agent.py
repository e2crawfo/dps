import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from copy import deepcopy
from collections import OrderedDict

from dps import cfg
from dps.rl import RLObject
from dps.utils.tf import vec_to_lst, lst_to_vec, trainable_variables, ScopedCell


class AgentHead(RLObject):
    def __init__(self, name):
        self.agent = None
        super(AgentHead, self).__init__(name)

    def __str__(self):
        return "<{} - {}>".format(self.__class__.__name__, self.display_name)

    def __repr__(self):
        return str(self)

    @property
    def display_name(self):
        if self.agent is None:
            return self.name
        else:
            return "{}.{}".format(self.agent.name, self.name)

    def set_agent(self, agent):
        self.agent = agent

    def trainable_variables(self, for_opt):
        return self.agent.trainable_variables(for_opt=for_opt)

    @property
    def n_params(self):
        raise Exception("NotImplemented")


class Agent(RLObject):
    """

    Parameters
    ----------
    name: str
        Name for the agent.
    build_controller: function(int, str) -> RNNCell
        A function that accepts an integer representing an output size,
        as well as a string giving a scope name, and returns an instance
        of RNNCell that accepts observations as input and outputs utilities.
    heads: list of AgentHead instances
        The heads that determine the agent's output based on utilities.

    """
    def __init__(self, name, build_controller, heads):
        self.name = name

        self.heads = OrderedDict([(head.name, head) for head in heads])
        assert len(set(self.heads)) == len(heads), "Multiple heads with the same name."

        self._head_offsets = dict()
        offset = 0

        for k, head in self.heads.items():
            head.set_agent(self)
            start = offset
            offset += head.n_params
            end = offset
            self._head_offsets[k] = (start, end)

        self.size = offset
        self.controller = build_controller(self.size, self.name)
        assert isinstance(self.controller, ScopedCell)

        self.set_params_op = None
        self.flat_params = None

        self.build_started = False

    def trainable_variables(self, for_opt):
        return trainable_variables(self.controller.scope.name, for_opt=for_opt)

    def build_core_signals(self, context):
        context.get_signal('utils', self)

    def generate_signal(self, key, context):
        self.build_started = True

        if key == 'utils':
            obs = context.get_signal('obs')
            batch_size = context.get_signal('batch_size')

            T = cfg.T
            if T is None:
                initial_state = self.controller.zero_state(batch_size, tf.float32)
                utils, _ = dynamic_rnn(
                    self.controller, obs, initial_state=initial_state,
                    parallel_iterations=1, swap_memory=False, time_major=True)
            else:
                # If we know T at graph creation time, we can run the controller
                # using a python while loop, rather than a tensorflow while loop.
                # Using a tensorflow while loop makes some things impossible, such as
                # second order differentiation.

                state = self.controller.zero_state(batch_size, tf.float32)

                utils = []
                for t in range(T):
                    u, state = self.controller(obs[t, ...], state)
                    utils.append(u)
                utils = tf.stack(utils)

            return utils
        else:
            raise Exception("NotImplemented")

    def __getitem__(self, head_name):
        return self.get_head(head_name)

    def get_head(self, head_name):
        try:
            return self.heads[head_name]
        except KeyError:
            raise Exception("Agent {} does not have a head named {}.".format(self, head_name))

    def get_utils(self, head_name, context):
        if not isinstance(head_name, str):
            head_name = head_name.name

        utils = context.get_signal('utils', self, gradient=True)

        start, end = self._head_offsets[head_name]
        return utils[..., start:end]

    def get_one_step_utils(self, obs, controller_state, head_name):
        self.build_started = True
        utils, next_controller_state = self.controller(obs, controller_state)
        start, end = self._head_offsets[head_name]
        return utils[..., start:end], next_controller_state

    def add_head(self, head, existing_head=None, start=None, end=None):
        """ Add a head to the agent. Mainly used for attaching multiple heads to the same location. """
        assert (start is None) == (end is None)
        assert (existing_head is None) != (start is None)
        assert head.name not in self.heads

        if existing_head is not None:
            if not isinstance(existing_head, str):
                existing_head = existing_head.name
            start, end = self._head_offsets[existing_head]

        assert 0 <= start < end <= self.size

        self._head_offsets[head.name] = (start, end)
        self.heads[head.name] = head
        head.set_agent(self)

    def build_set_params(self):
        self.build_started = True
        if self.set_params_op is None:
            variables = self.trainable_variables(for_opt=False)
            self.flat_params_ph = tf.placeholder(
                tf.float32, lst_to_vec(variables).shape, name="{}_flat_params_ph".format(self.name))
            params_lst = vec_to_lst(self.flat_params_ph, variables)

            ops = []
            for p, v in zip(params_lst, variables):
                op = v.assign(p)
                ops.append(op)
            self.set_params_op = tf.group(*ops, name="{}_set_params".format(self.name))

    def set_params_flat(self, flat_params):
        self.build_set_params()
        sess = tf.get_default_session()
        sess.run(self.set_params_op, feed_dict={self.flat_params_ph: flat_params})

    def build_get_params(self):
        self.build_started = True
        if self.flat_params is None:
            variables = self.trainable_variables(for_opt=False)
            self.flat_params = tf.identity(lst_to_vec(variables), name="{}_flat_params".format(self.name))

    def get_params_flat(self):
        self.build_get_params()
        sess = tf.get_default_session()
        flat_params = sess.run(self.flat_params)
        return flat_params

    def deepcopy(self, new_name):
        if self.build_started:
            raise ValueError("Cannot copy Agent once the build process has started.")

        new_agent = deepcopy(self)
        new_agent.name = new_name

        if hasattr(new_agent.controller, 'name'):
            new_agent.controller.name = "copy_of_" + self.controller.name
        return new_agent
