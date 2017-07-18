import numpy as np


class RolloutBatch(dict):
    """ Assumes components are stored with shape (time, batch_size) + element_shape,
        where `element_shape` is the shape of elements in the component.

    Observations, actions and rewards are given special treatment, but other
    per-time step attributes can also be supplied as kwargs.

    Parameters
    ----------
    obs: list-like
        Rollout observations.
    actions: list-like
        Rollout actions.
    rewards: list-like
        Rollout rewards.
    info: list-like (each element actions dict)
        Per-time-step information.
    metadata: dictionary
        Information about the batch.
    kwargs:
        Other columns to include.

    """
    def __init__(
            self, obs=None, actions=None, rewards=None,
            info=None, metadata=None, **kwargs):

        kwargs['obs'] = list([] if obs is None else obs)
        kwargs['actions'] = list([] if actions is None else actions)
        kwargs['rewards'] = list([] if rewards is None else rewards)

        for k, v in kwargs.items():
            self[k] = list(v)

        self._info = list(info or [])
        self._metadata = metadata or {}

    def _get(self, key):
        """ Internal counterpart to `__getitem__` which returns list, not array. """
        try:
            value = super(RolloutBatch, self).__getitem__(key)
        except KeyError:
            value = []
            self[key] = value
        return value

    def __getitem__(self, key):
        val = super(RolloutBatch, self).__getitem__(key)
        return np.array(val)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            super(RolloutBatch, self).__getattr__(key)

    @property
    def o(self):
        return self.obs

    @property
    def a(self):
        return self.actions

    @property
    def r(self):
        return self.rewards

    @property
    def _obs(self):
        return self._get("obs")

    @property
    def _actions(self):
        return self._get("actions")

    @property
    def _rewards(self):
        return self._get("rewards")

    @property
    def info(self):
        return self._info

    @property
    def T(self):
        return len(self._obs)

    @property
    def batch_size(self):
        return self._obs[0].shape[0]

    @property
    def obs_shape(self):
        return self._obs[0].shape[1:]

    @property
    def action_shape(self):
        return self._actions[0].shape[1:]

    @property
    def reward_shape(self):
        return self._rewards[0].shape[1:]

    def __len__(self):
        return self.T

    def clear(self):
        self._o = []
        self._a = []
        self._r = []
        self._info = []
        self._metadata = {}
        super(RolloutBatch, self).clear()

    def append(self, o, a, r, info=None, **kwargs):
        self._get('obs').append(o)
        self._get('actions').append(a)
        self._get('rewards').append(r)

        if info is None:
            info = {}
        self._info.append(info)

        for k, v in kwargs.items():
            self._get(k).append(v)

    def extend(self, other):
        if not isinstance(other, RolloutBatch):
            raise Exception("Cannot concatenate a RolloutBatch with {}.".format(other))

        self._info.extend(other._info)

        for k in other:
            self._get(k).extend(other._get(k))
        self._metadata.update(other._metadata)

    @staticmethod
    def concat(rollouts):
        rollouts = list(rollouts)
        if not rollouts:
            return []

        info = list_concat([r.info for r in rollouts])

        all_keys = set()
        for r in rollouts:
            all_keys |= r.keys()

        kwargs = {}
        for k in all_keys:
            kwargs[k] = list_concat([r._get(k) for r in rollouts])

        metadata = {}
        for r in rollouts:
            metadata.update(r._metadata)

        return RolloutBatch(info=info, metadata=metadata, **kwargs)

    def __add__(self, other):
        if not isinstance(other, RolloutBatch):
            raise Exception("Cannot concatenate a RolloutBatch with {}.".format(other))
        return RolloutBatch.concat([self, other])


def list_concat(lsts):
    return [item for l in lsts for item in l]
