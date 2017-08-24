import numpy as np

from dps.rl import RolloutBatch


def test_rollouts():
    T = 5
    batch_size = 2
    shape = (T, batch_size)
    obs_shape = (3,)
    action_shape = (5,)
    reward_shape = (2,)
    entropy_shape = (4,)
    n = np.product(shape + obs_shape)
    ne = np.product(shape + entropy_shape)

    obs = np.arange(n).reshape(*shape, *obs_shape)
    actions = np.ones(shape + action_shape)
    rewards = np.zeros(shape + reward_shape)
    entropy = 0.1*np.arange(ne).reshape(*shape, *entropy_shape)
    exploration = [0.05] * batch_size
    eagerness = 100

    info = [{'t': t} for t in range(T)]

    r1 = RolloutBatch(
        obs, actions, rewards, entropy=entropy, info=info, static=dict(exploration=exploration, eagerness=eagerness))

    for t in range(T):
        assert r1.info[t]['t'] == t

    assert np.all(r1.o == obs)
    assert np.all(r1.a == actions)
    assert np.all(r1.r == rewards)
    assert np.all(r1.entropy == entropy)

    o = np.ones((batch_size,) + obs_shape)
    a = np.ones((batch_size,) + action_shape)
    r = np.ones((batch_size,) + reward_shape)
    e = np.ones((batch_size,) + entropy_shape)

    r1.append(o, a, r, entropy=e, info={'t': T})

    assert np.all(r1.o == np.concatenate([obs, o[np.newaxis, ...]], axis=0))
    assert np.all(r1.a == np.concatenate([actions, a[np.newaxis, ...]], axis=0))
    assert np.all(r1.r == np.concatenate([rewards, r[np.newaxis, ...]], axis=0))
    assert np.all(r1.entropy == np.concatenate([entropy, e[np.newaxis, ...]], axis=0))

    for t in range(T+1):
        assert r1.info[t]['t'] == t

    for explore in r1.get_static('exploration'):
        assert explore == exploration[0]

    for eager in r1.get_static('eagerness'):
        assert eager == eagerness


def test_rollouts_shapes_and_types():
    shape = (5, 2)
    obs_shape = (3,)
    action_shape = (5,)
    reward_shape = (2,)
    entropy_shape = (4,)
    n = np.product(shape + obs_shape)
    ne = np.product(shape + entropy_shape)
    r1 = RolloutBatch(
        np.arange(n).reshape(*shape, *obs_shape),
        np.ones(shape + action_shape),
        np.zeros(shape + reward_shape),
        entropy=0.1*np.arange(ne).reshape(*shape, *entropy_shape),
    )

    assert len(r1) == shape[0]
    assert r1.T == shape[0]
    assert r1.batch_size == shape[1]
    assert r1.obs_shape == obs_shape
    assert r1.action_shape == action_shape

    assert(isinstance(r1.o, np.ndarray))
    assert(isinstance(r1.obs, np.ndarray))
    assert(isinstance(r1['obs'], np.ndarray))
    assert(isinstance(r1._get('obs'), list))
    assert(isinstance(r1._obs, list))

    assert(isinstance(r1.a, np.ndarray))
    assert(isinstance(r1.actions, np.ndarray))
    assert(isinstance(r1['actions'], np.ndarray))
    assert(isinstance(r1._get('actions'), list))
    assert(isinstance(r1._actions, list))

    assert(isinstance(r1.r, np.ndarray))
    assert(isinstance(r1.rewards, np.ndarray))
    assert(isinstance(r1['rewards'], np.ndarray))
    assert(isinstance(r1._get('rewards'), list))
    assert(isinstance(r1._rewards, list))

    assert(isinstance(r1.entropy, np.ndarray))
    assert(isinstance(r1['entropy'], np.ndarray))
    assert(isinstance(r1._get('entropy'), list))

    assert(isinstance(r1.o, np.ndarray))
    assert(isinstance(r1.a, np.ndarray))
    assert(isinstance(r1.r, np.ndarray))
    assert(isinstance(r1['entropy'], np.ndarray))

    shape2 = (10, 2)
    n = np.product(shape2 + obs_shape)
    ne = np.product(shape2 + entropy_shape)
    r2 = RolloutBatch(
        np.arange(n).reshape(*shape2, *obs_shape),
        np.ones(shape2 + action_shape),
        np.zeros(shape2 + reward_shape),
        entropy=0.1*np.arange(ne).reshape(*shape2, *entropy_shape),
    )

    assert len(r2) == shape2[0]
    assert r2.T == shape2[0]
    assert r2.batch_size == shape2[1]
    assert r2.obs_shape == obs_shape
    assert r2.action_shape == action_shape

    shape3 = (1, 2)
    n = np.product(shape3 + obs_shape)
    ne = np.product(shape3 + entropy_shape)
    r3 = RolloutBatch(
        np.arange(n).reshape(*shape3, *obs_shape),
        np.ones(shape3 + action_shape),
        np.zeros(shape3 + reward_shape),
        entropy=0.1*np.arange(ne).reshape(*shape3, *entropy_shape),
    )

    assert len(r3) == shape3[0]
    assert r3.T == shape3[0]
    assert r3.batch_size == shape3[1]
    assert r3.obs_shape == obs_shape
    assert r3.action_shape == action_shape

    combined = RolloutBatch.concat([r1, r2, r3])

    new_shape = (16, 2)

    assert len(combined) == new_shape[0]
    assert combined.T == new_shape[0]
    assert combined.batch_size == new_shape[1]
    assert combined.obs_shape == obs_shape
    assert combined.action_shape == action_shape

    assert(isinstance(combined.o, np.ndarray))
    assert(isinstance(combined.obs, np.ndarray))
    assert(isinstance(combined['obs'], np.ndarray))
    assert(isinstance(combined._get('obs'), list))
    assert(isinstance(combined._obs, list))

    assert(isinstance(combined.a, np.ndarray))
    assert(isinstance(combined.actions, np.ndarray))
    assert(isinstance(combined['actions'], np.ndarray))
    assert(isinstance(combined._get('actions'), list))
    assert(isinstance(combined._actions, list))

    assert(isinstance(combined.r, np.ndarray))
    assert(isinstance(combined.rewards, np.ndarray))
    assert(isinstance(combined['rewards'], np.ndarray))
    assert(isinstance(combined._get('rewards'), list))
    assert(isinstance(combined._rewards, list))

    assert(isinstance(combined.entropy, np.ndarray))
    assert(isinstance(combined['entropy'], np.ndarray))
    assert(isinstance(combined._get('entropy'), list))

    assert(combined.o.shape == new_shape + obs_shape)
    assert(combined.a.shape == new_shape + action_shape)
    assert(combined.r.shape == new_shape + reward_shape)
    assert(combined['entropy'].shape == new_shape + entropy_shape)

    r1.extend(r2)
    r1.extend(r3)

    assert len(r1) == new_shape[0]
    assert r1.T == new_shape[0]
    assert r1.batch_size == new_shape[1]
    assert r1.obs_shape == obs_shape
    assert r1.action_shape == action_shape

    assert(isinstance(r1.o, np.ndarray))
    assert(isinstance(r1.obs, np.ndarray))
    assert(isinstance(r1['obs'], np.ndarray))
    assert(isinstance(r1._get('obs'), list))
    assert(isinstance(r1._obs, list))

    assert(isinstance(r1.a, np.ndarray))
    assert(isinstance(r1.actions, np.ndarray))
    assert(isinstance(r1['actions'], np.ndarray))
    assert(isinstance(r1._get('actions'), list))
    assert(isinstance(r1._actions, list))

    assert(isinstance(r1.r, np.ndarray))
    assert(isinstance(r1.rewards, np.ndarray))
    assert(isinstance(r1['rewards'], np.ndarray))
    assert(isinstance(r1._get('rewards'), list))
    assert(isinstance(r1._rewards, list))

    assert(isinstance(r1.entropy, np.ndarray))
    assert(isinstance(r1['entropy'], np.ndarray))
    assert(isinstance(r1._get('entropy'), list))

    assert(r1.o.shape == new_shape + obs_shape)
    assert(r1.a.shape == new_shape + action_shape)
    assert(r1.r.shape == new_shape + reward_shape)
    assert(r1.entropy.shape == new_shape + entropy_shape)


def test_rollouts_split_join():
    T = 5
    batch_size = 2
    shape = (T, batch_size)
    obs_shape = (3,)
    action_shape = (5,)
    reward_shape = (2,)
    entropy_shape = (4,)
    n = np.product(shape + obs_shape)
    ne = np.product(shape + entropy_shape)

    obs1 = np.arange(n).reshape(*shape, *obs_shape)
    actions1 = np.ones(shape + action_shape)
    rewards1 = np.zeros(shape + reward_shape)
    entropy1 = 0.1*np.arange(ne).reshape(*shape, *entropy_shape)
    exploration1 = [0.05] * batch_size

    info1 = [{'t': t} for t in range(T)]

    r1 = RolloutBatch(
        obs1, actions1, rewards1, entropy=entropy1, info=info1, static=dict(exploration=exploration1))

    obs2 = np.arange(n).reshape(*shape, *obs_shape)
    actions2 = np.ones(shape + action_shape)
    rewards2 = np.zeros(shape + reward_shape)
    entropy2 = 0.1*np.arange(ne).reshape(*shape, *entropy_shape)
    exploration2 = [0.1] * batch_size

    info2 = [{'t': t} for t in range(T)]

    r2 = RolloutBatch(
        obs2, actions2, rewards2, entropy=entropy2, info=info2, static=dict(exploration=exploration2))

    r3 = RolloutBatch.join([r1, r2])

    assert len(r3) == T
    assert tuple(r3.get_static('exploration')) == (0.05, 0.05, 0.1, 0.1)

    r4 = RolloutBatch.join([r1, r2, r3, r3, r2, r1])

    assert len(r4) == T
    assert tuple(r4.get_static('exploration')) == (0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, .05)

    splitted = r4.split()

    for i, r in enumerate(splitted):
        assert r.get_static('exploration')[0] == r4.get_static('exploration')[i]

    for r in splitted:
        assert len(r) == T
        assert r.batch_size == 1
