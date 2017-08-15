import pytest

from dps import cfg
from dps.utils import Config, ConfigStack


def test_basic():
    c = Config()

    c.a = 1
    assert c.a == 1
    assert c['a'] == 1

    c['b'] = 2
    assert c.b == 2
    assert c['b'] == 2

    with pytest.raises(AssertionError):
        c[1] = 2

    with pytest.raises(AssertionError):
        c['1'] = 2

    with pytest.raises(KeyError):
        c['1']

    with pytest.raises(AttributeError):
        c.c


def test_config_stack():
    old_stack = cfg._stack + []
    cfg.clear_stack()

    try:
        with pytest.raises(KeyError):
            cfg['1']

        with pytest.raises(KeyError):
            cfg['a']

        with pytest.raises(AttributeError):
            cfg.a

        with Config(a=1, b=2, c=lambda x: x + 1, x=Config(x=1), y=Config(y=2)):
            assert cfg['a'] == 1
            assert cfg.a == 1

            assert cfg['b'] == 2
            assert cfg.b == 2

            assert cfg['c'](2) == 3
            assert cfg.c(2) == 3

            assert cfg.x.x == 1
            assert cfg['x']['x'] == 1

            assert cfg.y.y == 2
            assert cfg['y']['y'] == 2

            assert set(cfg.keys()) == set('a b c x:x y:y'.split())

            with Config(a=10, c=100, d='a', e=100, y=Config(y=3), z=Config(z='a')):
                assert cfg['a'] == 10
                assert cfg.a == 10

                assert cfg['b'] == 2
                assert cfg.b == 2

                assert cfg['c'] == 100
                assert cfg.c == 100

                assert cfg['d'] == 'a'
                assert cfg.d == 'a'

                assert cfg['e'] == 100
                assert cfg.e == 100

                assert cfg.x.x == 1
                assert cfg['x']['x'] == 1

                assert cfg.y.y == 3
                assert cfg['y']['y'] == 3

                assert cfg.z.z == 'a'
                assert cfg['z']['z'] == 'a'

                assert set(cfg.keys()) == set('a b c d e x:x y:y z:z'.split())

            assert cfg['a'] == 1
            assert cfg.a == 1

            assert cfg['b'] == 2
            assert cfg.b == 2

            assert cfg['c'](2) == 3
            assert cfg.c(2) == 3

            assert set(cfg.keys()) == set('a b c x:x y:y'.split())

            with pytest.raises(KeyError):
                cfg['d']
            with pytest.raises(AttributeError):
                cfg.d

            with pytest.raises(KeyError):
                cfg['e']
            with pytest.raises(AttributeError):
                cfg.e

        with pytest.raises(KeyError):
            cfg['a']
        with pytest.raises(AttributeError):
            cfg.a
        with pytest.raises(KeyError):
            cfg['b']
        with pytest.raises(AttributeError):
            cfg.b
        with pytest.raises(KeyError):
            cfg['c']
        with pytest.raises(AttributeError):
            cfg.c
        assert set(cfg.keys()) == set()
    finally:
        ConfigStack._stack = old_stack


def test_nested():
    config = Config(
        a=1,
        b=2,
        c=Config(z=2, d=1),
        e=Config(u=Config(y=3, f=2), x=4),
        g=dict(h=10),
        i=dict(j=Config(k=11), r=1),
        l=Config(m=dict(n=12))
    )

    config.update(a=10)
    assert config.a == 10

    config.update(dict(a=100))
    assert config.a == 100

    config.update(dict(c=dict(d=100)))
    assert config.c.d == 100
    assert config.c.z == 2

    config.update(dict(e=dict(x=5)))
    assert config.e.x == 5
    assert config.e.u.y == 3
    assert config.e.u.f == 2

    config.update(dict(e=dict(u=dict(y=4))))
    assert config.e.x == 5
    assert config.e.u.y == 4
    assert config.e.u.f == 2

    config.update(dict(e=dict(u='a')))
    assert config.e.x == 5
    assert config.e.u == 'a'

    assert config.a == 100
    assert config.b == 2
    assert config.c.d == 100
    assert config.c.z == 2

    config.update(dict(i=dict(j=dict(k=120))))
    assert config.i['j'].k == 120
    assert config.i['r'] == 1

    config.update(dict(i=dict(r=dict(k=120))))
    assert config.i['j'].k == 120
    assert config.i['r']['k'] == 120


def test_get_set_item():
    config = Config(
        a=1,
        b=2,
        c=Config(z=2, d=1),
        e=Config(u=Config(y=3, f=2), x=4),
        g=dict(h=10),
        i=dict(j=Config(k=11), r=1),
        l=Config(m=dict(n=12))
    )

    assert config['a'] == 1
    config['a'] = 2
    assert config['a'] == 2

    assert config.c.z == 2
    assert config['c:z'] == 2
    config['c:z'] = 3
    assert config['c:z'] == 3
    assert config.c.z == 3

    assert config.e.u.y == 3
    assert config['e:u:y'] == 3
    config['e:u:y'] = 4
    assert config.e.u.y == 4
    assert config['e:u:f'] == 2
    assert config.e.u.f == 2
    assert config['e:u:f'] == 2
    assert config.e.x == 4
    assert config['e:x'] == 4

    assert config['i:j:k'] == 11
    assert config['i:r'] == 1
    config['i:j:k'] = 23
    config['i:r'] = 12
    assert config['i:j:k'] == 23
    assert config['i:r'] == 12

    with pytest.raises(KeyError):
        config["z"]
    with pytest.raises(AttributeError):
        config.z

    config.z = 10
    assert config.z == 10

    with pytest.raises(KeyError):
        config["c:a"]

    config.c.a = 2
    assert config["c:a"] == 2
    assert config.c.a == 2

    with pytest.raises(KeyError):
        config["c:b"]
    with pytest.raises(KeyError):
        config["c:b:c"]
    config["c:b:c"] = 40
    assert config["c:b:c"] == 40
    assert config.c.b.c == 40
    assert config["c"]["b"]["c"] == 40

    assert "c:b:c" in config

    with config:
        assert "c:b:c" in cfg
        assert cfg["c:b:c"] == config["c:b:c"]
        assert config["c:b:c"] == cfg.c.b.c
        assert cfg.c.b.c == config.c.b.c
