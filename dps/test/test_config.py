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

        with Config(a=1, b=2, c=lambda x: x + 1):
            assert cfg['a'] == 1
            assert cfg.a == 1

            assert cfg['b'] == 2
            assert cfg.b == 2

            assert cfg['c'](2) == 3
            assert cfg.c(2) == 3

            with Config(a=10, c=100, d='a', e=100):
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

                assert set(cfg.keys()) == set('a b c d e'.split())

            assert cfg['a'] == 1
            assert cfg.a == 1

            assert cfg['b'] == 2
            assert cfg.b == 2

            assert cfg['c'](2) == 3
            assert cfg.c(2) == 3

            assert set(cfg.keys()) == set('a b c'.split())

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
