py.test --durations=0 -m "not slow"
py.test --durations=0 -m slow --capture=no
