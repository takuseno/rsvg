## RSVG
Recurrent Stochastic Value Gradient implementation with Tensorflow.

https://arxiv.org/pdf/1512.04455.pdf

## requirements
- Python3

## dependencies
- tensorflow
- gym[atari]
- opencv-python
- git+https://github.com/imai-laboratory/lightsaber

## usage
### training
```
$ python train.py [--render] --final-steps 10000000
```

### playing
```
$ python train.py [--render] --load {path of models} --demo
```

### implementation
This is inspired by following projects.

- [DQN](https://github.com/imai-laboratory/dqn)
