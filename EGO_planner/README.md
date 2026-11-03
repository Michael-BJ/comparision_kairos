# EGO-Planner

Status: empty package, no method code yet.

**Role:** Classical gradient-based local planner from depth.

**Source:** Paper: https://arxiv.org/abs/2008.08835. Code: https://github.com/ZJU-FAST-Lab/ego-planner

**Plan:** Bridge it to the Gazebo and PX4 stack of the comparison worlds.

- It does not model obstacle motion, so dynamic hits are expected.
- Probably ROS 1 while this stack is ROS 2 Jazzy; not checked yet.

Test rules: see `../README.md`.
