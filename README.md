# Sampling-Based Planning Under Temporal Logic Specifications: A Forward-Invariance Approach

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://shields.io/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-available-lightgrey)](https://your-docs-link-here.com)

---

## Overview

This project explores sampling-based motion planning methods that satisfy temporal logic specifications by leveraging forward invariance properties.  
Our goal is to develop scalable algorithms that guarantee logical correctness while efficiently exploring the state space.

## Features

- Temporal logic-based specifications based on Signal temporal Logic
- Forward invariance control synthesis
- Sampling-based planning (e.g., RRT, PRM adaptations)
- Modular and extensible framework

## Installation

## On Linux
Clone the repository:


```bash
git  clone https://github.com/gregoriomarchesini/stl_rrt_py.git
cd   stl_rrt_py
sudo bash pre_install.sh # preinstall gmp dep
pip  install -e .
```

## On Windows


```bash
git clone https://github.com/gregoriomarchesini/stl_rrt_py.git
cd  stl_rrt_py
pip install -e .
```






