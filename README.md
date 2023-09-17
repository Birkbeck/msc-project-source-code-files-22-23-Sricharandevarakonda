# Cloud-Based Reinforcement Learning System for Autonomous Vehicles

Welcome to the Cloud-Based Reinforcement Learning System for Autonomous Vehicles repository! This project aims to develop a cloud based system for training and deploying autonomous vehicles using reinforcement learning techniques. The system leverages cloud computing resources to accelerate training and enable real-time decision-making for safe and efficient autonomous driving.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Autonomous vehicles are rapidly evolving and have the potential to revolutionize transportation. This project focuses on implementing a cloud-based solution that employs reinforcement learning algorithms to train autonomous vehicles. By utilizing cloud resources, we can significantly speed up the training process, allowing the vehicles to learn complex driving behaviors more efficiently.

## Features

- **Cloud-Based Training:** Leverage the power of cloud computing to train autonomous vehicles on vast amounts of driving data, reducing training time.
- **Reinforcement Learning:** Implement state-of-the-art reinforcement learning algorithms to enable vehicles to learn optimal driving strategies.
- **Real-Time Decision Making:** The trained models can make real-time decisions based on sensor input, enabling safe and efficient autonomous driving.
- **Scalability:** The architecture is designed to be scalable, allowing for the integration of more vehicles and advanced algorithms.
- **Data Management:** Efficiently manage and preprocess large-scale driving datasets for training purposes.

## Installation

1. Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/Sricharandevarakonda/Autonomous_RL.git
   ```
1. Navigate to the project directory:
   ```bash
   cd Autonomous_RL
   ```
1. Set up a virtual environment (recommended) and install the required dependencies:
  ```bash
      virtualenv venv
      source venv/bin/activate
      pip install -r requirements.txt
   ```
## Usage
The model code directory contains following files

• car environment.py

• ddpg torch.py

• ddpg train.py

• dqn tensorflow.py

• dqn train.py

The file car environment.py contains the definition of the Carla environment.
This environment provides a step() function can accept action from
the RL agent and return a tuple of (next state, reward, done, info).
The agent reads the state and decides the next action based on that. This
is the way the training loop continues for each episode.
The RM agents for DDPG and DQN models are defined inside the files
ddpg torch.py and dqn tensorflow.py respectively. To train both the models,
separate scripts are provided inside files ddpg train.py and dqn train.py.
In these file, the environment configuration to be trained (without trafficwith
trafficweather condition) has to be defined while initialing the environment.
The signature for the function to initialize the environment is as follows:
   ```bash
CarEnv(traffic=False, random weather=False)
   ```
To start the simulation with traffic, the parameter traffic has to be set to
True. Similarly, to use a random wether from available presets, the parameter
random weather should be set to True.

## License
This project is licensed under the MIT License.

