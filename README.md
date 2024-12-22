# OOP coursework☀️
## Name and Student ID🎁
Martin Chan: 23126386     
Ningwei Bai: 23059619

## Project Overview🎋
This coursework is for COMP0213: Object-Oriented Programming for Robotics and Artificial Intelligence. The project uses the PyBullet library in Python for simulations to generate data on whether an object can be grasped. The goal is to train a machine learning model to predict whether specific parameters can successfully grasp an object. The project includes two robotic hand models and two object models, creating four combinations to collect data for machine learning.

## Environment 🔔
* Operating system: Windows 11 23H2 (OS for owner)
* Python Version: 3.8.20 
* Required Dependencies: See requirements.txt

## Usage🎉
### Running the Main Script
Main script file: _main.py_
### Model file
The .urdf files for the model used are located in ./model. The three fingers gripper is located in model/threeFingers/sdh.urdf. The model for the cube object is: ./model/cube.urdf, and the model for the sphere object is: ./model/ball.urdf.
### Data file
The data pre-generated from the main function is in ./data, which contains the pre-generated data for two fingers gripper and three fingers gripper both with the cube object and the sphere object. The added noise data is also provided in the ./data file.
### Plots file
The plots that generated from the _main.py_ is saved in the _./plots_ file and the _plot_between_noise_ file.
