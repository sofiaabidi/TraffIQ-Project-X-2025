# **Understanding SUMO & Traci**

## What is it?
SUMO (Simulation of Urban Mobility) is an open source, microscopic, multi-modal traffic simulation software. It allows to simulate how a complex traffic flow consisting of many individual vehicles moves through a given road network structure after a given traffic demand.

* SUMO has a fast OpenGL graphical user interface that enables the user to design each element of each roadway at an intersection, including the number of roadway strips, directions, and the number of lanes, the location of traffic signals and the phase sequence and duration of traffic signals. 

* The package Traci in SUMO can easily enable it to communicate with python, which facilitates the simulation. 

* By calling Traci, various data from the traffic simulation, such as vehicle waiting time, number of vehicles on the road, and vehicle speed, can be obtained in real time enabling us to optimize the model better.

![SUMO_emergency](https://upload.wikimedia.org/wikipedia/commons/1/1e/Eclipse_SUMO%2C_screenshot_showing_two_microscopic_views_in_SUMO_version_1.6.0.png)

## Emergency Vehicle Preemption

What we have implemented is the follows:

1. Detects emergency vehicles based on their type and identifies their movement direction (NS/EW).
2. Predicts the next traffic light they will encounter and adjusts its phase accordingly.
3. Extends green duration or shortens other phases to give emergency vehicles right of way.
4. Resets signals to normal operation once emergency vehicles have passed.


## Result

<video width="800" autoplay loop muted>
  <source src="emergency.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>