# Adaptive Social Behavior Trees for Intuitive Human-Robot Collaboration

## Setup

### Prerequisites 
- Python 3.6 or higher
- ROS Melodic or higher
- Gazebo 9 or higher


### Running the code
1. Clone the repository
2. Create an ROS workspace package
3. Copy the two folders into the `src` folder of the ROS workspace package
4. Build the workspace
5. Run the following command to start the simulation:
```bash
  ros2 launch robot_autonomy robot_behavior_py.launch.py
```


Base code for the project is taken from [here](https://github.com/sea-bass/turtlebot3_behavior_demos.git) and 
modified to include the ToM believes and incorporated the AI techniques.