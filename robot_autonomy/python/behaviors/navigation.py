

import py_trees
import transforms3d

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import rclpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class GetLocationFromQueue(py_trees.behaviour.Behaviour):
    """Gets a location name from the queue"""

    def __init__(self, name, location_dict):
        super(GetLocationFromQueue, self).__init__(name)
        self.location_dict = location_dict
        self.bb = py_trees.blackboard.Blackboard()

    def update(self):
        """Checks for the status of the navigation action"""
        loc_list = self.bb.get("loc_list")
        if len(loc_list) == 0:
            self.logger.info("No locations available")
            return py_trees.common.Status.FAILURE
        else:
            target_location = loc_list.pop()
            self.logger.info(f"Selected location {target_location}")
            target_pose = self.location_dict[target_location]
            self.bb.set("target_pose", target_pose)
            return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")


class GoToPose(py_trees.behaviour.Behaviour):
    """Wrapper behavior around the `move_base` action client"""

    def __init__(self, name, pose, node):
        super(GoToPose, self).__init__(name)
        self.pose = pose
        self.client = None
        self.node = node
        self.bb = py_trees.blackboard.Blackboard()

    def initialise(self):
        """Sends the initial navigation action goal"""
        # Check if there is a pose available in the blackboard
        try:
            target_pose = self.bb.get("target_pose")
            if target_pose is not None:
                self.pose = target_pose
        except:
            pass

        self.client = ActionClient(self.node, NavigateToPose, "/navigate_to_pose")
        self.client.wait_for_server()

        self.goal_status = None
        x, y, theta = self.pose
        self.logger.info(f"Going to [x: {x}, y: {y}, theta: {theta}] ...")
        self.goal = self.create_move_base_goal(x, y, theta)
        self.send_goal_future = self.client.send_goal_async(self.goal)
        self.send_goal_future.add_done_callback(self.goal_callback)

    def goal_callback(self, future):
        res = future.result()
        if res is None or not res.accepted:
            return
        future = res.get_result_async()
        future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        # If there is a result, we consider navigation completed and save the
        # result code to be checked in the `update()` method.
        self.goal_status = future.result().status

    def update(self):
        """Checks for the status of the navigation action"""
        # If there is a result, we can check the status of the action directly.
        # Otherwise, the action is still running.
        if self.goal_status is not None:
            if self.goal_status == GoalStatus.STATUS_SUCCEEDED:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.client = None
        self.bb.set("target_pose", None)

    def create_move_base_goal(self, x, y, theta):
        """Creates a MoveBaseGoal message from a 2D navigation pose"""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        quat = transforms3d.euler.euler2quat(0, 0, theta)
        goal.pose.pose.orientation.w = quat[0]
        goal.pose.pose.orientation.x = quat[1]
        goal.pose.pose.orientation.y = quat[2]
        goal.pose.pose.orientation.z = quat[3]
        return goal


# Define the Deep Q-Network (DQN) architecture
class DeepQNetwork(keras.Model):
    def __init__(self, n_actions, input_shape):
        super().__init__()
        self.input_layer = layers.Input(input_shape)
        self.dense1 = layers.Dense(64, activation='relu')(self.input_layer)
        self.dense2 = layers.Dense(64, activation='relu')(self.dense1)
        self.output_layer = layers.Dense(n_actions, activation='linear')(self.dense2)

    def call(self, inputs):
        return self.output_layer(inputs)

class NavigationAgent(Node):
    def __init__(self):
        super().__init__('navigation_agent')

        # Initialize reinforcement learning parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = 0.001  # Learning rate
        self.batch_size = 64  # Batch size for training
        self.memory_size = 10000  # Size of the replay buffer

        # Initialize robot state and action space
        self.n_actions = 5  # Number of possible actions (e.g., move forward, turn left, turn right, etc.)
        self.state_size = 24  # Size of the state vector (e.g., laser scan data)

        # Initialize the DQN model
        self.dqn_model = DeepQNetwork(self.n_actions, (self.state_size,))
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

        # Initialize the replay buffer and other variables
        self.replay_buffer = []
        self.current_state = None
        self.target_model = None

        # Subscribe to the laser scan topic and the velocity command topic
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def scan_callback(self, scan_msg):
        # Preprocess the laser scan data and update the current state
        self.current_state = self.preprocess_scan(scan_msg)

        # Select an action using the DQN model
        action = self.select_action()

        # Execute the action and observe the next state and reward
        next_state, reward, done = self.take_action(action)

        # Store the experience in the replay buffer
        self.store_experience(self.current_state, action, reward, next_state, done)

        # Train the DQN model on a batch of experiences from the replay buffer
        self.train_dqn()

        # Update the target model periodically
        if self.episode_step % 1000 == 0:
            self.update_target_model()

        # Update the current state
        self.current_state = next_state

    def preprocess_scan(self, scan_msg):
        # Preprocess the laser scan data and convert it to a state vector
        # (e.g., normalize the ranges, remove invalid measurements, etc.)
        state = np.array([range for range in scan_msg.ranges if range > 0.0])
        return state

    def select_action(self):
        # Select an action based on the current state and the DQN model
        if np.random.rand() <= self.epsilon:
            # Explore: Select a random action
            action = np.random.randint(self.n_actions)
        else:
            # Exploit: Select the action with the highest Q-value
            state_tensor = tf.convert_to_tensor([self.current_state], dtype=tf.float32)
            q_values = self.dqn_model(state_tensor)
            action = np.argmax(q_values[0])

        return action

    def take_action(self, action):
        # Execute the selected action and observe the next state and reward
        # (e.g., send velocity commands, update the robot's pose, calculate the reward)
        next_state = self.current_state  # Placeholder for the next state
        reward = 0.0  # Placeholder for the reward
        done = False  # Placeholder for the terminal state

        return next_state, reward, done

    def store_experience(self, state, action, reward, next_state, done):
        # Store the experience in the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

        # Limit the size of the replay buffer
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.pop(0)

    def train_dqn(self):
        # Sample a batch of experiences from the replay buffer
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = np.random.choice(self.replay_buffer, batch_size, replace=False)

        # Separate the batch into separate arrays
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Calculate the target Q-values using the target model
        next_q_values = self.target_model.predict(next_states)
        target_q_values = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)

        # Update the DQN model using the target Q-values
        with tf.GradientTape() as tape:
            q_values = self.dqn_model(states)
            one_hot_actions = tf.one_hot(actions, self.n_actions)
            masked_q_values = tf.reduce_sum(one_hot_actions * q_values, axis=1)
            loss = tf.reduce_mean((target_q_values - masked_q_values) ** 2)

        grads = tape.gradient(loss, self.dqn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn_model.trainable_variables))

        # Update the exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        # Update the target model weights with the DQN model weights
        self.target_model.set_weights(self.dqn_model.get_weights())

def main(args=None):
    rclpy.init(args=args)
    navigation_agent = NavigationAgent()
    rclpy.spin(navigation_agent)
    navigation_agent.destroy_node()
    rclpy.shutdown()
