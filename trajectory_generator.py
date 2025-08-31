from controller import Robot, Supervisor
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

class TrajectoryGenerator:
    MOTOR_MAX_SPEED = 6.28        # Maximum speed for TIAGo motors in radians per second
    DISTANCE_GAIN = 1.0          
    HEADING_GAIN = 5.0            
    WHEEL_RADIUS = 0.10         
    AXLE_LENGTH = 0.455           
    TURN_ANGLE = np.pi          

  
    HEADING_CONTROL_GAIN = 2.8    
    DISTANCE_CONTROL_GAIN = 2.6   

    # Occupancy grid dimensions
    MAP_WIDTH = 250               
    MAP_HEIGHT = 300             

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self._init_motors()
        self._init_sensors()
        self._init_display()
        self._init_lidar()

        self.map = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT))
        self.trajectory_map = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT))
        self.kernel = np.ones((20, 20))
        self.WP = [
            (+0.641, -2.460), (+0.224, -3.072), (-0.690, -3.074), (-1.690, -2.841), (-1.703, -2.302),  
            (-1.702, -1.243), (-1.542, +0.422), (-0.382, +0.503), (+0.272, +0.503), (+0.383, +0.183),  
            (+0.733, -0.093), (+0.701, -0.600), (+0.732, -0.094), (+0.684, +0.152), (+0.100, +0.501),  
            (-0.682, +0.502), (-1.542, +0.424), (-1.762, -0.323), (-1.690, -1.242), (-1.803, -2.303),  
            (-1.683, -2.842), (-0.693, -3.072), (+0.223, -3.073), (+0.223, -3.072), (+0.643, -2.463),  
            (+0.632, -2.132), (+0.552, -2.213), (+0.714, -0.791), (+0.714, -0.792), (+0.711, +0.413)   
        ]
        self.index = 0 
        self.robot_stop = False  
        self.angles = np.linspace(2.094395, -2.094395, 667)  

    def _init_motors(self):
      
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

    def _init_sensors(self):
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

    def _init_display(self):
        self.display = self.robot.getDevice('display')
        self.marker = self.robot.getFromDef("marker").getField("translation")

    def _init_lidar(self):
        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

    def world_to_map(self, xw, yw):
        px = int(52 * xw + 124.8)
        py = int(-52 * yw + 93.834)
        px = np.clip(px, 0, self.MAP_WIDTH - 1)
        py = np.clip(py, 0, self.MAP_HEIGHT - 1)
        return px, py

    def calculate_motor_speeds(self, alpha, rho):
        if abs(alpha) > np.pi / 4:
            left_speed = -alpha * self.HEADING_CONTROL_GAIN / 2 + rho * self.DISTANCE_CONTROL_GAIN / 8
            right_speed = alpha * self.HEADING_CONTROL_GAIN / 2 + rho * self.DISTANCE_CONTROL_GAIN / 8
        else:
            left_speed = -alpha * self.HEADING_CONTROL_GAIN + rho * self.DISTANCE_CONTROL_GAIN
            right_speed = alpha * self.HEADING_CONTROL_GAIN + rho * self.DISTANCE_CONTROL_GAIN

        left_speed = np.clip(left_speed, -self.MOTOR_MAX_SPEED, self.MOTOR_MAX_SPEED)
        right_speed = np.clip(right_speed, -self.MOTOR_MAX_SPEED, self.MOTOR_MAX_SPEED)

        return left_speed, right_speed

    def update_waypoints(self, rho):
        if rho < 0.5:
            if self.index >= len(self.WP) - 1:  
                print(" = Final waypoint reached")
                self.stop_robot()
                self.show_convolved_map()
                return

            self.index += 1
            print(f" > Waypoint index: {self.index}, position: {self.WP[self.index]}")

            if self.index >= len(self.WP):
                print(f" = Waypoint reached, index: {self.index}")
                self.stop_robot()
                self.show_convolved_map()

    def stop_robot(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def show_convolved_map(self):
        cmap = signal.convolve2d(self.map, self.kernel, mode='same')
        cspace = cmap > 0.9  
        plt.imshow(cspace, cmap='gray')
        plt.title("Convolved Map")
        plt.show()

    def run(self):
        while self.robot.step(self.timestep) != -1:
            gps_values = self.gps.getValues()
            compass_values = self.compass.getValues()
            xw, yw = gps_values[0], gps_values[1]
            theta = np.arctan2(compass_values[0], compass_values[1])

            self.marker.setSFVec3f([*self.WP[self.index], 0])
            rho = np.hypot(xw - self.WP[self.index][0], yw - self.WP[self.index][1])
            alpha = np.arctan2(self.WP[self.index][1] - yw, self.WP[self.index][0] - xw) - theta
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            left_speed, right_speed = self.calculate_motor_speeds(alpha, rho)
            self.update_waypoints(rho)
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            self.process_lidar(xw, yw, theta)
            self.draw_trajectory(xw, yw)

    def process_lidar(self, xw, yw, theta):
        if self.display is None:
            print("Warning: Display device not initialized properly")
            return
        
        ranges = np.array(self.lidar.getRangeImage())
        ranges[ranges == np.inf] = 100  
        valid_ranges = ranges[80:-80]
        valid_angles = self.angles[80:-80]

        X_r = np.vstack((
            valid_ranges * np.cos(valid_angles),
            valid_ranges * np.sin(valid_angles),
            np.ones(len(valid_angles))
        ))
        w_T_r = np.array([
            [np.cos(theta), -np.sin(theta), xw],
            [np.sin(theta),  np.cos(theta), yw],
            [            0,              0,  1]
        ])

        D = w_T_r @ X_r
        for point in D.T:
            px, py = self.world_to_map(point[0], point[1])
            self.map[px, py] = min(self.map[px, py] + 0.01, 1.0) 
            color_byte = int(self.map[px, py] * 255)
            color = (color_byte << 16) | (color_byte << 8) | color_byte 
            self.display.setColor(color)
            self.display.drawPixel(px, py)

    def draw_trajectory(self, xw, yw):
        # Check if the display device is initialized
        if self.display is None:
            print("Warning: Display device not initialized properly")
            return
        px, py = self.world_to_map(xw, yw)
        self.trajectory_map[px, py] = 1.0 
        self.display.setColor(0xFF0000)    
        self.display.drawPixel(px, py)  

if __name__ == "__main__":
    traj_gen = TrajectoryGenerator()
    traj_gen.run()
