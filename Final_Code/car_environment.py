import carla
import cv2

import math
import time
import random

import numpy as np
from datetime import datetime

DASHCAM_CAPTURE = False
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
SECONDS_PER_EPISODE = 2000 # Episode length in terms of computation physical time.
HOST_IP = "0.0.0.0" # Carla Server IP
HOST_PORT = 2000    # Carla Server Port
VEHICLE_MODEL = "model3"
N_NPC_VEHICLES = 30

WEATHER_PRESETS = [ {"title": "ClearNoon", "preset" : carla.WeatherParameters.ClearNoon},
                    {"title": "CloudyNoon", "preset" : carla.WeatherParameters.CloudyNoon},
                    {"title": "WetNoon", "preset" : carla.WeatherParameters.WetNoon},
                    {"title": "WetCloudyNoon", "preset" : carla.WeatherParameters.WetCloudyNoon},
                    {"title": "MidRainyNoon", "preset" : carla.WeatherParameters.MidRainyNoon},
                    {"title": "HardRainNoon", "preset" : carla.WeatherParameters.HardRainNoon},
                    {"title": "SoftRainNoon", "preset" : carla.WeatherParameters.SoftRainNoon},
                    {"title": "ClearSunset", "preset" : carla.WeatherParameters.ClearSunset},
                    {"title": "CloudySunset", "preset" : carla.WeatherParameters.CloudySunset},
                    {"title": "WetSunset", "preset" : carla.WeatherParameters.WetSunset},
                    {"title": "WetCloudySunset", "preset" : carla.WeatherParameters.WetCloudySunset},
                    {"title": "MidRainSunset", "preset" : carla.WeatherParameters.MidRainSunset},
                    {"title": "HardRainSunset", "preset" : carla.WeatherParameters.HardRainSunset},
                    {"title": "SoftRainSunset", "preset" : carla.WeatherParameters.SoftRainSunset}]

class CarEnv:
    """
    Definition of Carla simulation environment client. It is created to expose an API similar to OpenAI gym/gymnasium, including reset(), step() and render() functions.
    """
    CAMERA_ON = DASHCAM_CAPTURE
    STEERING_AMOUNT = 1.0
    IMAGE_WIDTH = IMAGE_WIDTH
    IMAGE_HEIGHT = IMAGE_HEIGHT
    dashcam = None

    def __init__(self, model= "reinforcement_learning", traffic=False, random_weather=False):
        print("Initializing Carla Client")
        self.client = carla.Client(HOST_IP, HOST_PORT)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.simulation_title = model
        self.traffic = traffic
        if random_weather:
            try:
                current_weather = random.choice(WEATHER_PRESETS)
                self.world.set_weather(current_weather.preset)
                self.simulation_title += f"_{current_weather.title}"
            except Exception as e:
                print("Error while setting weather preset :", e)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter(VEHICLE_MODEL)[0]

        self.result = cv2.VideoWriter('./Training_Videos/filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (IMAGE_WIDTH, IMAGE_HEIGHT))
        self.actor_list = []
        self.frame = None

    def reset(self, ep_no):
        print("Resetting the Carla client")
        for actor in self.actor_list:
            # Clean up actors from previous episodes.
            actor.destroy()
        self.result.release()
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # Get the map's spawn points
        spawn_points = self.world.get_map().get_spawn_points()

        # Spawn N_NPC_VEHICLES vehicles randomly distributed throughout the map 
        # for each spawn point, we choose a random vehicle from the blueprint library
        if self.traffic:
            for i in range(0, N_NPC_VEHICLES):
                try:
                    npc_vehicle = self.world.try_spawn_actor(random.choice(self.blueprint_library.filter('vehicle.*')), random.choice(spawn_points))
                except Exception as e:
                    print("Error occered while spawning NPC vehicle", e)
                if npc_vehicle:
                    npc_vehicle.set_autopilot(True)
                    self.actor_list.append(npc_vehicle)

        # Setup dashboard camera                    
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.IMAGE_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.IMAGE_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")
        
        # Place dashboard camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data, ep_no))

        # This is done to make the vehicle ready faster
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        # Set up collision sensor.
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # Wait for the dashcam to be ready
        while self.dashcam is None:
            time.sleep(0.01)

        self.episode_start = time.time() # Record start time

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.dashcam

    def collision_data(self, event):
        self.collision_hist.append(event)

    def set_CAMERA_ON(self, folder_name, model_name, episode):
        self.CAMERA_ON = True

        self.result = cv2.VideoWriter(f'./{folder_name}/{model_name}_episode_{episode}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
    def render(self):
        return self.dashcam

    def process_img(self, image, ep_no):
        """
        Image processing, mostly for conversion between Carla and OpenCV formats.
        """
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 4))
        i3_ = i2[:, :, :3]
        i3 = i3_.copy()
        if self.CAMERA_ON:
            # Add frames to the video
            self.result.write(i3)
            cv2.waitKey(1)
        self.dashcam = i3

    def step(self, action):
        self.vehicle.apply_control(carla.VehicleControl(throttle=(action[0]+1)/2, steer=-action[1]*self.STEERING_AMOUNT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.dashcam, reward, done, None
