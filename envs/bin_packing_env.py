# envs/bin_packing_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from envs.state_manager import save_bin_state

class BinPacking3DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, instruction_path="instructions/instruction.json", render_mode="human"):
        super().__init__()
        self.bin_width = 10
        self.bin_height = 10
        self.bin_depth = 10

        self.instruction_path = instruction_path
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

        self.placed_boxes = []
        self.load_instruction()
        self.current_step = 0
        self.box_position = np.array(self.box_instruction["path"][0])

        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.bin_width, self.bin_height, self.bin_depth),
            shape=(3,),
            dtype=np.float32,
        )

    def load_instruction(self):
        with open(self.instruction_path, "r") as f:
            self.box_instruction = json.load(f)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.load_instruction()
        self.box_position = np.array(self.box_instruction["path"][0])
        return self.box_position, {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.box_instruction["path"])

        if not done:
            self.box_position = np.array(self.box_instruction["path"][self.current_step])
        else:
            self.placed_boxes.append({
                "position": self.box_instruction["path"][-1],
                "size": self.box_instruction["size"]
            })

        return self.box_position, 0.0, done, False, {}

    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        self.ax.clear()
        self.ax.set_xlim(0, self.bin_width)
        self.ax.set_ylim(0, self.bin_height)
        self.ax.set_zlim(0, self.bin_depth)

        # Draw transparent bin
        self.ax.bar3d(0, 0, 0, self.bin_width, self.bin_height, self.bin_depth,
                      alpha=0.05, color='gray', edgecolor='black')

        # Draw placed boxes
        for box in self.placed_boxes:
            px, py, pz = box["position"]
            sx, sy, sz = box["size"]
            self.ax.bar3d(px, py, pz, sx, sy, sz, color='green', alpha=0.6)

        # Draw current moving box
        x, y, z = self.box_position
        dx, dy, dz = self.box_instruction["size"]
        self.ax.bar3d(x, y, z, dx, dy, dz, color='blue', alpha=0.8)

        plt.draw()
        plt.pause(0.3)

    def close(self):
        if self.fig:
            plt.close(self.fig)
