# test_env.py

from envs.bin_packing_env import BinPacking3DEnv
from envs.state_manager import save_bin_state
from llm_generate import call_gpt4_for_path
import random

NUM_BOXES = 5

def generate_random_box():
    size = [random.randint(2, 3)] * 3
    color = random.choice(["red", "green", "blue", "orange", "purple"])
    return {
        "size": size,
        "color": color,
        "path": []
    }

env = BinPacking3DEnv()

for i in range(NUM_BOXES):
    print(f"\n🎯 Preparing box {i+1}...")

    new_box = generate_random_box()
    env.box_instruction = new_box

    save_bin_state(
        placed_boxes=env.placed_boxes,
        new_box=new_box,
        bin_dims=[env.bin_width, env.bin_height, env.bin_depth],
        path="instructions/bin_state.json"
    )

    print("🧠 Sending to GPT-4o...")
    call_gpt4_for_path()

    # Load and animate
    env.load_instruction()
    env.current_step = 0
    env.box_position = env.box_instruction["path"][0]

    done = False
    while not done:
        env.render()
        _, _, done, _, _ = env.step(0)

env.close()
print("\n✅ Simulation complete!")
