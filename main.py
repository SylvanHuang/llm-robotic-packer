# main.py

from envs.bin_packing_env import BinPacking3DEnv
from llm_generate import call_gpt4_for_path
from envs.state_manager import save_bin_state
import random
import os
import json

def generate_random_box():
    return {
        "size": [random.randint(1, 3), random.randint(1, 3), random.randint(1, 3)],
        "path": []  # Will be filled in by the LLM
    }

def write_instruction_file(box, path="instructions/instruction.json"):
    box["path"] = path
    with open("instructions/instruction.json", "w") as f:
        json.dump(box, f)

def main():
    env = BinPacking3DEnv()
    placed_boxes = []

    for i in range(10):
        print(f"\n🎯 Preparing box {i + 1}...")

        box = generate_random_box()

        # Write current bin state
        save_bin_state(
            placed_boxes=placed_boxes,
            new_box=box,
            bin_dims=[10, 10, 10],
            path="instructions/bin_state.json"
        )

        # Get path from GPT-4o
        gpt_response = call_gpt4_for_path()
        if not gpt_response or "path" not in gpt_response:
            print("❌ Invalid GPT response.")
            continue

        box["size"] = gpt_response["size"]
        box["path"] = gpt_response["path"]
        write_instruction_file(box, box["path"])

        # Run simulation
        obs = env.reset()
        done = False
        while not done:
            obs, _, done, _, _ = env.step(0)
            env.render()

        placed_boxes.append({
            "position": box["path"][-1],
            "size": box["size"]
        })

    env.close()

if __name__ == "__main__":
    main()
