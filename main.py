# main.py

from envs.bin_packing_env import BinPacking3DEnv
from llm_generate import call_gpt4_for_path
from envs.state_manager import save_bin_state, check_collision, is_supported
import random
import os
import json

def generate_random_box():
    return {
        "size": [random.randint(2, 4), random.randint(2, 4), random.randint(2, 4)],
        "path": []  # Will be filled in by the LLM
    }

def write_instruction_file(box, path="instructions/instruction.json"):
    box["path"] = path
    with open("instructions/instruction.json", "w") as f:
        json.dump(box, f)

def main():
    env = BinPacking3DEnv()
    placed_boxes = []

    for i in range(3):
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
        final_pos = box["path"][-1]

        # ✅ Collision + support check
        if check_collision(final_pos, box["size"], placed_boxes):
            print("❌ Collision detected. Skipping box.")
            continue

        if not is_supported(final_pos, box["size"], placed_boxes):
            print("❌ Unsupported (floating) box. Skipping box.")
            continue

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
