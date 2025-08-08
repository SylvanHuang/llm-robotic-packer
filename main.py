# main.py

from envs.bin_packing_env import BinPacking3DEnv
from llm_generate import call_gpt4_for_path
from envs.state_manager import save_bin_state, check_collision, is_supported
import random
import os
import json

def generate_random_box():
    return {
        "size": [random.randint(3, 6), random.randint(3, 6), random.randint(3, 6)],
        "path": []  # Will be filled in by the LLM
    }

def write_instruction_file(box, path="instructions/instruction.json"):
    box["path"] = path
    with open("instructions/instruction.json", "w") as f:
        json.dump(box, f)

def main():
    env = BinPacking3DEnv()
    placed_boxes = []

    for i in range(7):
        print(f"\n🎯 Preparing box {i + 1}...")

        box = generate_random_box()

        save_bin_state(
            placed_boxes=placed_boxes,
            new_box=box,
            bin_dims=[10, 10, 10],
            path="instructions/bin_state.json"
        )

        # 🔁 Try up to 3 times if the box placement is invalid
        max_attempts = 3
        feedback = ""
        for attempt in range(max_attempts):
            print(f"⏳ Attempt {attempt + 1}...")
            gpt_response = call_gpt4_for_path(feedback=feedback)

            if not gpt_response or "path" not in gpt_response:
                print("❌ Invalid GPT response.")
                continue

            box["size"] = gpt_response["size"]
            box["path"] = gpt_response["path"]
            final_pos = box["path"][-1]

            if check_collision(final_pos, box["size"], placed_boxes):
                print("❌ Collision detected.")
                feedback = "Your last path caused a collision with another box. Please generate a different path that avoids all collisions."
                continue

            if not is_supported(final_pos, box["size"], placed_boxes):
                print("❌ Box is floating.")
                feedback = "Your last path ended in a position where the box was floating. Boxes must be on the floor or supported by another box."
                continue

            # ✅ Valid placement
            write_instruction_file(box, box["path"])
            obs = env.reset()
            done = False
            while not done:
                obs, _, done, _, _ = env.step(0)
                env.render()

            placed_boxes.append({
                "position": box["path"][-1],
                "size": box["size"]
            })
            break  # Stop retrying on success

        else:
            print("❌ Failed to place box after 3 attempts.")

    env.close()

if __name__ == "__main__":
    main()
