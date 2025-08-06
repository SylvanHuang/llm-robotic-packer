# simulate_llm.py

import json
import random
import time
import os

def generate_random_path(box_size, bin_depth):
    x, y = random.randint(0, 7), random.randint(0, 7)
    z_start = 10
    z_end = 0
    path = [[x, y, z] for z in range(z_start, z_end - 1, -1)]
    return path

def mock_llm_response(state_path="instructions/bin_state.json", instruction_path="instructions/instruction.json"):
    time.sleep(1)  # simulate LLM thinking...

    with open(state_path, 'r') as f:
        state = json.load(f)

    new_box = state["new_box"]
    size = new_box["size"]
    color = new_box["color"]

    # Simulate LLM generating a valid path for that box
    path = generate_random_path(size, bin_depth=state["bin"]["dimensions"][2])

    new_instruction = {
        "size": size,
        "color": color,
        "path": path
    }

    with open(instruction_path, 'w') as f:
        json.dump(new_instruction, f, indent=2)

    print(f"[LLM] Instruction generated at {instruction_path}")
