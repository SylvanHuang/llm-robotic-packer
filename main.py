from envs.bin_packing_env import BinPacking3DEnv
from envs.state_manager import save_bin_state
from envs.validator import validate_instruction
from envs.metrics import compute_metrics
from llm_generate import call_gpt4_for_path
import random

NUM_BOXES = 5
MAX_RETRIES = 3

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
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n🔁 Attempt {attempt} for box {i+1}")

        env.box_instruction = new_box

        save_bin_state(
            placed_boxes=env.placed_boxes,
            new_box=new_box,
            bin_dims=[env.bin_width, env.bin_height, env.bin_depth],
            path="instructions/bin_state.json"
        )

        call_gpt4_for_path()
        env.load_instruction()

        is_valid = validate_instruction(
            box_instruction=env.box_instruction,
            placed_boxes=env.placed_boxes,
            bin_dims=[env.bin_width, env.bin_height, env.bin_depth]
        )

        if is_valid:
            env.current_step = 0
            env.box_position = env.box_instruction["path"][0]

            done = False
            while not done:
                env.render()
                _, _, done, _, _ = env.step(0)

            # ✅ Print metrics
            metrics = compute_metrics(env.placed_boxes, [env.bin_width, env.bin_height, env.bin_depth])
            print(f"\n📊 Metrics after box {i+1}:\n{metrics}")
            success = True
            break
        else:
            print("⚠️ Invalid placement. Retrying...\n")

    if not success:
        print(f"❌ Failed to place box {i+1} after {MAX_RETRIES} attempts.")

env.close()
print("\n✅ Simulation complete!")
