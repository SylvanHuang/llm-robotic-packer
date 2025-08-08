# main.py

from envs.bin_packing_env import BinPacking3DEnv
from envs.state_manager import save_bin_state, check_collision, is_supported, is_within_bounds
from llm_generate import choose_rotation_and_anchor, call_gpt4_for_path_to_target
import random
import os
import json
from datetime import datetime

BIN_DIMS = [10, 10, 10]

def generate_random_box():
    return {
        "size": [random.randint(3, 6), random.randint(3, 6), random.randint(3, 6)],
        "path": []
    }

def write_instruction_file(box, path="instructions/instruction.json"):
    box["path"] = path
    with open("instructions/instruction.json", "w") as f:
        json.dump(box, f)

def _lookup_anchor(state, rotation_index, anchor_id):
    for item in state["anchors_indexed"]:
        if item["rotation_index"] == rotation_index:
            for a in item["anchors"]:
                if a["id"] == anchor_id:
                    return item["size"], a["pos"]
    return None, None

def main():
    env = BinPacking3DEnv()
    placed_boxes = []

    for i in range(10):
        print(f"\n🎯 Preparing box {i + 1}...")
        box = generate_random_box()
        original_size = list(box["size"])

        save_bin_state(placed_boxes=placed_boxes, new_box=box, bin_dims=BIN_DIMS, path="instructions/bin_state.json")

        max_attempts = 3
        feedback_pick = ""
        for attempt in range(max_attempts):
            print(f"⏳ Pick attempt {attempt + 1}...")
            pick = choose_rotation_and_anchor(feedback=feedback_pick)
            if not pick or "rotation_index" not in pick or "anchor_id" not in pick:
                print("❌ Invalid pick response.")
                feedback_pick = "Return JSON: {'rotation_index': <int>, 'anchor_id': 'rX_aY'}"
                continue

            # Map ID -> (chosen_size, final_pos)
            with open("instructions/bin_state.json", "r") as f:
                state = json.load(f)
            chosen_size, final_pos = _lookup_anchor(state, pick["rotation_index"], pick["anchor_id"])
            if not chosen_size or not final_pos:
                print("❌ Pick refers to unknown rotation/anchor.")
                feedback_pick = "Choose an anchor_id from anchors_indexed."
                continue

            if chosen_size != original_size:
                print(f"🔄 LLM rotated the box: {original_size} -> {chosen_size}")

            # Validate the chosen final pose before asking for a path
            if not is_within_bounds(final_pos, chosen_size, BIN_DIMS):
                print("❌ Out of bounds.")
                feedback_pick = "Selected anchor is out of bounds. Pick another."
                continue
            if check_collision(final_pos, chosen_size, placed_boxes):
                print("❌ Collision at selected anchor.")
                feedback_pick = "Selected anchor collides. Pick another."
                continue
            if not is_supported(final_pos, chosen_size, placed_boxes):
                print("❌ Not supported at selected anchor.")
                feedback_pick = "Selected anchor not supported. Pick another."
                continue

            # Now ask for a path to this fixed target
            feedback_path = ""
            for path_attempt in range(2):
                path_resp = call_gpt4_for_path_to_target(final_pos, feedback=feedback_path)
                if not path_resp or "path" not in path_resp or not path_resp["path"]:
                    print("❌ Invalid path JSON.")
                    feedback_path = "Return JSON with 'path': [[x,y,z], ...] ending exactly at the target."
                    continue

                box["size"] = chosen_size
                box["path"] = path_resp["path"]

                # Final sanity: last point must equal final_pos
                if box["path"][-1] != final_pos:
                    print("❌ Path does not end at target.")
                    feedback_path = "Your path must end exactly at the target coordinates."
                    continue

                # ✅ Valid placement (already validated pose)
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

                # 📸 Snapshot (valid only)
                if not hasattr(env, "snapshot_dir"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    env.snapshot_dir = os.path.join("snapshots", timestamp)
                    os.makedirs(env.snapshot_dir, exist_ok=True)
                placement_index = getattr(env, "snapshot_count", 0) + 1
                env.fig.savefig(os.path.join(env.snapshot_dir, f"placement_{placement_index}.png"))
                env.snapshot_count = placement_index

                break  # path success

            else:
                # Failed to get a good path; retry pick
                feedback_pick = "Failed to produce a valid path to your selected anchor. Pick a different anchor."

            # If we reached here via successful path, stop retrying pick
            if box["path"]:
                break

        else:
            print("❌ Failed to place box after 3 attempts.")

    env.close()

if __name__ == "__main__":
    main()
