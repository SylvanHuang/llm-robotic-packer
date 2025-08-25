import os
import json
import time
import random
from datetime import datetime
from typing import List

from envs.bin_packing_env import BinPacking3DEnv
from envs.state_manager import save_bin_state, check_collision, is_supported, is_within_bounds
from envs.metrics import save_run_metrics
from config import choose_rotation_and_anchor, generate_path

# ------------------------ Runtime knobs ------------------------
BIN_DIMS = [10, 10, 10]
MAX_BOXES = 25
RANDOM_SEED = 7

# Quieten various progress bars / threads that can trip macOS accelerators
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

random.seed(RANDOM_SEED)

# ------------------------ Helper utils ------------------------
def _volume(size: List[int]) -> int:
    return int(size[0] * size[1] * size[2])

def _bin_volume() -> int:
    return int(BIN_DIMS[0] * BIN_DIMS[1] * BIN_DIMS[2])

def _fill_ratio(placed_boxes: List[dict]) -> float:
    used = sum(_volume(pb["size"]) for pb in placed_boxes)
    return used / max(1, _bin_volume())

def _count_555(placed_boxes: List[dict]) -> int:
    return sum(1 for pb in placed_boxes if sorted(pb["size"]) == [5, 5, 5])

# ------------------------ Smart sampler ------------------------
def _sample_size_for_phase(fill: float, big_cubes_used: int) -> List[int]:
    """
    Sample a size with dims in {2,3,4,5}, each ≤ 5, and by default at most ONE side == 5.
    Early phase: allow some 5s to build planes; late phase: favor 2–3–4 bricks to finish cavities.
    """
    # Base weights by phase (favor smaller later)
    if fill < 0.35:
        weights = {2: 1, 3: 3, 4: 4, 5: 3}
        allow_555 = big_cubes_used < 2 and random.random() < 0.05  # at most two 5-cubes total
    elif fill < 0.7:
        weights = {2: 2, 3: 4, 4: 4, 5: 2}
        allow_555 = False
    else:
        weights = {2: 5, 3: 4, 4: 3, 5: 1}
        allow_555 = False

    def draw_dim() -> int:
        bag = []
        for v, w in weights.items():
            bag.extend([v] * w)
        return random.choice(bag)

    # Rarely allow one 5×5×5 early
    if allow_555 and random.random() < 0.5:
        return [5, 5, 5]

    # Otherwise enforce "at most one side equals 5"
    dims = []
    count5 = 0
    for _ in range(3):
        v = draw_dim()
        if v == 5:
            if count5 >= 1:
                v = random.choice([2, 3, 4])
            else:
                count5 += 1
        dims.append(v)
    return dims

def _has_any_anchors(state_path: str) -> bool:
    try:
        with open(state_path, "r") as f:
            st = json.load(f)
        anchors = st.get("anchors_indexed", [])
        for item in anchors:
            if item.get("anchors"):
                return True
    except Exception:
        pass
    return False

def _generate_feasible_box(placed_boxes: List[dict]) -> List[int]:
    """
    LfD-style feasibility loop:
      1) propose a size (≤5 per dim, <=1 side==5, phase-aware),
      2) write state & enumerate anchors,
      3) if no anchors for ANY rotation -> shrink and retry.
    """
    fill = _fill_ratio(placed_boxes)
    big_cubes_used = _count_555(placed_boxes)

    hi = 5
    for _retry in range(6):
        # sample under the current ceiling
        size = _sample_size_for_phase(fill, big_cubes_used)
        # clamp by hi and avoid re-growing
        size = [min(x, hi) for x in size]
        # avoid degenerate extremely small volume early
        if fill < 0.2 and _volume(size) < 24:
            # bump one side up to help create planes
            i = random.randrange(3)
            size[i] = min(5, max(size[i], 4))

        # Prepare a transient "box" to compute anchors
        probe_box = {"size": list(map(int, size)), "path": []}
        save_bin_state(
            placed_boxes=placed_boxes,
            new_box=probe_box,
            bin_dims=BIN_DIMS,
            path="instructions/bin_state.json",
        )
        if _has_any_anchors("instructions/bin_state.json"):
            return probe_box["size"]

        # shrink ceiling and try again
        hi = max(2, hi - 1)

    # Last resort: 2×2×2
    return [2, 2, 2]

def generate_smart_box(placed_boxes: List[dict]) -> dict:
    """Produce the next box using the feasibility-aware sampler above."""
    return {"size": _generate_feasible_box(placed_boxes), "path": []}

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

# ------------------------ Main loop ------------------------
def main():
    env = BinPacking3DEnv()
    placed_boxes = []

    run_stats = {
        "pick_calls": 0,
        "path_calls": 0,
        "pick_invalid_json": 0,
        "pick_unknown_anchor": 0,
        "path_invalid_json": 0,
        "path_not_at_target": 0,
        "pick_latency": [],
        "path_latency": [],
        "rotation_hist": {},
        "paths": [],
    }

    for i in range(MAX_BOXES):
        print(f"\n🎯 Preparing box {i + 1}...")

        # curriculum + feasibility check
        box = generate_smart_box(placed_boxes)
        original_size = list(box["size"])

        save_bin_state(
            placed_boxes=placed_boxes,
            new_box=box,
            bin_dims=BIN_DIMS,
            path="instructions/bin_state.json",
        )

        max_attempts = 3
        feedback_pick = (
            "Constraints: no overhangs; the box's base must be fully supported at the final Z. "
            "Prefer corners (minimal x and y) when feasible. Output JSON only."
        )

        for attempt in range(max_attempts):
            print(f"⏳ Pick attempt {attempt + 1}...")
            run_stats["pick_calls"] += 1
            t0 = time.perf_counter()
            pick = choose_rotation_and_anchor(feedback=feedback_pick)
            run_stats["pick_latency"].append(time.perf_counter() - t0)

            if not pick or "rotation_index" not in pick or "anchor_id" not in pick:
                run_stats["pick_invalid_json"] += 1
                print("❌ Invalid pick response.")
                feedback_pick = (
                    "Return JSON: {'rotation_index': <int>, 'anchor_id': 'rX_aY'}. "
                    "Do not invent IDs."
                )
                continue

            # Map ID -> (chosen_size, final_pos)
            with open("instructions/bin_state.json", "r") as f:
                state = json.load(f)
            chosen_size, final_pos = _lookup_anchor(state, pick["rotation_index"], pick["anchor_id"])
            if not chosen_size or not final_pos:
                run_stats["pick_unknown_anchor"] += 1
                print("❌ Pick refers to unknown rotation/anchor.")
                feedback_pick = "Choose an anchor_id from anchors_indexed. Do not invent IDs."
                continue

            # rotation histogram
            ridx = int(pick.get("rotation_index", -1))
            run_stats["rotation_hist"][str(ridx)] = run_stats["rotation_hist"].get(str(ridx), 0) + 1

            if chosen_size != original_size:
                print(f"🔄 LLM rotated the box: {original_size} -> {chosen_size}")

            # Validate the chosen final pose before asking for a path
            if not is_within_bounds(final_pos, chosen_size, BIN_DIMS):
                print("❌ Out of bounds.")
                feedback_pick = (
                    "Selected anchor is out of bounds. "
                    "Prefer corners and wall-flush placements. Pick another."
                )
                continue
            if check_collision(final_pos, chosen_size, placed_boxes):
                print("❌ Collision at selected anchor.")
                feedback_pick = "Selected anchor collides with existing boxes. Pick another."
                continue
            if not is_supported(final_pos, chosen_size, placed_boxes):
                print("❌ Not supported at selected anchor.")
                feedback_pick = (
                    "NO OVERHANGS: the entire base must be supported by floor or box tops at the same Z. "
                    "Pick a different anchor with full support."
                )
                continue

            # Now ask for a path to this fixed target
            feedback_path = (
                "Path rules: axis-aligned; start from above bin; final approach is a straight vertical descent onto the target; "
                "end exactly at the target coordinates. Output JSON: {'path': [[x,y,z], ...]}"
            )
            for path_attempt in range(2):
                run_stats["path_calls"] += 1
                t1 = time.perf_counter()
                path_resp = generate_path(final_pos, feedback=feedback_path)
                run_stats["path_latency"].append(time.perf_counter() - t1)

                if not path_resp or "path" not in path_resp or not path_resp["path"]:
                    run_stats["path_invalid_json"] += 1
                    print("❌ Invalid path JSON.")
                    feedback_path = "Return JSON with 'path': [[x,y,z], ...] ending exactly at the target."
                    continue

                box["size"] = chosen_size
                box["path"] = path_resp["path"]

                # Final sanity: last point must equal final_pos
                if box["path"][-1] != final_pos:
                    run_stats["path_not_at_target"] += 1
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
                run_stats["paths"].append(list(map(list, box["path"])))

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
                # Failed to get a good path; retry pick with sharper hint
                feedback_pick = (
                    "Failed to produce a valid path to your selected anchor. "
                    "Pick a different anchor closer to the corner and fully supported."
                )

            # If we reached here via successful path, stop retrying pick
            if box["path"]:
                break

        else:
            print("❌ Failed to place box after 3 attempts.")

    # ---- write metrics (new)
    save_run_metrics(BIN_DIMS, placed_boxes, run_stats, snapshot_dir=getattr(env, "snapshot_dir", None))
    env.close()


if __name__ == "__main__":
    main()