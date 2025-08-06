# llm_generate.py

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

BIN_STATE_PATH = "instructions/bin_state.json"
INSTRUCTION_PATH = "instructions/instruction.json"

SYSTEM_PROMPT = """
You are a 3D robotic box-packing assistant. Your job is to place a new box into a bin by generating a realistic path (sequence of 3D coordinates) that starts from above the bin and ends with the box resting securely on the floor or on top of another box.

Here are the important physical and logical constraints:

1. The bin has fixed dimensions. Boxes must **not go outside** these dimensions.
2. Boxes **must be fully inside the bin** after placement.
3. Boxes cannot float in mid-air. The final box position must be **either on the floor (z=0)** or **directly on top of another box** (aligned and touching).
4. The placement path should simulate **gravity** — the box starts from z=10 and moves down step by step.
5. There must be **no collisions** with already placed boxes at any point along the path.
6. The output must be a **JSON object** with keys:
   - "size": [x, y, z]
   - "color": string
   - "path": list of [x, y, z] positions (starting from z=10 down to resting place)

⚠️ DO NOT include markdown or explanations — only return the JSON object.
"""



def call_gpt4_for_path():
    with open(BIN_STATE_PATH, "r") as f:
        bin_state = f.read()

    user_prompt = f"""Bin state:\n{bin_state}\n\nGenerate the path to place the new box."""

    print("⏳ Contacting GPT-4o...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.2
    )

    reply = response.choices[0].message.content

    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        print("❌ JSON parsing failed.")
        parsed = {}

    return parsed

