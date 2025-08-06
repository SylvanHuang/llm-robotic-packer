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
You are a box-packing AI assistant. Your job is to place a new box inside a 3D bin by generating a path (sequence of positions) for the box to follow until it is placed in its final position.

Instructions:
- Input will include the bin dimensions, previously placed boxes, and a new box (size and color).
- Output must be a pure JSON object with "size", "color", and "path" keys.
- DO NOT wrap your response in markdown or any text. ONLY return the raw JSON.
- "path" should be a list of 3D [x, y, z] coordinates.
- Ensure the box stays within bin dimensions.
- Avoid overlapping already placed boxes.
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
        with open(INSTRUCTION_PATH, 'w') as f:
            json.dump(parsed, f, indent=2)
        print("✅ GPT-4o generated instruction.json successfully.")
    except Exception as e:
        print("❌ Failed to parse GPT response:", e)
        print("Raw output:\n", reply)
