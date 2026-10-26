import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import importlib.util
from openai import OpenAI
import json
import re
from pathlib import Path

sys.path.insert(0, r"d:\\privacy\\privacy+persona\scripts")
from config import DEEPSEEK_API_KEY


FOUR_DIMENSION_METADATA = {
    "V": {
        "name": "Privacy Vigilance–Delegation Stance",
        "high_code": "V",
        "low_code": "D",
        "high_text": "high privacy concern and vigilance about data practices",
        "low_text": "low privacy concern and tendency to trust organizations with personal data",
    },
    "C": {
        "name": "Control Orientation",
        "high_code": "A",
        "low_code": "S",
        "high_text": "actively takes control of privacy settings and protective measures",
        "low_text": "passive about privacy controls, rarely adjusting settings or taking protective measures",
    },
    "U": {
        "name": "Privacy–Utility Tradeoff",
        "high_code": "P",
        "low_code": "U",
        "high_text": "prioritizes privacy and security over convenience and social benefits",
        "low_text": "prioritizes convenience and social utility over privacy concerns",
    },
    "L": {
        "name": "Privacy Literacy",
        "high_code": "H",
        "low_code": "L",
        "high_text": "high knowledge and understanding of security and privacy concepts",
        "low_text": "limited knowledge and understanding of security and privacy concepts",
    },
}

# Dimension order for sentence composition
DIM_ORDER = ["V", "C", "U", "L"]


@lru_cache(maxsize=1)
def load_dimension_module():
    module_path = Path(__file__).resolve().parents[1] / "persona_generator" / "4-dimensional_personality.py"
    spec = importlib.util.spec_from_file_location("privacy_dimension_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load dimension module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_persona_code(persona_traits):
    """Compute the 4-letter persona code (e.g., VAPH, DSUL) from persona traits."""
    try:
        dimension_module = load_dimension_module()
        scores = dimension_module.compute_dimension_scores({"persona_traits": persona_traits})
    except Exception:
        return None

    code_parts = []
    for dim in DIM_ORDER:
        score = scores.get(dim)
        if score is None:
            return None
        metadata = FOUR_DIMENSION_METADATA[dim]
        is_high = score >= 0.5
        code_parts.append(metadata["high_code"] if is_high else metadata["low_code"])

    return "".join(code_parts)


def build_one_sentence_description(persona_traits):
    """
    Build a single natural-language sentence describing the persona
    based solely on the four dimensions (V/C/U/L).

    Example for VAPH:
    "I am a person with high privacy concern and vigilance about data practices,
     who actively takes control of privacy settings and protective measures,
     prioritizes privacy and security over convenience and social benefits,
     and has high knowledge and understanding of security and privacy concepts."
    """
    try:
        dimension_module = load_dimension_module()
        scores = dimension_module.compute_dimension_scores({"persona_traits": persona_traits})
    except Exception:
        return None

    descriptions = []
    for dim in DIM_ORDER:
        score = scores.get(dim)
        if score is None:
            return None
        metadata = FOUR_DIMENSION_METADATA[dim]
        is_high = score >= 0.5
        descriptions.append(metadata["high_text"] if is_high else metadata["low_text"])

    # Compose into one flowing sentence:
    # "I am a person with [V], who [C], [U], and has [L]."
    sentence = (
        f"I am a person with {descriptions[0]}, "
        f"who {descriptions[1]}, "
        f"{descriptions[2]}, "
        f"and has {descriptions[3]}."
    )
    return sentence


# ============================================================
# Construct the system prompt — ONE-SENTENCE PERSONA (baseline)
# Simpler than four_dim_baseline, more informative than none_persona.
# ============================================================
def construct_system_prompt(persona_traits, persona_id):
    one_sentence = build_one_sentence_description(persona_traits)
    persona_code = build_persona_code(persona_traits) or "Unknown"

    if one_sentence is None:
        persona_block = ""
    else:
        persona_block = (
            f"[PERSONA_DESCRIPTION]\n"
            f"{one_sentence}\n"
            f"[/PERSONA_DESCRIPTION]\n\n"
        )

    system_content = (
        f"You are currently acting, expressing and making decisions only based on the persona described below. "
        f"No other personas or behaviors inconsistent with this persona should be introduced. "
        f"Before performing any task, check yourself to see if you have seen this persona. "
        f"If missing, only reply {{\"error\": \"MISSING_PERSONA\", \"the action\": \"REQUEST_RELOAD\"}}.\n\n"
        f"{persona_block}"
        f"[DECISION RULES]\n"
        f"- Anchor ALL decisions on the persona description above — it is your ONLY source of persona information.\n"
        f"- Infer likely behaviors and attitudes from the described traits; do NOT fabricate specific trait values beyond what the description suggests.\n"
        f"[/DECISION RULES]\n\n"
        f"[OUTPUT FORMAT]\n"
        f"When performing tasks, please strictly follow the output format requirements below. "
        f"Do not add any additional explanations or content outside the specified format.\n"
        f"Output format: Only output a single JSON object with the following structure:\n"
        f"{{\n\"persona\": \"{persona_id}\",\n\"task_output\": \"final content\",\n\"reason\": \"brief explanation of the reasoning behind the task_output\"\n}}"
        f"[/OUTPUT FORMAT]\n"
    )

    return system_content


def construct_user_prompt(task):
    """Baseline user prompt — only the task description."""
    user_content = f"Task Description:\n{json.dumps(task, indent=2)}"
    return user_content


def repair_json_response(raw_content):
    if raw_content is None:
        return None

    text = raw_content.strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if start_candidates:
        text = text[min(start_candidates):]

    text = _balance_json_delimiters(text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _balance_json_delimiters(text):
    pairs = {"{": "}", "[": "]"}
    closers = {v: k for k, v in pairs.items()}
    stack = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(ch)
        elif ch in closers and stack and stack[-1] == closers[ch]:
            stack.pop()

    if in_string:
        text += '"'

    while stack:
        text += pairs[stack.pop()]

    return text


def ask_deepseek(system_content, user_content):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}}
    )
    raw_content = response.choices[0].message.content
    response_text = repair_json_response(raw_content)
    try:
        if response_text is None:
            raise ValueError("Empty JSON response")
        parsed = json.loads(response_text)
    except Exception:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}}
        )
        raw_content = response.choices[0].message.content
        response_text = repair_json_response(raw_content)
        try:
            if response_text is None:
                raise ValueError("Empty JSON response")
            parsed = json.loads(response_text)
        except Exception:
            print(raw_content)
            print("Error decoding JSON response")
            return None
    print(response_text)
    return parsed


def main(persona_dir, results_dir, task_dir, task_name, persona_ids=None):
    persona_dir = os.path.join(os.path.dirname(__file__), persona_dir)
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    task_dir = os.path.join(os.path.dirname(__file__), task_dir)

    if task_name not in ["assessment_task_v2", "contextual_task_v3", "setting_task", "knowledge_task"]:
        raise ValueError("Invalid task_name. Must be one of: assessment_task_v2, contextual_task_v3, setting_task, knowledge_task")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(task_dir, f"{task_name}.json"), "r", encoding="utf-8") as f:
        task_data = json.load(f)

    persona_files = [f for f in os.listdir(persona_dir) if f.endswith(".json")]
    for persona_file in persona_files:
        with open(os.path.join(persona_dir, persona_file), "r", encoding="utf-8") as f:
            persona_data = json.load(f)

        persona_ids = persona_ids or [f"P{i}" for i in range(1, 54)]
        if persona_data['persona_id'] not in persona_ids:
            continue

        if not os.path.exists(os.path.join(results_dir, persona_data['persona_id'])):
            os.makedirs(os.path.join(results_dir, persona_data['persona_id']))

        for i in range(1, 4):
            for task in task_data:
                try:
                    system_content = construct_system_prompt(persona_data["persona_traits"], persona_data["persona_id"])
                    user_content = construct_user_prompt(task)
                    response = ask_deepseek(system_content, user_content)
                    while response is None:
                        print("Retrying...")
                        response = ask_deepseek(system_content, user_content)

                    with open(os.path.join(results_dir, persona_data['persona_id'], f"{persona_data['persona_id']}_{task_name}_{i}.json"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "task": task,
                            "response": response,
                        }, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"Error processing task {task}: {e}")
                    continue


if __name__ == "__main__":
    persona_codes = ["DAUH", "DSUH", "DSUL", "VAPH", "VAPL", "VAUH", "VSPH", "VSPL", "VSUH", "VSUL"]
    max_parallel = 8
    task = "knowledge_task"  # Change the task type as needed.

    def run_one_persona_code(persona_code):
        print(f"[START] {persona_code}")
        main(
            persona_dir=f"../persona/user_study_personas/{persona_code}",
            results_dir=f"../results/baseline_one_sentence/{task}/{persona_code}",
            task_dir="../task",
            task_name=task
        )
        print(f"[DONE] {persona_code}")

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_one_persona_code, code): code for code in persona_codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] {code}: {e}")
