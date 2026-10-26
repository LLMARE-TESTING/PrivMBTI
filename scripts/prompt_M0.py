import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json
import re
from pathlib import Path

sys.path.insert(0, r"d:\\privacy\\privacy+persona\scripts")
from config import DEEPSEEK_API_KEY


def construct_system_prompt():
    """
    Baseline system prompt — no persona information whatsoever.
    The LLM answers the task directly as a neutral assistant.
    """
    system_content = (
        "Answer the following questions with an objective attitude.\n\n"
        "[OUTPUT FORMAT]\n"
        "When performing tasks, please strictly follow the output format requirements below. Do not add any additional explanations or content outside the specified format.\n"
        "Output format: Only output a single JSON object with the following structure:\n"
        "{\n"
        '  "task_output": "your answer to the task",\n'
        '  "reason": "brief explanation of the reasoning behind your answer"\n'
        "}\n"
        "[/OUTPUT FORMAT]\n"
    )
    return system_content


def construct_user_prompt(task):
    """
    Baseline user prompt — only the task description, no persona traits.
    """
    user_content = f"Task Description:\n{json.dumps(task, indent=2)}"
    return user_content


def repair_json_response(raw_content):
    """Repair malformed JSON from LLM output."""
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
    """Balance JSON braces/brackets and close unterminated strings."""
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
    """
    Call the DeepSeek API, retry once on JSON parse failure.
    Returns the parsed JSON object, or None on failure.
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw_content = response.choices[0].message.content
    response_text = repair_json_response(raw_content)
    try:
        if response_text is None:
            raise ValueError("Empty JSON response")
        parsed = json.loads(response_text)
    except Exception:
        # Retry once
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
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


def main(results_dir, task_dir, task_name):
    """
    Run baseline evaluation: for each task, call LLM without any persona
    information. Results serve as a baseline for comparison with persona-
    conditioned responses.

    Parameters
    ----------
    results_dir : str
        Path (relative to this script) where baseline results will be saved.
    task_dir : str
        Path (relative to this script) to the directory containing task JSON files.
    task_name : str
        Name of the task file (without .json extension).
        Must be one of: assessment_task_v2, contextual_task_v3, setting_task, knowledge_task
    """
    # Resolve paths relative to this script
    script_dir = os.path.dirname(__file__)
    task_dir = os.path.join(script_dir, task_dir)
    results_dir = os.path.join(script_dir, results_dir)

    allowed_tasks = [
        "assessment_task_v2",
        "contextual_task_v3",
        "setting_task",
        "knowledge_task",
    ]
    if task_name not in allowed_tasks:
        raise ValueError(
            f"Invalid task_name '{task_name}'. Must be one of: {', '.join(allowed_tasks)}"
        )

    # Create results directory
    baseline_dir = os.path.join(results_dir, "baseline_none_persona")
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # Load tasks
    task_path = os.path.join(task_dir, f"{task_name}.json")
    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    # Build prompts once (no persona dependency, so system prompt is constant)
    system_content = construct_system_prompt()

    # Run each task 3 times for consistency with persona-based evaluation
    for task in task_data:
        for i in range(1, 4):
            try:
                user_content = construct_user_prompt(task)
                response = ask_deepseek(system_content, user_content)
                while response is None:
                    print("Retrying...")
                    response = ask_deepseek(system_content, user_content)

                # Save result as JSON line
                result_path = os.path.join(
                    baseline_dir, f"baseline_{task_name}_{i}.json"
                )
                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "task": task,
                                "response": response,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            except Exception as e:
                print(f"Error processing task {task}: {e}")
                continue


if __name__ == "__main__":
    # Baseline runs are persona-independent, so we only parallelize across
    # different task types rather than different personas.
    tasks = [
        "knowledge_task"
    ]
    max_parallel = min(len(tasks), 4)

    def run_one_task(task_name):
        print(f"[START] baseline - {task_name}")
        main(
            results_dir="../results",
            task_dir="../task",
            task_name=task_name,
        )
        print(f"[DONE] baseline - {task_name}")

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_one_task, task): task for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] baseline - {task}: {e}")
