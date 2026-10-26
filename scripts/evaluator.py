import json
import argparse
import os
import glob
import sys
import re
import signal
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import importlib.util
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
import pandas as pd
import httpx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from config import DEEPSEEK_API_KEY, GLM_API_KEY, GPT_API_KEY

QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or getattr(config, "QWEN_API_KEY", "")
)
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL") or getattr(
    config, "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.getenv("QWEN_MODEL") or getattr(config, "QWEN_MODEL", "Qwen3.8-Flash")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or getattr(config, "CLAUDE_API_KEY", "")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL") or getattr(config, "CLAUDE_BASE_URL", "")
CLAUDE_MODEL = "claude-sonnet-5"
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY") or getattr(config, "MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL") or getattr(
    config, "MINIMAX_BASE_URL", "https://api.minimax.io/v1"
)
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL") or getattr(config, "MINIMAX_MODEL", "MiniMax-M2.7")


def resolve_evaluation_model(model):
    if model in ("qwen", QWEN_MODEL):
        return QWEN_MODEL
    if model in ("claude", CLAUDE_MODEL):
        return CLAUDE_MODEL
    if model in ("minimax", "minimax2.7", "MiniMax2.7", "MiniMax-M2.7", MINIMAX_MODEL):
        return MINIMAX_MODEL
    raise ValueError(f"Unsupported evaluation model: {model}")

TRAIT_DESCRIPTIONS = {
    "five_person_traits": {
        "openness": "The tendency to be curious, imaginative, and open to new apps and services.",
        "conscientiousness": "Tendency to regularly check account security settings and manage passwords systematically",
        "extraversion": "Tendency to share personal information, experiences, and location openly on social platforms",
        "agreeableness": "Tendency to comply with data collection requests or grant permissions to avoid friction or conflict",
        "neuroticism": "Tendency to overreact to security alerts or, conversely, to avoid engaging with them due to anxiety",
    },
    "attitudes_and_perceptions": {
        "privacy_concern": "Overall level of concern regarding privacy and security issues",
        "data_concerns": "The degree of concern about the collection, storage, or misuse of their personal data",
        "online_trust": "Trust in websites, including the reliability of their information and their handling of personal data",
        "home_network_trust": "The degree of trust in home Wi-Fi/network security",
        "data_collecting_entity_trust": "The degree of trust in entities that collect personal data (e.g., apps, platforms, and service providers)",
        "threat_source": "Perception of whether security threats originate locally or remotely",
        "view_of_others_awareness": "Perceptions of others' security awareness",
        "privacy_protection_actions": "Whether users believe that privacy protection measures or actions are necessary",
        "attitudes_toward_being_monitored": "Concern about being monitored or tracked, especially regarding the collection and tracking of personal data",
        "security_vs_convenience": "Preference for security versus convenience, affecting whether you prioritize safety or ease of use in decisions involving personal data",
        "social_vs_privacy": "Trade-off between social needs and privacy protection",
    },
    "knowledge_and_learning": {
        "knowledge_of_security_privacy_concepts": "Security/privacy knowledge breadth x depth",
        "sources_of_security_knowledge": "Primary channels through which security knowledge is acquired",
        "types_of_perceived_threats": "Types of threats the user consciously recognizes as personally relevant",
    },
    "behaviors_and_practices": {
        "daily_online_hours": "Average daily online time across all devices (hours)",
        "social_networking_use": "Frequency and depth of social networking engagement",
        "ability_to_help_others": "Demonstrated capacity to diagnose and resolve security/privacy issues for others",
        "behavior_changes": "Magnitude of behavioral change following a security incident or warning",
        "initiative_on_reviewing_policies": "Self-initiated frequency of reading privacy policies or ToS before consent",
        "proactive_or_reactive": "Security posture orientation: proactive (preemptive hardening, hygiene routines) vs. reactive (responds only after incidents or external prompts)",
        "tendency_to_adjust_website_settings": "Tendency to adjust default website/browser privacy settings",
        "password_layer": "Diversity of password security levels (weak/strong/multi-level across account sensitivity tiers)",
        "tendency_to_censor_information": "Tendency to selectively omit, obfuscate, or withhold personal data",
        "completeness_of_fallback_authentication": "Coverage and robustness of account recovery mechanisms",
        "system_protection": "Use of system protection tools",
        "number_of_passwords": "Count of distinct passwords actively in use across accounts",
        "recording_passwords": "Method and security of password storage",
    },
}

SCALE_ANCHOR = """
## Scale Definitions (3-point, applied per field category)

### Trait, Behavioral & Attitudinal Fields
(applies to: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, Initiative to Review Policies, Tendency to Adjust Settings, Tendency to Censor Information, Privacy Concern, Data Concern, Online Trust, Home Network Trust, Data-collecting Entities Trust)
- low: Largely absent or disengaged; does not manifest consistently in observable behavior or stated perceptions.
- medium: Present but situational; expression depends on context or is readily overridden by competing factors.
- high: Dominant and stable; reliably shapes perceptions, decisions, and behaviors across contexts.

### Knowledge / Competence Fields
(applies to: Knowledge Indicators, Capacity to Help Others)
- low: Aware of security/privacy as a concept but cannot correctly apply or explain any specific mechanism.
- medium: Functional literacy; understands common threats and basic countermeasures but has significant blind spots.
- high: Demonstrates transferable, cross-domain understanding; can reason about novel threats and advise others accurately.

### Frequency / Quantity Fields
(applies to: Social Networking Use)
- low: Minimal or infrequent; below average for the user's demographic context.
- medium: Moderate; roughly consistent with typical usage patterns for the user's demographic.
- high: Heavy or above-average; meaningfully exceeds typical patterns for the user's demographic.

### Enumerated Fields (value is self-descriptive, no scale applies)
(applies to: Source of Threats, View of Others' Awareness, Security vs Convenience, Social vs Privacy, Daily Online Time, Number of Passwords, Password Layer, Recording Passwords, Proactive or Reactive)
Values for these fields are fixed categorical labels whose meaning is explicit in the label itself. Interpret them literally without mapping to the low/medium/high scale above.
""".strip()

FOUR_DIMENSION_METADATA = {
    "V": {
        "name": "Privacy Vigilance–Delegation Stance ",
        "high_code": "V",
        "low_code": "D",
        "high_text": "High level of concern, tendency to be vigilant.",
        "low_text": "Low level of concern, tendency to trust.",
    },
    "C": {
        "name": "Control Orientation",
        "high_code": "A",
        "low_code": "S",
        "high_text": "Actively taking control and being willing to pay the cognitive and behavioral costs for privacy.",
        "low_text": "Passive control, unwilling to actively manage/configure personal privacy.",
    },
    "U": {
        "name": "Privacy–Utility Tradeoff",
        "high_code": "P",
        "low_code": "U",
        "high_text": "More likely to prioritize privacy over social or convenience benefits.",
        "low_text": "More willing to trade privacy for convenience, utility, or social participation.",
    },
    "L": {
        "name": "Privacy Literacy",
        "high_code": "H",
        "low_code": "L",
        "high_text": "Higher security/privacy literacy and stronger ability to reason about threats.",
        "low_text": "Lower security/privacy literacy and less confident reasoning about threats.",
    },
}


@lru_cache(maxsize=1)
def load_dimension_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "persona_generator"
        / "4-dimensional_personality.py"
    )
    spec = importlib.util.spec_from_file_location("privacy_dimension_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load dimension module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_four_dimension_profile(persona_traits):
    try:
        dimension_module = load_dimension_module()
        scores = dimension_module.compute_dimension_scores({"persona_traits": persona_traits})
    except Exception:
        return None

    profile_parts = []
    persona_code = []

    for dim in ["V", "C", "U", "L"]:
        score = scores.get(dim)
        if score is None:
            continue

        metadata = FOUR_DIMENSION_METADATA[dim]
        is_high = score >= 0.5
        code = metadata["high_code"] if is_high else metadata["low_code"]
        explanation = metadata["high_text"] if is_high else metadata["low_text"]
        persona_code.append(code)
        profile_parts.append(
            f"- [{dim} | {metadata['name']}] score={score:.2f}, pole={code}: {explanation}"
        )

    if not profile_parts:
        return None

    return (
        "### Four-Dimension Summary:\n"
        f"- [Persona Code] {'-'.join(persona_code)}\n"
        + "\n".join(profile_parts)
        + "\n"
    )

KEY_TRAIT_SPECS = [
    ("Privacy Concern", "Overall level of concern regarding privacy and security issues", ("attitudes_and_perceptions", "privacy_concern")),
    ("Data Concern", "The degree of concern about the collection, storage, or misuse of their personal data", ("attitudes_and_perceptions", "data_concerns")),
    ("Source of Threats", "Perception of whether security threats originate locally or remotely", ("attitudes_and_perceptions", "threat_source")),
    ("Security vs Convenience", "Preference for security versus convenience, affecting whether you prioritize safety or ease of use in decisions involving personal data", ("attitudes_and_perceptions", "security_vs_convenience")),
    ("Social vs Privacy", "Trade-off between social needs and privacy protection", ("attitudes_and_perceptions", "social_vs_privacy")),
    ("Proactive or Reactive", "Security posture orientation: proactive (preemptive hardening, hygiene routines) vs. reactive (responds only after incidents or external prompts)", ("behaviors_and_practices", "proactive_or_reactive")),
    ("Knowledge Indicators", "Security/privacy knowledge breadth x depth", ("knowledge_and_learning", "knowledge_of_security_privacy_concepts")),
]

def retry_on_timeout(max_retries=3, base_delay=2):
    """Retry timeout, connection, and rate-limit errors with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    is_network_error = (
                        isinstance(e, (httpx.TimeoutException, APITimeoutError,
                                       httpx.ConnectError, APIConnectionError,
                                       ConnectionError, RateLimitError))
                        or "timeout" in str(e).lower()
                        or "timed out" in str(e).lower()
                        or "connection" in str(e).lower()
                    )
                    if is_network_error and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"[NETWORK ERROR] {func.__name__} attempt {attempt + 1}/{max_retries} failed, "
                              f"retrying in {delay}s... (error: {e})")
                        time.sleep(delay)
                        last_exception = e
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


def get_trait_description(trait_name: str) -> str | None:
    if not trait_name:
        return None

    normalized = trait_name.lower().replace(" ", "_")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
    for group in TRAIT_DESCRIPTIONS.values():
        for key, desc in group.items():
            candidate = key.lower().replace(" ", "_")
            candidate = "".join(ch for ch in candidate if ch.isalnum() or ch == "_")
            if candidate == normalized:
                return desc
    return None


def build_persona_description(persona_traits):
    return (
        f"### Five Person Traits (3-point scale):\n"
        f"- [Openness] {TRAIT_DESCRIPTIONS['five_person_traits']['openness']}: {persona_traits['five_person_traits']['openness']}\n"
        f"- [Conscientiousness] {TRAIT_DESCRIPTIONS['five_person_traits']['conscientiousness']}: {persona_traits['five_person_traits']['conscientiousness']}\n"
        f"- [Extraversion] {TRAIT_DESCRIPTIONS['five_person_traits']['extraversion']}: {persona_traits['five_person_traits']['extraversion']}\n"
        f"- [Agreeableness] {TRAIT_DESCRIPTIONS['five_person_traits']['agreeableness']}: {persona_traits['five_person_traits']['agreeableness']}\n"
        f"- [Neuroticism] {TRAIT_DESCRIPTIONS['five_person_traits']['neuroticism']}: {persona_traits['five_person_traits']['neuroticism']}\n\n"
        f"### Attitudes and Perceptions (3-point scale):\n"
        f"- [Privacy Concern] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['privacy_concern']}: {persona_traits['attitudes_and_perceptions']['privacy_concern']}\n"
        f"- [Data Concern] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['data_concerns']}: {persona_traits['attitudes_and_perceptions']['data_concerns']}\n"
        f"- [Online Trust] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['online_trust']}: {persona_traits['attitudes_and_perceptions']['online_trust']}\n"
        f"- [Home Network Trust] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['home_network_trust']}: {persona_traits['attitudes_and_perceptions']['home_network_trust']}\n"
        f"- [Data-collecting Entities Trust] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['data_collecting_entity_trust']}: {persona_traits['attitudes_and_perceptions']['data_collecting_entity_trust']}\n"
        f"- [Source of Threats] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['threat_source']}: {persona_traits['attitudes_and_perceptions']['threat_source']}\n"
        f"- [View of Others' Awareness] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['view_of_others_awareness']}: {persona_traits['attitudes_and_perceptions']['view_of_others_awareness']}\n"
        f"- [Privacy Protection Actions] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['privacy_protection_actions']}: {persona_traits['attitudes_and_perceptions']['privacy_protection_actions']}\n"
        f"- [Attitudes toward Being Monitored] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['attitudes_toward_being_monitored']}: {persona_traits['attitudes_and_perceptions']['attitudes_toward_being_monitored']}\n"
        f"- [Security vs Convenience] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['security_vs_convenience']}: {persona_traits['attitudes_and_perceptions']['security_vs_convenience']}\n"
        f"- [Social vs Privacy] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['social_vs_privacy']}: {persona_traits['attitudes_and_perceptions']['social_vs_privacy']}\n\n"
        f"### Knowledge and Learning:\n"
        f"- [Knowledge Indicators] {TRAIT_DESCRIPTIONS['knowledge_and_learning']['knowledge_of_security_privacy_concepts']}: {persona_traits['knowledge_and_learning']['knowledge_of_security_privacy_concepts']}\n"
        f"- [Learning Source] {TRAIT_DESCRIPTIONS['knowledge_and_learning']['sources_of_security_knowledge']}: {persona_traits['knowledge_and_learning']['sources_of_security_knowledge']}\n"
        f"- [Types of Attacks] {TRAIT_DESCRIPTIONS['knowledge_and_learning']['types_of_perceived_threats']}: {persona_traits['knowledge_and_learning']['types_of_perceived_threats']}\n\n"
        f"### Behaviors and Practices:\n"
        f"- [Daily Online Time] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['daily_online_hours']}: {persona_traits['behaviors_and_practices']['daily_online_hours']}\n"
        f"- [Social Networking Use] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['social_networking_use']}: {persona_traits['behaviors_and_practices']['social_networking_use']}\n"
        f"- [Capacity to Help Others] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['ability_to_help_others']}: {persona_traits['behaviors_and_practices']['ability_to_help_others']}\n"
        f"- [Behavior Changes] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['behavior_changes']}: {persona_traits['behaviors_and_practices']['behavior_changes']}\n"
        f"- [Initiative to Review Policies] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['initiative_on_reviewing_policies']}: {persona_traits['behaviors_and_practices']['initiative_on_reviewing_policies']}\n"
        f"- [Proactive or Reactive] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['proactive_or_reactive']}: {persona_traits['behaviors_and_practices']['proactive_or_reactive']}\n"
        f"- [Tendency to Adjust Website Settings] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['tendency_to_adjust_website_settings']}: {persona_traits['behaviors_and_practices']['tendency_to_adjust_website_settings']}\n"
        f"- [Password Layer] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['password_layer']}: {persona_traits['behaviors_and_practices']['password_layer']}\n"
        f"- [Tendency to Censor Information] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['tendency_to_censor_information']}: {persona_traits['behaviors_and_practices']['tendency_to_censor_information']}\n"
        f"- [Completeness of Fallback Authentication] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['completeness_of_fallback_authentication']}: {persona_traits['behaviors_and_practices']['completeness_of_fallback_authentication']}\n"
        f"- [System Protection Tools] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['system_protection']}: {persona_traits['behaviors_and_practices']['system_protection']}\n"
        f"- [Number of Passwords] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['number_of_passwords']}: {persona_traits['behaviors_and_practices']['number_of_passwords']}\n"
        f"- [Recording Passwords] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['recording_passwords']}: {persona_traits['behaviors_and_practices']['recording_passwords']}\n"
    )


def build_key_traits(persona_traits):
    lines = []
    for label, desc, (group, key) in KEY_TRAIT_SPECS:
        value = persona_traits.get(group, {}).get(key, "unknown")
        lines.append(f"- [{label}] {desc}: {value}")
    return "\n".join(lines)


def build_relevant_traits_section(relevant_traits):
    if not relevant_traits:
        return "No task-specific relevant traits were provided. Infer the likely influential traits from the persona and task, but keep the judgment conservative."

    lines = ["The most relevant persona traits for this task:"]
    for trait, value in relevant_traits.items():
        desc = get_trait_description(trait)
        if desc:
            lines.append(f"- {trait} ({desc}): {value}")
        else:
            lines.append(f"- {trait}: {value}")
    return "\n".join(lines)


@retry_on_timeout(max_retries=5, base_delay=2)
def ask_qwen(system_content, user_content=None):
    """Return a validated evaluation, retrying network and malformed JSON responses."""
    if not QWEN_API_KEY:
        raise ValueError("Set QWEN_API_KEY in config.py or QWEN_API_KEY/DASHSCOPE_API_KEY in the environment.")
    return _ask_evaluation_json(
        system_content, user_content, api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL, model=QWEN_MODEL,
        extra_body={"enable_thinking": False},
    )


@retry_on_timeout(max_retries=5, base_delay=2)
def ask_claude(system_content, user_content=None):
    """Call claude-sonnet-5 through the configured OpenAI-compatible gateway."""
    if not CLAUDE_API_KEY or not CLAUDE_BASE_URL:
        raise ValueError("Set CLAUDE_API_KEY and CLAUDE_BASE_URL in config.py or the environment.")
    return _ask_evaluation_json(
        system_content, user_content, api_key=CLAUDE_API_KEY,
        base_url=CLAUDE_BASE_URL, model=CLAUDE_MODEL,
    )


@retry_on_timeout(max_retries=5, base_delay=2)
def ask_minimax(system_content, user_content=None):
    """Call MiniMax M2.7 through its OpenAI-compatible API."""
    if not MINIMAX_API_KEY:
        raise ValueError("Set MINIMAX_API_KEY in config.py or the environment.")
    return _ask_evaluation_json(
        system_content, user_content, api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL, model=MINIMAX_MODEL,
    )


def _ask_evaluation_json(system_content, user_content, *, api_key, base_url, model, extra_body=None):
    """Parse and validate evaluations, retrying malformed responses once."""
    if user_content is None:
        user_content, system_content = system_content, ""
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    with OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(120.0, connect=30.0),
        max_retries=0,
    ) as client:
        request = dict(model=model, messages=messages, temperature=0, stream=False)
        if extra_body is not None:
            request["extra_body"] = extra_body
        for attempt in range(2):
            response = client.chat.completions.create(**request)
            response_text = repair_json_response(response.choices[0].message.content)
            try:
                parsed = json.loads(response_text) if response_text else None
            except (TypeError, ValueError):
                parsed = None
            if is_valid_evaluation(parsed):
                # Record the requested model locally, overriding any model-generated label.
                parsed["model"] = model
                print(json.dumps(parsed, ensure_ascii=False))
                return parsed
            print(f"[INVALID EVALUATION] {model} attempt {attempt + 1}/2")
    return None


def is_valid_evaluation(value):
    return (
        isinstance(value, dict)
        and type(value.get("consistency_score")) is int
        and value["consistency_score"] in (1, 2, 3)
        and isinstance(value.get("justification"), str)
        and bool(value["justification"].strip())
        and value["justification"] != "Failed to get a valid response from the model."
    )


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


@retry_on_timeout(max_retries=5, base_delay=2)
def ask_deepseek(system_content, user_content):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=httpx.Timeout(120.0, connect=30.0),
        max_retries=0,
    )

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
            print(response_text)
            raise ValueError("Empty JSON response")
        parsed = json.loads(response_text)
    except Exception:
        print(raw_content)
        print("Error decoding JSON response")
        return raw_content
    print(response_text)
    return parsed




@retry_on_timeout(max_retries=5, base_delay=2)
def ask_gpt(system_content, user_content):
    """Call the GPT-compatible gateway and return the model's JSON response."""
    client = OpenAI(
        api_key=GPT_API_KEY,
        base_url="https://sub.fancybag.cn/v1",
        timeout=httpx.Timeout(120.0, connect=30.0),
        max_retries=0,
    )

    request_kwargs = {
        "model": "gpt-5.6-terra",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "stream": False,
        # The installed OpenAI SDK does not expose this as a named argument.
        "extra_body": {"reasoning_effort": "none"},
    }

    raw_content = None
    for attempt in range(2):
        response = client.chat.completions.create(**request_kwargs)
        raw_content = response.choices[0].message.content
        response_text = repair_json_response(raw_content)
        try:
            if response_text is None:
                raise ValueError("Empty JSON response")
            parsed = json.loads(response_text)
            print(response_text)
            return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            if attempt == 0:
                continue

    print(raw_content)
    print("Error decoding JSON response")
    return None


def evaluator(PERSONA, task, response, relevant_traits=None, model="qwen"):
    selected_model = resolve_evaluation_model(model)
    if isinstance(PERSONA, dict):
        persona_block = build_persona_description(PERSONA)
        key_traits_block = build_key_traits(PERSONA)
        four_dimension_profile = build_four_dimension_profile(PERSONA)
        four_dimension_block = ""
        if four_dimension_profile:
            four_dimension_block = (
                "[FOUR_DIMENSION_PROFILE]\n"
                f"{four_dimension_profile}"
                "The four-dimension summary is a global anchor for the persona. Use it to keep the persona's overall orientation stable, but do not let it override concrete raw traits when a task clearly depends on them.\n"
                "[/FOUR_DIMENSION_PROFILE]\n\n"
            )
    else:
        persona_block = str(PERSONA)
        key_traits_block = (
            "Key traits were not provided in structured form. Use the persona description above and prioritize Privacy Concern, "
            "Data Concern, Source of Threats, Security vs Convenience, Social vs Privacy, Proactive or Reactive, and Knowledge "
            "Indicators when they appear."
        )
        four_dimension_block = ""

    relevant_traits_block = build_relevant_traits_section(relevant_traits)
    task_block = json.dumps(task, ensure_ascii=False, indent=2) if not isinstance(task, str) else task
    response_block = json.dumps(response, ensure_ascii=False, indent=2) if not isinstance(response, str) else response

    content = (
        "You are an evaluator that judges whether a model's response aligns with a given user persona.\n\n"

        f"{four_dimension_block}"
        "[PERSONA]\n"
        f"{SCALE_ANCHOR}\n\n"
        f"{persona_block}\n"
        "[/PERSONA]\n"
        
        "[KEY TRAITS]\n"
        f"{key_traits_block}\n"
        "[/KEY TRAITS]\n"
        
        "[RELEVANT TRAITS]\n"
        f"{relevant_traits_block}\n"
        "[/RELEVANT TRAITS]\n"
        
        "[TASK]\n"
        f"{task_block}\n"
        "[/TASK]\n"
        
        "[MODEL RESPONSE]\n"
        f"{response_block}\n"
        "[/MODEL RESPONSE]\n"
        
        "[SCORING RUBRIC (1-3)]\n"
        "1 = clearly inconsistent: the response is unlikely for this persona and strongly contradicts the persona's core tendencies, relevant traits, or likely behavior.\n"
        "2 = plausible / generally aligned: the response is a reasonable answer that this persona could choose, even if it is not the most prototypical or best-matching answer.\n"
        "3 = strongly consistent: the response is not only plausible, but also a particularly natural and well-matched choice for this persona.\n"
        "[/SCORING RUBRIC]\n"
        
        "[NOTES]\n"
        "- Use a permissive scoring standard. Judge whether the answer is plausibly consistent with the persona, not whether it is the single best choice. Assign at least 2 to any reasonably plausible answer, and reserve 1 only for clear contradictions or implausible choices. When uncertain, prefer 2 over 1 or 3.\n"
        "- First anchor yourself on the four-dimension summary to understand the persona's overall direction, then use the raw traits to decide how that direction shows up in this specific task.\n"
        "- Prioritize decisions based on key traits and relevant traits, but do not ignore overall personality traits.\n"
        "- If the reply option is 'technically necessary,' then the default is 'consistent with persona.'\n"
        "[/NOTES]\n"
        
        "[OUTPUT FORMAT]\n"
        "Return your evaluation in JSON format:\n"
        "{\n"
        "  \"consistency_score\": 1,\n"
        "  \"justification\": \"...\"\n"
        "}"
        "[/OUTPUT FORMAT]\n"
    )

    if selected_model == CLAUDE_MODEL:
        ask_model = ask_claude
    elif selected_model == MINIMAX_MODEL:
        ask_model = ask_minimax
    else:
        ask_model = ask_qwen
    return ask_model("", content)


def load_json_rows(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json_rows(file_path, rows):
    file_path = os.fspath(file_path)
    directory = os.path.dirname(os.path.abspath(file_path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=directory,
            prefix=f".{os.path.basename(file_path)}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def get_persona_traits_for_result(persona_root, results_root, result_file):
    rel_path = os.path.relpath(result_file, results_root)
    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"Unexpected result path structure: {result_file}")

    persona_code = parts[-3]
    persona_id = parts[-2]
    persona_path = os.path.join(persona_root, persona_code, f"{persona_id}.json")
    with open(persona_path, "r", encoding="utf-8") as f:
        persona_data = json.load(f)
    return persona_data["persona_traits"], persona_data["persona_id"]


def needs_evaluation(evaluation, mode, selected_model):
    """Resume skips only valid evaluations already saved for the target model."""
    if mode not in ("fill", "update", "resume"):
        raise ValueError("mode must be 'fill', 'update', or 'resume'")
    if mode == "update" or not is_valid_evaluation(evaluation):
        return True
    return mode == "resume" and evaluation.get("model") != selected_model


def reevaluate_result_file(result_file, results_root, persona_root, mode="fill", model="qwen", stop_event=None):
    selected_model = resolve_evaluation_model(model)
    if mode not in ("fill", "update", "resume"):
        raise ValueError("mode must be 'fill', 'update', or 'resume'")
    persona_traits, persona_id = get_persona_traits_for_result(
        persona_root,
        results_root,
        result_file,
    )
    rows = load_json_rows(result_file)
    skipped_existing = 0
    filled_missing = 0
    skipped_failed = 0

    try:
        for idx, row in enumerate(rows, start=1):
            if stop_event is not None and stop_event.is_set():
                print(f"[STOP] {result_file}: stopping before line {idx}; saving completed evaluations")
                break

            if not needs_evaluation(row.get("evaluation"), mode, selected_model):
                skipped_existing += 1
                continue

            task = row.get("task")
            response = row.get("response")
            relevant_traits = row.get("relevant_traits")

            if task is None or response is None:
                print(f"[SKIP] {result_file} line {idx}: missing task or response")
                continue

            evaluation = evaluator(persona_traits, task, response, relevant_traits, model)
            if evaluation is None and not (stop_event is not None and stop_event.is_set()):
                print(f"[RETRY] {persona_id} {os.path.basename(result_file)} line {idx}")
                evaluation = evaluator(persona_traits, task, response, relevant_traits, model)

            if not is_valid_evaluation(evaluation):
                print(f"[FAILED] {result_file} line {idx}: leaving evaluation unchanged")
                skipped_failed += 1
                continue

            row["evaluation"] = evaluation
            filled_missing += 1
    except BaseException:
        if filled_missing:
            try:
                write_json_rows(result_file, rows)
                print(f"[CHECKPOINT] {result_file}: saved {filled_missing} evaluations before error/interruption")
            except BaseException as checkpoint_error:
                print(f"[CHECKPOINT FAILED] {result_file}: {checkpoint_error}")
        raise

    if filled_missing:
        write_json_rows(result_file, rows)
        print(f"[UPDATED] {result_file} (updated {filled_missing}, skipped existing {skipped_existing}, failed {skipped_failed}, total {len(rows)} rows)")
    else:
        print(
            f"[SKIP] {result_file} "
            f"(valid evaluations {skipped_existing}, failed {skipped_failed}, total {len(rows)} rows)"
        )


def reevaluate_results_directory(results_root=None, persona_root=None, persona_code=None, persona_id=None, dry_run=False, workers=4, mode="fill", model="qwen"):
    """Fill, update, or resume evaluations for a selected model."""
    if mode not in ("fill", "update", "resume"):
        raise ValueError("mode must be 'fill', 'update', or 'resume'")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    selected_model = resolve_evaluation_model(model)
    base_dir = os.path.dirname(__file__)
    if results_root is None:
        results_root = os.path.abspath(
            os.path.join(base_dir, "..", "results_glm")
        )
    if persona_root is None:
        persona_root = os.path.abspath(
            os.path.join(base_dir, "..", "persona", "synthetic_users_v6")
        )

    code_pattern = persona_code if persona_code else "*"
    persona_pattern = persona_id if persona_id else "P*"
    pattern = os.path.join(results_root, "**", code_pattern, persona_pattern, "*.json")
    result_files = sorted(glob.glob(pattern, recursive=True))
    print(f"[SCAN] found {len(result_files)} json files under {results_root}")
    print(f"[MODEL] {selected_model}; persona root: {persona_root}")
    mode_description = {
        "fill": "fill missing or invalid evaluations",
        "update": "replace existing evaluations on success",
        "resume": "skip valid evaluations from the selected model; replace others on success",
    }
    print(f"[MODE] {mode}: {mode_description[mode]}")

    if dry_run:
        pending = 0
        for result_file in result_files:
            get_persona_traits_for_result(persona_root, results_root, result_file)
            pending += sum(
                needs_evaluation(row.get("evaluation"), mode, selected_model)
                and row.get("task") is not None and row.get("response") is not None
                for row in load_json_rows(result_file)
            )
        print(f"[DRY RUN] {pending} rows need evaluation; no API calls or file writes")
        return

    print(f"[WORKERS] {workers}")
    failed_files = []
    stop_event = threading.Event()

    def request_graceful_stop(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_sigterm_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_graceful_stop)
    # Each file has one owner; rows within a file are evaluated sequentially.
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    reevaluate_result_file,
                    result_file,
                    results_root,
                    persona_root,
                    mode=mode,
                    model=model,
                    stop_event=stop_event,
                ): result_file
                for result_file in result_files
            }
            try:
                for completed, future in enumerate(as_completed(futures), start=1):
                    result_file = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failed_files.append(result_file)
                        print(f"[FILE FAILED] {result_file}: {exc}")
                    print(f"[PROGRESS] {completed}/{len(result_files)} files; {len(failed_files)} file errors")
            except BaseException:
                stop_event.set()
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                print("[STOP] stopping workers; active files are saving completed evaluations")
                raise
    finally:
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
    if failed_files:
        raise RuntimeError(f"{len(failed_files)} files failed; see [FILE FAILED] messages and rerun to retry.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill, update, or resume evaluations using Qwen, Claude, or MiniMax.")
    parser.add_argument("persona_code", nargs="?")
    parser.add_argument("persona_id", nargs="?")
    parser.add_argument("--results-root")
    parser.add_argument("--persona-root")
    parser.add_argument("--model", choices=list(dict.fromkeys([
                            "qwen", QWEN_MODEL, "claude", CLAUDE_MODEL,
                            "minimax", "minimax2.7", "MiniMax2.7", "MiniMax-M2.7", MINIMAX_MODEL,
                        ])),
                        default="qwen", help="Evaluation model (default: qwen). Use --mode update to replace valid scores from another model.")
    parser.add_argument("--mode", choices=("fill", "update", "resume"), default="fill",
                        help="fill: skip valid evaluations (default); update: reevaluate all eligible rows; resume: skip valid evaluations from the selected model. Old evaluations are preserved on failure.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and count pending rows without calling the API.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent result files (default: 4; use 1 for sequential processing).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    reevaluate_results_directory(**vars(args))
