import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import importlib.util
from openai import OpenAI
import json
import pandas as pd
from pathlib import Path
import re
from evaluator import evaluator
from zai import ZhipuAiClient
sys.path.insert(0, r"d:\\privacy\\privacy+persona\scripts")
from config import DEEPSEEK_API_KEY, GLM_API_KEY


# =============================================================================
# Trait descriptions, hints, and metadata (same as prompt.py)
# =============================================================================

# Store each trait description centrally; the key is the trait name and the value is its description.
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

KEY_DECISION_TRAITS = [
  "privacy_concern",
  "data_concerns",
  "threat_source",
  "security_vs_convenience",
  "social_vs_privacy",
  "proactive_or_reactive",
  "knowledge_of_security_privacy_concepts",
]

TRAIT_PRIORITY = {
  "privacy_concern": 12,
  "data_concerns": 11,
  "security_vs_convenience": 10,
  "social_vs_privacy": 10,
  "proactive_or_reactive": 10,
  "knowledge_of_security_privacy_concepts": 10,
  "threat_source": 9,
  "attitudes_toward_being_monitored": 9,
  "privacy_protection_actions": 9,
  "online_trust": 8,
  "data_collecting_entity_trust": 8,
  "home_network_trust": 8,
  "tendency_to_censor_information": 8,
  "tendency_to_adjust_website_settings": 8,
  "initiative_on_reviewing_policies": 8,
  "password_layer": 8,
  "recording_passwords": 8,
  "completeness_of_fallback_authentication": 8,
  "system_protection": 7,
  "behavior_changes": 7,
  "types_of_perceived_threats": 7,
  "sources_of_security_knowledge": 6,
  "social_networking_use": 6,
  "daily_online_hours": 5,
  "ability_to_help_others": 5,
  "number_of_passwords": 5,
  "conscientiousness": 5,
  "extraversion": 4,
  "agreeableness": 3,
  "openness": 3,
  "neuroticism": 3,
  "view_of_others_awareness": 2,
}

TASK_TRAIT_HINTS = {
  "privacy_concern": [
    "privacy", "personal information", "personal data", "sensitive data",
    "data sharing", "share data", "disclose", "expose", "exposure",
  ],
  "data_concerns": [
    "collect", "collection", "store", "storage", "retain", "retention", "third party", "third-party", "data use", "misuse", "reuse",
  ],
  "online_trust": [
    "website", "site", "online service", "web service"
  ],
  "home_network_trust": [
    "wifi", "wi-fi", "router", "home network", "public network", "network security",
  ],
  "data_collecting_entity_trust": [
    "company", "provider", "service provider", "developer", "platform", "vendor", "manufacturer",
  ],
  "threat_source": [
    "threat", "attack", "attacker", "hacker", "malware", "phishing", "scam", "remote", "local",
  ],
  "view_of_others_awareness": [
    "others", "friends", "family", "coworkers", "colleagues", "people around", "other users",
  ],
  "privacy_protection_actions": [
    "protect", "protection", "safeguard", "precaution", "measure", "prevention",
  ],
  "attitudes_toward_being_monitored": [
    "monitor", "monitored", "tracking", "tracked", "surveillance", "watching", "logging",
  ],
  "security_vs_convenience": [
    "convenience", "convenient", "easy", "easier", "quick", "fast", "friction", "effort",
  ],
  "social_vs_privacy": [
    "social", "friends", "following", "followers", "like", "post", "share", "profile", "audience", "community",
  ],
  "knowledge_of_security_privacy_concepts": [
    "understand", "know", "knowledge", "security concept", "privacy concept", "risk", "secure",
  ],
  "sources_of_security_knowledge": [
    "learn", "article", "news", "documentation", "docs", "forum", "video", "course",
  ],
  "types_of_perceived_threats": [
    "phishing", "malware", "identity theft", "account takeover", "breach", "threat", "attack",
  ],
  "daily_online_hours": [
    "time online", "hours online", "online time", "screen time",
  ],
  "social_networking_use": [
    "social media", "social network", "post", "share", "comment", "message", "timeline", "feed",
  ],
  "ability_to_help_others": [
    "help others", "advise", "advice", "teach", "support", "explain", "recommend",
  ],
  "behavior_changes": [
    "after", "incident", "warning", "alert", "breach", "change behavior", "changed behavior",
  ],
  "initiative_on_reviewing_policies": [
    "policy", "policies", "terms", "tos", "privacy notice", "privacy policy", "consent",
  ],
  "proactive_or_reactive": [
    "beforehand", "in advance", "prevent", "preventive", "preemptive", "after an incident", "reactive",
  ],
  "tendency_to_adjust_website_settings": [
    "settings", "permissions", "default settings", "browser setting", "cookie", "toggle",
  ],
  "password_layer": [
    "password strength", "strong password", "weak password", "different passwords", "password policy",
  ],
  "tendency_to_censor_information": [
    "withhold", "omit", "hide", "censor", "limit sharing", "minimal disclosure",
  ],
  "completeness_of_fallback_authentication": [
    "recovery", "backup", "backup email", "backup code", "fallback", "reset", "2fa", "mfa",
  ],
  "system_protection": [
    "antivirus", "firewall", "system protection", "os update", "security tool", "defender",
  ],
  "number_of_passwords": [
    "reuse password", "same password", "multiple passwords", "password reuse",
  ],
  "recording_passwords": [
    "write down password", "store password", "password manager", "remember password", "record password",
  ],
}

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
    """


# =============================================================================
# Common helper functions (same as prompt.py)
# =============================================================================

def get_trait_description(trait_name: str) -> str | None:
  """Return a description for a trait name, which may be a label or field name."""
  import re as _re

  if not trait_name:
    return None

  def _normalize(name: str) -> str:
    name = name.lower().replace(" ", "_")
    name = _re.sub(r"[^a-z0-9_]", "", name)
    return name

  norm = _normalize(trait_name)
  for group in TRAIT_DESCRIPTIONS.values():
    for key, desc in group.items():
      if _normalize(key) == norm:
        return desc
  return None


def flatten_persona_traits(persona_traits):
  flat_traits = {}
  for section_traits in persona_traits.values():
    for key, value in section_traits.items():
      flat_traits[key] = value
  return flat_traits


@lru_cache(maxsize=1)
def load_dimension_module():
  module_path = Path(__file__).resolve().parents[1] / "persona_generator" / "4-dimensional_personality.py"
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


def score_rule_based_traits(task, persona_traits):
  task_text = json.dumps(task, ensure_ascii=False).lower()
  flat_traits = flatten_persona_traits(persona_traits)
  trait_scores = {trait: 0 for trait in flat_traits}

  for trait in flat_traits:
    normalized_trait = trait.replace("_", " ").lower()
    for token in normalized_trait.split():
      if len(token) >= 4 and token in task_text:
        trait_scores[trait] += 2

    for hint in TASK_TRAIT_HINTS.get(trait, []):
      if hint in task_text:
        trait_scores[trait] += 3

    if trait in KEY_DECISION_TRAITS:
      trait_scores[trait] += 1

  return trait_scores


def select_rule_based_traits(task, persona_traits, candidate_limit=10):
  flat_traits = flatten_persona_traits(persona_traits)
  trait_scores = score_rule_based_traits(task, persona_traits)
  sorted_traits = sorted(
    flat_traits.items(),
    key=lambda item: (
      trait_scores.get(item[0], 0),
      TRAIT_PRIORITY.get(item[0], 0),
      item[0],
    ),
    reverse=True,
  )

  selected_traits = []
  for trait, _value in sorted_traits:
    if trait_scores.get(trait, 0) > 0:
      selected_traits.append(trait)
    if len(selected_traits) >= candidate_limit:
      break

  for trait in KEY_DECISION_TRAITS:
    if trait in flat_traits and trait not in selected_traits:
      selected_traits.append(trait)
    if len(selected_traits) >= candidate_limit:
      break

  if not selected_traits:
    for trait, _value in sorted_traits[:candidate_limit]:
      selected_traits.append(trait)

  return {trait: flat_traits[trait] for trait in selected_traits}


def merge_relevant_traits(rule_traits, llm_traits, max_traits=8):
  merged = {}

  for trait in KEY_DECISION_TRAITS:
    if trait in rule_traits and trait not in merged:
      merged[trait] = rule_traits[trait]

  sorted_llm_traits = sorted(
    llm_traits.items(),
    key=lambda item: (TRAIT_PRIORITY.get(item[0], 0), item[0]),
    reverse=True,
  )
  for trait, value in sorted_llm_traits:
    if trait not in merged:
      merged[trait] = value

  sorted_rule_traits = sorted(
    rule_traits.items(),
    key=lambda item: (TRAIT_PRIORITY.get(item[0], 0), item[0]),
    reverse=True,
  )
  for trait, value in sorted_rule_traits:
    if trait not in merged:
      merged[trait] = value

  return dict(list(merged.items())[:max_traits])


# =============================================================================
# RAG (Retrieval-Augmented Generation) functions
# =============================================================================

@lru_cache(maxsize=1)
def _load_qa_trait_mapping():
  """
  Load question_number -> trait mapping from rag_docs.jsonl.
  Uses the first participant's data since question numbers are consistent
  across all participants.
  Returns: dict[int, str]  e.g. {1: "openness", 2: "conscientiousness", ...}
  """
  jsonl_path = Path(__file__).resolve().parent / ".." / "data" / "rag_docs_new_30%.jsonl"
  # Normalize to avoid ../ in path
  jsonl_path = jsonl_path.resolve()
  mapping = {}
  seen_numbers = set()
  with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      entry = json.loads(line)
      qn = entry["question_number"]
      if qn not in seen_numbers:
        mapping[qn] = entry["trait"]
        seen_numbers.add(qn)
  return mapping


@lru_cache(maxsize=128)
def load_participant_rag_docs(participant_number: int) -> list[dict] | None:
  """
  Load a participant's Q&A knowledge base document from
  data/rag_participant_docs/participant-{number}-*.json

  Each Q&A entry in the file is {"Q": "...", "A": "..."}.
  This function enriches each entry with:
    - "question_number": inferred from position (1-indexed)
    - "trait": matched from the JSONL mapping

  Returns: list of dicts or None if the file is not found.
  """
  docs_dir = Path(__file__).resolve().parent / ".." / "data" / "rag_participant_docs"
  docs_dir = docs_dir.resolve()

  # Find the file matching "participant-{number}-*.json"
  matches = list(docs_dir.glob(f"participant-{participant_number}-*.json"))
  if not matches:
    print(f"[RAG] No knowledge base file found for participant {participant_number}")
    return None

  file_path = matches[0]
  with open(file_path, "r", encoding="utf-8-sig") as f:
    qa_list = json.load(f)

  # Enrich each Q&A with question_number (position-based) and trait
  trait_mapping = _load_qa_trait_mapping()
  for idx, qa in enumerate(qa_list):
    qn = idx + 1  # 1-indexed question number
    qa["question_number"] = qn
    qa["trait"] = trait_mapping.get(qn, "unknown")

  return qa_list


def _tokenize_for_matching(text: str) -> set[str]:
  """
  Tokenize text into a set of lowercase meaningful words for matching.
  Filters out stop words and very short tokens.
  """
  if not text:
    return set()

  STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "us", "you", "your", "he", "she", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "what", "which",
    "who", "whom", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "about",
    "above", "after", "again", "against", "and", "as", "at", "but", "by",
    "for", "from", "in", "into", "of", "off", "on", "or", "out", "over",
    "to", "under", "up", "with", "also", "if", "then", "else", "nor",
    "here", "there", "now", "still", "yet", "already", "always", "ever",
  }

  # Remove punctuation and split
  text = re.sub(r"[^\w\s]", " ", text.lower())
  tokens = text.split()
  return {t for t in tokens if len(t) >= 3 and t not in STOP_WORDS}


def compute_qa_relevance(task, qa_item: dict) -> float:
  """
  Compute a relevance score between a task and a Q&A pair.

  Uses a combination of:
    1. Keyword overlap between task text and the Q&A question
    2. Trait-level matching: if the Q&A's trait is relevant to the task

  Returns a float score (higher = more relevant).
  """
  # Extract full task text
  if isinstance(task, dict):
    task_text = json.dumps(task, ensure_ascii=False)
  elif isinstance(task, str):
    task_text = task
  else:
    task_text = str(task)

  qa_question = qa_item.get("Q", "")

  # Token-level overlap
  task_tokens = _tokenize_for_matching(task_text)
  qa_tokens = _tokenize_for_matching(qa_question)

  if not task_tokens or not qa_tokens:
    return 0.0

  overlap = task_tokens & qa_tokens
  # Jaccard-like similarity
  keyword_score = len(overlap) / max(1, len(qa_tokens))

  # Trait-level matching: check if Q&A's trait appears in task hints
  trait_score = 0.0
  qa_trait = qa_item.get("trait", "")
  if qa_trait and qa_trait != "unknown":
    hints = TASK_TRAIT_HINTS.get(qa_trait, [])
    task_lower = task_text.lower()
    for hint in hints:
      if hint in task_lower:
        trait_score += 0.15
    trait_score = min(trait_score, 0.6)  # Cap trait contribution

  # Combine scores (keyword overlap is primary, trait is secondary)
  combined = keyword_score + trait_score
  return combined


def retrieve_top_k_qa(
    participant_docs: list[dict] | None,
    task,
    k: int = 3
) -> list[dict]:
  """
  Retrieve the top-k most relevant Q&A pairs from a participant's knowledge
  base for the given task.

  Args:
    participant_docs: list of enriched Q&A dicts (or None)
    task: the task (dict or string) to match against
    k: number of top results to return (default 3)

  Returns:
    list of up to k Q&A dicts, sorted by relevance (most relevant first).
    Returns an empty list if no suitable docs are found.
  """
  if not participant_docs:
    return []

  scored = []
  for qa in participant_docs:
    score = compute_qa_relevance(task, qa)
    if score > 0:
      scored.append((score, qa))

  # Sort by score descending
  scored.sort(key=lambda x: x[0], reverse=True)

  return [qa for _score, qa in scored[:k]]


def format_rag_block(top_qa_list: list[dict]) -> str:
  """
  Format the retrieved top-k Q&A pairs into a prompt block for insertion
  into the user_content.
  """
  if not top_qa_list:
    return ""

  lines = [
    "[RAG_RETRIEVED_KNOWLEDGE]",
    "Below are the most relevant Q&A pairs from this persona's knowledge base "
    "that may help inform the response to this task. Use these to better anchor "
    "the persona's response in their actual stated beliefs and preferences.",
    "",
  ]
  for i, qa in enumerate(top_qa_list, 1):
    lines.append(f"Q{i}: {qa.get('Q', '')}")
    lines.append(f"A{i}: {qa.get('A', '')}")
    lines.append("")

  lines.append("[/RAG_RETRIEVED_KNOWLEDGE]")
  return "\n".join(lines)


# =============================================================================
# Prompt construction (RAG-only persona context)
# =============================================================================

def construct_system_prompt(persona_traits, persona_id):
  """Build a RAG-only prompt; persona_traits is ignored for compatibility."""
  system_content = (
    "Respond to the task as the participant represented by the retrieved Q&A "
    "in RAG_RETRIEVED_KNOWLEDGE. Use those Q&A pairs as the only source "
    "of participant-specific beliefs, preferences, knowledge, and behaviors. "
    "Do not invent additional personality traits or a separate persona profile.\n"
    "If no retrieved Q&A is available, answer using only the task information "
    "without fabricating participant-specific details.\n\n"
    "[OUTPUT FORMAT]\n"
    "Only output a single JSON object with the following structure:\n"
    f'{{\n"persona": {json.dumps(persona_id)},\n'
    '"task_output": "final content",\n'
    '"reason": "brief explanation of the reasoning behind the task_output"\n}\n'
    "[/OUTPUT FORMAT]\n"
  )
  return system_content, ""


def construct_user_prompt(task, participant_rag_docs=None):
  """Return only retrieved Q&A and the task."""
  top_qa = retrieve_top_k_qa(participant_rag_docs, task, k=3) if participant_rag_docs else []
  rag_block = format_rag_block(top_qa)
  user_content = (
    (f"{rag_block}\n\n" if rag_block else "")
    + f"Task Description:\n{json.dumps(task, indent=2)}"
  )
  return user_content, top_qa


def task_traits(task, persona_traits):
    """
    Use the LLM to determine task-relevant traits from the task content.
    """
    # Simple keyword matching.
    relevant_traits = {}
    prompt = f"""
You are an expert in psychology and user modeling. Given the following task description, identify the most relevant user traits from the provided persona traits that would influence how a user might respond to this task.
Task Description:
{json.dumps(task, indent=2)}

Persona Traits:
{json.dumps(persona_traits, indent=2)}
Please return a JSON object containing five to seven relevant traits and their values that are most likely to impact the user's decisions or behavior in relation to the task. If no traits are relevant, return an empty JSON object: {{}}
    """
    try:
      relevant_traits = ask_deepseek("", prompt)
    except Exception:
        relevant_traits = {}
    return relevant_traits


# =============================================================================
# JSON repair and LLM calling (same as prompt.py)
# =============================================================================

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

def ask_glm(system_content, user_content):
    client = ZhipuAiClient(api_key=f"{GLM_API_KEY}")

    response = client.chat.completions.create(
        model="glm-5.1",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        stream=False,
        thinking={
        "type": "disabled",    # Enable deep reasoning mode.
        },
        max_tokens=65536  # Maximum output length.
    )
    raw_content = response.choices[0].message.content
    response_text = repair_json_response(raw_content)
    try:
        if response_text is None:
            raise ValueError("Empty JSON response")
        parsed = json.loads(response_text)
    except Exception:
        response = client.chat.completions.create(
            model="glm-5.1",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            stream=False,
            thinking={
                "type": "disabled"
            },
            max_tokens=65536  # Maximum output length.
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


def result_correction(system_content, task, response, evaluator_response):
    # Generate a correction prompt based on evaluator feedback.
    prompt = f"""
      You are an assistant that refines model responses based on evaluator feedback.
      Here is the original task:
      {task}
      Here is the model's original response:
      {response}
      Here is the evaluator's feedback:
      {evaluator_response}
      Please revise the model's response to address any issues highlighted by the evaluator.
      Return only the revised response in the same JSON format as the original response.
    """
    text = ask_deepseek(system_content, prompt)
    return text


def parse_response(response_text):
    """
    Parse LLM output and return a dictionary with three fields:
    {
      "persona": ...,
      "task_output": ...,
      "decision_points": [...]
    }
    Be as fault tolerant as possible: support direct dictionaries, complete JSON strings, or miscellaneous text containing JSON.
    """
    if response_text is None:
      return {
          "persona": None,
          "task_output": None,
          "decision_points": []
      }, False
    # Attempt direct parsing.
    try:
        obj = json.loads(response_text)
        return {
            "persona": obj.get("persona", None),
            "task_output": obj.get("task_output", None),
            "decision_points": obj.get("decision_points", [])
        }, True
    except Exception:
      # Extract target fields from the parsed result with type-tolerant handling.
      result = {
          "persona": None,
          "task_output": None,
          "decision_points": []
      }
      # Regular-expression matching.
      persona_match = re.search(r'"persona"\s*:\s*"(.*?)"', str(response_text))
      task_output_match = re.search(r'"task_output"\s*:\s*"(.*?)"', str(response_text), re.DOTALL)
      decision_points_match = re.search(r'"decision_points"\s*:\s*\[(.*?)\]', str(response_text), re.DOTALL)
      if persona_match:
          result["persona"] = persona_match.group(1)
      if task_output_match:
          result["task_output"] = task_output_match.group(1)
      if decision_points_match:
          points_str = decision_points_match.group(1)
          points = re.findall(r'"(.*?)"', points_str)
          result["decision_points"] = points
      if result["persona"] is not None or result["task_output"] is not None or result["decision_points"]:
          return result, True
      else:
          return result, False




# =============================================================================
# Main entry point
# =============================================================================

def _extract_participant_number(persona_id: str) -> int | None:
  """
  Extract the participant number from a persona_id like 'P30' -> 30.
  Returns None if the format doesn't match.
  """
  if not persona_id:
    return None
  m = re.match(r"^P(\d+)$", str(persona_id))
  if m:
    return int(m.group(1))
  return None


def main(persona_dir, results_dir, task_dir, task_name, persona_ids=None):
  # Define paths
  persona_dir = os.path.join(os.path.dirname(__file__), persona_dir)
  results_dir = os.path.join(os.path.dirname(__file__), results_dir)
  task_dir = os.path.join(os.path.dirname(__file__), task_dir)
  # assessment_task or contextual_task or setting_task
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

    # ---------- persona_ids filtering ----------
    # If persona_ids is explicitly passed (e.g. from command line), use it.
    # Otherwise, the calling code's run_one_persona_code handles the scope.
    if persona_ids is not None:
      if persona_data['persona_id'] not in persona_ids:
        continue

    if not os.path.exists(os.path.join(results_dir, persona_data['persona_id'])):
      os.makedirs(os.path.join(results_dir, persona_data['persona_id']))

    # ---------- RAG: load participant knowledge base docs ----------
    participant_number = _extract_participant_number(persona_data.get("persona_id", ""))
    participant_rag_docs = None
    if participant_number is not None:
      participant_rag_docs = load_participant_rag_docs(participant_number)
      if participant_rag_docs:
        print(f"[RAG] Loaded {len(participant_rag_docs)} Q&A docs for {persona_data['persona_id']} (participant #{participant_number})")
      else:
        print(f"[RAG] No docs found for {persona_data['persona_id']} (participant #{participant_number})")
    else:
      print(f"[RAG] Could not extract participant number from persona_id '{persona_data.get('persona_id', '')}'")

    for i in range(1, 4):  # Run multiple times.
      for task in task_data:
        try:
          # Select traits for evaluator compatibility; they are excluded from user_prompt.
          relevant_traits = task_traits(task, persona_data["persona_traits"])
          while len(relevant_traits) == 0:
              print("Retrying to get relevant traits...")
              relevant_traits = task_traits(task, persona_data["persona_traits"])
          system_content, _ = construct_system_prompt(None, persona_data["persona_id"])
          user_content, rag_qa = construct_user_prompt(task, participant_rag_docs)
          response = ask_deepseek(system_content, user_content)
          while response is None:
              print("Retrying...")
              response = ask_deepseek(system_content, user_content)

          # evaluator
          evaluation_result = evaluator(persona_data["persona_traits"], task, response, relevant_traits)
          if evaluation_result is None:
              print("Retrying evaluation...")
              evaluation_result = evaluator(persona_data["persona_traits"], task, response, relevant_traits)

          # Save as a JSON file.
          with open(os.path.join(results_dir, persona_data['persona_id'], f"{persona_data['persona_id']}_{task_name}_{i}.json"), "a", encoding="utf-8") as f:
              f.write(json.dumps({
                  "task": task,
                  "relevant_traits": relevant_traits,
                  "rag_qa": rag_qa,
                  "response": response,
                  "evaluation": evaluation_result
              }, ensure_ascii=False) + "\n")

        except Exception as e:
          print(f"Error processing task {task}: {e}")
          continue


if __name__ == "__main__":
  persona_codes = ["DAUH", "DSUH", "DSUL", "VAPH", "VAPL", "VSPH", "VSPL","VSUH", "VSUL", "VAUH"]
  # persona_codes = ["VAPH"]
  max_parallel = 10
  task="knowledge_task"  # or "contextual_task_v3" or "setting_task" or "knowledge_task"

  def run_one_persona_code(persona_code):
    print(f"[START] {persona_code}")
    main(
      persona_dir=f"../persona/user_study_personas/{persona_code}",
      results_dir=f"../results/user_study_personas_with_rag/20260817-{task}/{persona_code}",
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
