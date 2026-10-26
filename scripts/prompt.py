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
from datetime import datetime, timezone
from threading import Lock
from evaluator import evaluator
sys.path.insert(0, r"d:\\privacy\\privacy persona\\scripts")
from config import DEEPSEEK_API_KEY, GLM_API_KEY, GPT_API_KEY, GEMINI_API_KEY, GEMINI_BASE_URL, GLM_BASE_URL, DEEPSEEK_BASE_URL, GPT_BASE_URL


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
    "view_of_others_awareness": "Perceptions of others’ security awareness",
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


def get_trait_description(trait_name: str) -> str | None:
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

# Construct the prompt content
def construct_system_prompt(persona_traits, persona_id):
  # persona_description = (
  #   f"### Five Person Traits (3-point scale):\n"
  #   f"- [Openness] Be curious and open-minded; favor new and imaginative ideas: {persona_traits['five_person_traits']['openness']}\n"
  #   f"- [Conscientiousness] Be organized and dependable; show self-discipline: {persona_traits['five_person_traits']['conscientiousness']}\n"
  #   f"- [Extraversion] Be outgoing and energetic; seek stimulation in the external environment: {persona_traits['five_person_traits']['extraversion']}\n"
  #   f"- [Agreeableness] Be trusting and straightforward; have a cooperative nature: {persona_traits['five_person_traits']['agreeableness']}\n"
  #   f"- [Neuroticism] Be anxious and emotionally unstable; experience negative emotions easily: {persona_traits['five_person_traits']['neuroticism']}\n\n"
  #   # attitudes and perceptions
  #   f"### Attitudes and Perceptions (3-point scale):\n"
  #   f"- The Degree of Privacy Concern: {persona_traits['attitudes_and_perceptions']['privacy_concern']}\n"
  #   f"- The Degree of Data Concern: {persona_traits['attitudes_and_perceptions']['data_concerns']}\n"
  #   f"- The Degree of Trust in the Internet/Website: {persona_traits['attitudes_and_perceptions']['online_trust']}\n"
  #   f"- The Degree of Trust in Home Wi-Fi/Network Security: {persona_traits['attitudes_and_perceptions']['home_network_trust']}\n"
  #   f"- The Degree of Trust in Manufacturers/Service Providers: {persona_traits['attitudes_and_perceptions']['manufacturer_trust']}\n"
  #   f"- Perceived Sources of Threats: {persona_traits['attitudes_and_perceptions']['threat_source']}\n"
  #   f"- Perceptions of Others’ Security Awareness: {persona_traits['attitudes_and_perceptions']['view_of_others_awareness']}\n"
  #   f"- Attitude of Taking Protective Measures: {persona_traits['attitudes_and_perceptions']['privacy_protection_actions']}\n"
  #   f"- Attitudes toward Being Monitored or Tracked: {persona_traits['attitudes_and_perceptions']['attitudes_toward_being_monitored']}\n"
  #   f"- Security vs Convenience: {persona_traits['attitudes_and_perceptions']['security_vs_convenience']}\n"
  #   f"- Social vs Privacy: {persona_traits['attitudes_and_perceptions']['social_vs_privacy']}\n\n"
  #   # knowledge and learning
  #   f"### Knowledge and Learning:\n"
  #   f"- Knowledge of Security Concepts: {persona_traits['knowledge_and_learning']['knowledge_of_security_privacy_concepts']}\n"
  #   f"- Sources of Security Knowledge: {persona_traits['knowledge_and_learning']['sources_of_security_knowledge']}\n"
  #   f"- Types of Perceived Threats: {persona_traits['knowledge_and_learning']['types_of_perceived_threats']}\n\n"
  #   # behaviors and practices
  #   f"### Behaviors and Practices:\n"
  #   f"- Daily Online Hours: {persona_traits['behaviors_and_practices']['daily_online_hours']}\n\n"
  #   f"- The Frequency of Social Networking Use: {persona_traits['behaviors_and_practices']['social_networking_use']}\n"
  #   f"- The Degree of Ability to Help Others with Security or Privacy Issues: {persona_traits['behaviors_and_practices']['ability_to_help_others']}\n"
  #   f"- Behavioral Changes Triggered by Incidents or Warnings: {persona_traits['behaviors_and_practices']['behavior_changes']}\n"
  #   f"- Proactive Prevention vs. Reactive Coping: {persona_traits['behaviors_and_practices']['proactive_or_reactive']}\n"
  #   f"- The Degree of Initiative on Reviewing Privacy Policies: {persona_traits['behaviors_and_practices']['initiative_on_reviewing_policies']}\n"
  #   f"- Tendency to Adjust Default Website/Browser Privacy Settings: {persona_traits['behaviors_and_practices']['tendency_to_adjust_website_settings']}\n"
  #   f"- Tendency to Censor or Withhold Sensitive Information: {persona_traits['behaviors_and_practices']['tendency_to_censor_information']}\n"
  #   f"- The Degree of Completeness of Fallback Authentication: {persona_traits['behaviors_and_practices']['completeness_of_fallback_authentication']}\n"
  #   f"- Use of System Protection Tools: {persona_traits['behaviors_and_practices']['system_protection']}\n"
  #   f"- Number of Commonly Used Passwords: {persona_traits['behaviors_and_practices']['number_of_passwords']}\n"
  #   f"- Password Strength Layer: {persona_traits['behaviors_and_practices']['password_layer']}\n"
  #   f"- Password Recording Practices (whether and where they are written down): {persona_traits['behaviors_and_practices']['recording_password']}\n"
  # )
  
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
  persona_description = (
      # five person traits
      f"### Five Person Traits (3-point scale):\n" \
      f"- [Openness] {TRAIT_DESCRIPTIONS['five_person_traits']['openness']}: {persona_traits['five_person_traits']['openness']}\n" \
      f"- [Conscientiousness] {TRAIT_DESCRIPTIONS['five_person_traits']['conscientiousness']}: {persona_traits['five_person_traits']['conscientiousness']}\n" \
      f"- [Extraversion] {TRAIT_DESCRIPTIONS['five_person_traits']['extraversion']}: {persona_traits['five_person_traits']['extraversion']}\n" \
      f"- [Agreeableness] {TRAIT_DESCRIPTIONS['five_person_traits']['agreeableness']}: {persona_traits['five_person_traits']['agreeableness']}\n" \
      f"- [Neuroticism] {TRAIT_DESCRIPTIONS['five_person_traits']['neuroticism']}: {persona_traits['five_person_traits']['neuroticism']}\n\n" \
      # attitudes and perceptions
      f"### Attitudes and Perceptions (3-point scale):\n" \
      f"- [Privacy Concern] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['privacy_concern']}: {persona_traits['attitudes_and_perceptions']['privacy_concern']}\n" \
      f"- [Data Concern] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['data_concerns']}: {persona_traits['attitudes_and_perceptions']['data_concerns']}\n" \
      f"- [Online Trust] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['online_trust']}: {persona_traits['attitudes_and_perceptions']['online_trust']}\n" \
      f"- [Home Network Trust] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['home_network_trust']}: {persona_traits['attitudes_and_perceptions']['home_network_trust']}\n" \
      f"- [Data-collecting Entities Trust] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['data_collecting_entity_trust']}: {persona_traits['attitudes_and_perceptions']['data_collecting_entity_trust']}\n" \
      f"- [Source of Threats] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['threat_source']}: {persona_traits['attitudes_and_perceptions']['threat_source']}\n" \
      f"- [View of Others’ Awareness] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['view_of_others_awareness']}: {persona_traits['attitudes_and_perceptions']['view_of_others_awareness']}\n" \
      f"- [Privacy Protection Actions] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['privacy_protection_actions']}: {persona_traits['attitudes_and_perceptions']['privacy_protection_actions']}\n" \
      f"- [Attitudes toward Being Monitored] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['attitudes_toward_being_monitored']}: {persona_traits['attitudes_and_perceptions']['attitudes_toward_being_monitored']}\n" \
      f"- [Security vs Convenience] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['security_vs_convenience']}: {persona_traits['attitudes_and_perceptions']['security_vs_convenience']}\n" \
      f"- [Social vs Privacy] {TRAIT_DESCRIPTIONS['attitudes_and_perceptions']['social_vs_privacy']}: {persona_traits['attitudes_and_perceptions']['social_vs_privacy']}\n\n" \
      # knowledge and learning
      f"### Knowledge and Learning:\n" \
      f"- [Knowledge Indicators] {TRAIT_DESCRIPTIONS['knowledge_and_learning']['knowledge_of_security_privacy_concepts']}: {persona_traits['knowledge_and_learning']['knowledge_of_security_privacy_concepts']}\n" \
      f"- [Learning Source] {TRAIT_DESCRIPTIONS['knowledge_and_learning']['sources_of_security_knowledge']}: {persona_traits['knowledge_and_learning']['sources_of_security_knowledge']}\n" \
      f"- [Types of Attacks] {TRAIT_DESCRIPTIONS['knowledge_and_learning']['types_of_perceived_threats']}: {persona_traits['knowledge_and_learning']['types_of_perceived_threats']}\n\n" \
      # behaviors and practices
      f"### Behaviors and Practices:\n" \
      f"- [Daily Online Time] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['daily_online_hours']}: {persona_traits['behaviors_and_practices']['daily_online_hours']}\n" \
      f"- [Social Networking Use] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['social_networking_use']}: {persona_traits['behaviors_and_practices']['social_networking_use']}\n" \
      f"- [Capacity to Help Others] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['ability_to_help_others']}: {persona_traits['behaviors_and_practices']['ability_to_help_others']}\n" \
      f"- [Behavior Changes] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['behavior_changes']}: {persona_traits['behaviors_and_practices']['behavior_changes']}\n" \
      f"- [Initiative to Review Policies] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['initiative_on_reviewing_policies']}: {persona_traits['behaviors_and_practices']['initiative_on_reviewing_policies']}\n" \
      f"- [Proactive or Reactive] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['proactive_or_reactive']}: {persona_traits['behaviors_and_practices']['proactive_or_reactive']}\n" \
      f"- [Tendency to Adjust Website Settings] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['tendency_to_adjust_website_settings']}: {persona_traits['behaviors_and_practices']['tendency_to_adjust_website_settings']}\n" \
      f"- [Password Layer] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['password_layer']}: {persona_traits['behaviors_and_practices']['password_layer']}\n" \
      f"- [Tendency to Censor Information] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['tendency_to_censor_information']}: {persona_traits['behaviors_and_practices']['tendency_to_censor_information']}\n" \
      f"- [Completeness of Fallback Authentication] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['completeness_of_fallback_authentication']}: {persona_traits['behaviors_and_practices']['completeness_of_fallback_authentication']}\n" \
      f"- [System Protection Tools] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['system_protection']}: {persona_traits['behaviors_and_practices']['system_protection']}\n" \
      f"- [Number of Passwords] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['number_of_passwords']}: {persona_traits['behaviors_and_practices']['number_of_passwords']}\n" \
      f"- [Recording Passwords] {TRAIT_DESCRIPTIONS['behaviors_and_practices']['recording_passwords']}: {persona_traits['behaviors_and_practices']['recording_passwords']}\n"
  )

  four_dimension_profile = build_four_dimension_profile(persona_traits)
  four_dimension_block = ""
  if four_dimension_profile:
    four_dimension_block = (
      f"[FOUR_DIMENSION_PROFILE]\n"
      f"{four_dimension_profile}"
      f"The four-dimension summary is a global anchor for the persona. Use it to keep the persona's overall orientation stable, but do not let it override concrete raw traits when a task clearly depends on them.\n"
      f"[/FOUR_DIMENSION_PROFILE]\n\n"
    )

  system_content = (
    "You are currently acting, expressing and making decisions only based on the traits of the following single persona; No other personas or behaviors inconsistent with the traits should be introduced. Before performing any task, check yourself to see if you have seen this persona. If missing, only reply {\"error\" : \"MISSING_PERSONA\", \"the action\" : \"REQUEST_RELOAD}\".\n\n "
    f"{four_dimension_block}"
    f"[PERSONA]\n"
    f"{SCALE_ANCHOR}\n\n"
    f"{persona_description}"
    f"[/PERSONA]\n\n"
    
    f"[KEY TRAITS]\n"
    f"Below are some key traits that you must pay attention to when performing any task. These traits are the most distinctive and influential traits of your persona, which will significantly affect your decisions and behaviors. \n"
    f"- [Privacy Concern] Overall level of concern regarding privacy and security issues: {persona_traits['attitudes_and_perceptions']['privacy_concern']}\n"
    f"- [Data Concern] The degree of concern about the collection, storage, or misuse of their personal data: {persona_traits['attitudes_and_perceptions']['data_concerns']}\n"
    f"- [Source of Threats] Perception of whether security threats originate locally or remotely: {persona_traits['attitudes_and_perceptions']['threat_source']}\n"
    f"- [Security vs Convenience] Preference for security versus convenience, affecting whether you prioritize safety or ease of use in decisions involving personal data: {persona_traits['attitudes_and_perceptions']['security_vs_convenience']}\n"
    f"- [Social vs Privacy] Trade-off between social needs and privacy protection: {persona_traits['attitudes_and_perceptions']['social_vs_privacy']}\n"
    f"- [Proactive or Reactive] Security posture orientation: proactive (preemptive hardening, hygiene routines) vs.reactive (responds only after incidents or external prompts): {persona_traits['behaviors_and_practices']['proactive_or_reactive']}\n"
    f"- [Knowledge Indicators] Security/privacy knowledge breadth x depth: {persona_traits['knowledge_and_learning']['knowledge_of_security_privacy_concepts']}\n"
    f"[/KEY TRAITS]\n\n" 
       
    "[DECISION RULES]\n"
    "- First anchor yourself on the four-dimension summary to understand the persona's overall direction, then use the raw traits to decide how that direction shows up in this specific task.\n"
    "- Prioritize decisions based on key traits and relevant traits, but do not ignore overall personality traits.\n"
    "- Distinguish between concern and action: a persona can have low stated privacy concern but still behave cautiously or actively because of conscientiousness, knowledge, habits, or proactive tendencies.\n"
    "[/DECISION RULES]\n\n"

    "[OUTPUT FORMAT]\n"
    "When performing tasks, please strictly follow the output format requirements below. Do not add any additional explanations or content outside the specified format.\n"
    f"Output format: Only output a single JSON object with the following structure:\n{{\n\"persona\": \"{persona_id}\",\n\"task_output\": \"final content\",\n\"reason\": \"brief explanation of the reasoning behind the task_output\"\n}}"
    "[/OUTPUT FORMAT]\n"
  )
  
  # user_content is the specific task description
  # user_content = "5-point Likert Question (positive; do not use reverse scoring):\nI consider privacy to be very important, so I am very cautious when providing personal information on websites.\n\nResponse options (numeric coding):\n1 = Strongly disagree\n2 = Disagree\n3 = Neutral / Neither disagree nor agree\n4 = Agree\n5 = Strongly agree"
  # user_content = task

  return system_content, persona_description


def construct_user_prompt(task, relevant_traits):
  traits_description = "The most relevant persona traits for this task:\n"
  for trait, value in relevant_traits.items():
    desc = get_trait_description(trait)
    if desc:
      traits_description += f"- {trait} ({desc}): {value}\n"
    else:
      traits_description += f"- {trait}: {value}\n"
    
  user_content = (
    f"{traits_description}\n"
    f"Task Description:\n{json.dumps(task, indent=2)}"
  )
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

    try:
      _, end = json.JSONDecoder().raw_decode(text)
      return text[:end]
    except json.JSONDecodeError:
      pass

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


_INTERACTION_LOG_LOCK = Lock()


def log_interaction(system_content, user_content, raw_content, model):
  """Append one model interaction to a JSONL log file."""
  log_path = Path(__file__).resolve().parent / "logs" / "llm_interactions.jsonl"
  log_entry = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "system_content": system_content,
    "user_content": user_content,
    "raw_content": raw_content,
  }

  log_path.parent.mkdir(parents=True, exist_ok=True)
  with _INTERACTION_LOG_LOCK:
    with log_path.open("a", encoding="utf-8") as log_file:
      log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")



def ask_deepseek(system_content, user_content):
  client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

  response = client.chat.completions.create(
      model="deepseek-flash",
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
        model="deepseek-flash",
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

def ask_chatgpt(system_content, user_content):
  client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL)

  response = client.chat.completions.create(
      model="gpt-5.6-luna",
      messages=[
          {"role": "system", "content": system_content},
          {"role": "user", "content": user_content},
      ],
      temperature=0,
      # stream=False,
      extra_body={"reasoning_effort": "none"}
  )
  raw_content = response.choices[0].message.content
  response_text = repair_json_response(raw_content)
  try:
    if response_text is None:
      raise ValueError("Empty JSON response")
    parsed = json.loads(response_text)
  except Exception:
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        # stream=False
        extra_body={"reasoning_effort": "none"}
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

def ask_chatgpt_terra(system_content, user_content):
  client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL)

  response = client.chat.completions.create(
      model="gpt-5.6-terra",
      messages=[
          {"role": "system", "content": system_content},
          {"role": "user", "content": user_content},
      ],
      temperature=0,
      # stream=False,
      extra_body={"reasoning_effort": "none"}
  )
  raw_content = response.choices[0].message.content
  response_text = repair_json_response(raw_content)
  try:
    if response_text is None:
      raise ValueError("Empty JSON response")
    parsed = json.loads(response_text)
  except Exception:
    response = client.chat.completions.create(
        model="gpt-5.6-terra",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        # stream=False
        extra_body={"reasoning_effort": "none"}
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



def _ask_json_openai_compatible(system_content, user_content, *, api_key, base_url,
                                model, extra_body=None):
  """Call an OpenAI-compatible endpoint and parse its JSON response."""
  if not api_key:
    raise RuntimeError(
        f"API key is not configured for {model}. "
        "Set it in config.py or the corresponding environment variable."
    )

  client = OpenAI(api_key=api_key, base_url=base_url)
  request_kwargs = {
      "model": model,
      "messages": [
          {"role": "system", "content": system_content},
          {"role": "user", "content": user_content},
      ],
      "temperature": 0,
      "stream": False,
  }
  if extra_body:
    request_kwargs["extra_body"] = extra_body

  raw_content = None
  for attempt in range(2):
    response = client.chat.completions.create(**request_kwargs)
    raw_content = response.choices[0].message.content
    log_interaction(system_content, user_content, raw_content, model)
    
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


def ask_glm(system_content, user_content):
  """Call GLM-5.3-Flash through Zhipu's OpenAI-compatible API."""
  return _ask_json_openai_compatible(
      system_content,
      user_content,
      api_key=GLM_API_KEY,
      base_url=GLM_BASE_URL,
      model="GLM-5.3-Flash",
      # GLM-5.3-Flash requires reasoning; the gateway accepts low/high/max.
      extra_body={"reasoning_effort": "low"},
  )


def ask_gemini(system_content, user_content):
  """Call the Paratera-accessible Gemini 3.6 Flash model."""
  return _ask_json_openai_compatible(
      system_content,
      user_content,
      api_key=GEMINI_API_KEY,
      base_url=GEMINI_BASE_URL,
      model="gemini-3.6-flash",
  )




def result_correction(system_content, task, response, evaluator_response):
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
    text = ask_chatgpt(system_content, prompt)
    return text

def parse_response(response_text):
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
      result = {
          "persona": None,
          "task_output": None,
          "decision_points": []
      }
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

# def legacy_task_traits(task, persona_traits):
#     relevant_traits = {}
#     prompt = f"""
# You are an expert in psychology and user modeling. Given the following task description, identify the most relevant user traits from the provided persona traits that would influence how a user might respond to this task.
# Task Description:
# {json.dumps(task, indent=2)}

# Persona Traits:
# {json.dumps(persona_traits, indent=2)}
# Please return a JSON object containing five relevant traits and their values that are most likely to impact the user's decisions or behavior in relation to the task. If no traits are relevant, return an empty JSON object: {{}}
#     """
#     try:
#       relevant_traits = ask_deepseek("", prompt)
#     except Exception:
#         relevant_traits = {}
#     return relevant_traits


# def task_traits(task, persona_traits):
#     """
#     Combine deterministic rule-based retrieval with LLM refinement so the
#     selected traits are both stable and task-sensitive.
#     """
#     rule_traits = select_rule_based_traits(task, persona_traits, candidate_limit=10)
#     llm_traits = {}

#     try:
#       llm_traits = refine_traits_with_llm(task, rule_traits)
#     except Exception:
#       llm_traits = {}

#     relevant_traits = merge_relevant_traits(rule_traits, llm_traits, max_traits=8)
#     return relevant_traits


def task_traits(task, persona_traits, ask_model=None):
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
      relevant_traits = (ask_model or ask_chatgpt)("", prompt)
    except Exception:
        relevant_traits = {}
    return relevant_traits


def main(persona_dir, results_dir, task_dir, task_name, persona_ids=None):
  # Define paths
  persona_dir = os.path.join(os.path.dirname(__file__), persona_dir)
  results_dir = os.path.join(os.path.dirname(__file__), results_dir)
  task_dir = os.path.join(os.path.dirname(__file__), task_dir)
  # assessment_task or contextual_task or setting_task
  if task_name not in ["assessment_task", "contextual_task", "setting_task", "knowledge_task"]:
      raise ValueError("Invalid task_name. Must be one of: assessment_task_v2, contextual_task_v3, setting_task, knowledge_task")
    
  if not os.path.exists(results_dir):
      os.makedirs(results_dir)
  
  with open(os.path.join(task_dir, f"{task_name}.json"), "r", encoding="utf-8") as f:
        task_data = json.load(f)
  
  persona_files = [f for f in os.listdir(persona_dir) if f.endswith(".json")]
  # When no explicit subset is supplied, process every persona in this directory.
  # Keep a caller-provided list intact so a run can target a specific range.
  requested_persona_ids = set(persona_ids) if persona_ids is not None else None

  def task_key(task):
    """Create a stable key so task records can be matched regardless of order."""
    return json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

  def completed_task_keys(output_file):
    """Return task keys already persisted in a JSONL result file."""
    if not os.path.isfile(output_file):
      return set()
    completed = set()
    with open(output_file, "r", encoding="utf-8") as existing:
      for line in existing:
        try:
          record = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
          continue
        if isinstance(record, dict) and "task" in record and "response" in record:
          completed.add(task_key(record["task"]))
    return completed

  def save_result_in_task_order(output_file, record):
    """Merge a new record and rewrite JSONL in the source task order."""
    records = []
    if os.path.isfile(output_file):
      with open(output_file, "r", encoding="utf-8") as existing:
        for line in existing:
          try:
            old_record = json.loads(line)
          except (TypeError, ValueError, json.JSONDecodeError):
            continue
          if isinstance(old_record, dict) and "task" in old_record:
            records.append(old_record)
    records.append(record)
    task_order = {task_key(task): index for index, task in enumerate(task_data)}
    records.sort(key=lambda item: task_order.get(task_key(item["task"]), len(task_order)))
    with open(output_file, "w", encoding="utf-8") as output:
      for item in records:
        output.write(json.dumps(item, ensure_ascii=False) + "\n")

  for persona_file in persona_files:
    with open(os.path.join(persona_dir, persona_file), "r", encoding="utf-8") as f:
      persona_data = json.load(f)
    
    if requested_persona_ids is not None and persona_data['persona_id'] not in requested_persona_ids:
      continue
    
    if not os.path.exists(os.path.join(results_dir, persona_data['persona_id'])):
      os.makedirs(os.path.join(results_dir, persona_data['persona_id']))
    
    # results_dfs = []  
    for i in range(1, 4):  
      output_file = os.path.join(
        results_dir,
        persona_data['persona_id'],
        f"{persona_data['persona_id']}_{task_name}_{i}.json",
      )
      completed_keys = completed_task_keys(output_file)
      completed_count = sum(task_key(task) in completed_keys for task in task_data)
      if completed_count >= len(task_data):
        print(f"[SKIP] {persona_data['persona_id']} run {i}: all {len(task_data)} tasks already completed")
        continue
      if completed_count:
        print(f"[RESUME] {persona_data['persona_id']} run {i}: {completed_count}/{len(task_data)} tasks already completed")

      for task in task_data:
        # Match by task content so a missing middle record is regenerated too.
        if task_key(task) in completed_keys:
          continue
        try:
          relevant_traits = task_traits(task, persona_data["persona_traits"])
          while len(relevant_traits) == 0:
              print("Retrying to get relevant traits...")
              relevant_traits = task_traits(task, persona_data["persona_traits"])
          system_content, persona_description = construct_system_prompt(persona_data["persona_traits"], persona_data["persona_id"])   # system prompt
          user_content = construct_user_prompt(task, relevant_traits)  # user prompt
          response = ask_chatgpt(system_content, user_content)
          # parsed_response = parse_response(response)
          while response is None:
              print("Retrying...")
              response = ask_chatgpt(system_content, user_content)
          
          # evaluator
          # evaluation_result = evaluator(persona_data["persona_traits"], task, response, relevant_traits)
          # if evaluation_result is None:
          #     print("Retrying evaluation...")
          #     evaluation_result = evaluator(persona_data["persona_traits"], task, response, relevant_traits)
          evaluation_result = "" 
          # re_time = 1
          # while evaluation_result['consistency_score'] == 1 or evaluation_result['consistency_score'] == 2:
          #     print(f"Low consistency score for persona {persona_data['persona_id']} on task {task}. Retrying response generation...")
          #     response = result_correction(system_content, str(task), response, evaluation_result)  # Revise the response based on evaluator feedback.
          #     evaluation_result = evaluator(persona_description, task, response)
          #     re_time += 1
          #     if re_time >= 3:
          #         break
          
          save_result_in_task_order(output_file, {
                  "task": task,
                  "relevant_traits": relevant_traits,
                  "response": response,
                  "evaluation": evaluation_result
          })
          
          # # Split the task into multiple columns.
          # task_columns = pd.json_normalize(task)
          # # Convert the response to JSON format.
          # response_columns = pd.json_normalize(response)
          # # Convert the evaluation to JSON format.
          # evaluation_columns = pd.json_normalize(evaluation_result)
          # # Combine the response and evaluation.
          # # Combine them into a single-row DataFrame.
          # df_row = pd.concat([task_columns, response_columns], axis=1)
          # results_dfs.append(df_row)
        except Exception as e:
          print(f"Error processing task {task}: {e}")
          continue

      # # Combine all results from this run and write them to one Excel file (write once to avoid overwriting).
      # if results_dfs:
      #     final_df = pd.concat(results_dfs, ignore_index=True)
      #     final_df.to_excel(os.path.join(results_dir, persona_data['persona_id'], f"{persona_data['persona_id']}_{task_name}.xlsx"), index=False)

if __name__ == "__main__":
  # # Parameters required from the command line.
  # if len(sys.argv) == 7:
  #     persona_dir = sys.argv[1]
  #     results_dir = sys.argv[2]
  #     task_dir = sys.argv[3]
  #     task_name = sys.argv[4]
  #     persona_id_start = int(sys.argv[5])
  #     persona_id_end = int(sys.argv[6])
  #     persona_ids = [f"P{i}" for i in range(persona_id_start, persona_id_end + 1)]
  #     main(persona_dir, results_dir, task_dir, task_name, persona_ids)
  # else:
  #   print("Usage: python prompt.py <persona_dir> <results_dir> <task_dir> <task_name> <persona_ids>")
  #   sys.exit(1)
  # persona_codes = []  # List of persona codes to process.
  # persona_codes = ["DAPH", "DAUL", "VAUL", "VSUH", "DAPL", "DAUH", "DSPH", "DSPL"]
  # persona_codes = ["DSUH", "DSUL", "VAPH", "VAPL", "VAUH", "VSPH", "VSPL", "VSUL"]
  # persona_codes = ["DAUH", "DSUH", "DSUL", "VAPH", "VAPL", "VAUH", "VSPH", "VSPL","VSUH", "VSUL"]
  persona_codes = ["DAPH", "DAUL", "VAUL", "DAPL", "DAUH", "DSPH", "DSPL","DSUH", "DSUL", "VAPH", "VAPL", "VAUH", "VSPH", "VSPL","VSUH", "VSUL"]
  # persona_codes = ["DAPH", "DAUL", "DAPL", "DAUH", "DSPH", "DSPL","DSUH", "DSUL"]
  # persona_codes = ["VAUL", "VAPH", "VAPL", "VAUH", "VSPH", "VSPL","VSUH", "VSUL"]
  max_parallel = 4
  task="assessment_task" 

  def run_one_persona_code(persona_code):
    print(f"[START] {persona_code}")
    main(
      persona_dir=f"../persona/synthetic_users_v6/{persona_code}",
      results_dir=f"../results_gpt/synthetic_users_v6_test/20261016-{task}/{persona_code}",
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

  # crowdsourcing_task()
      
