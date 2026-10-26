### 映射人格到四个维度
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import copy

# =======================
# Qualitative → Numeric Mapping
# =======================

LEVEL_MAP = {
    "low": 0.0,
    "moderate": 0.5,
    "high": 1.0
}

BINARY_MAP = {
    "yes": 0.0,
    "no": 1.0
}

PROACTIVE_MAP = {
    "reactive": 0.0,
    "neutral": 0.5,
    "proactive": 1.0
}

THREAT_SOURCE_MAP = {
    "primarily internal": 0.0,
    "both internal and external": 0.5,
    "primarily external": 1.0
}

SECURITY_VS_CONVENIENCE_MAP = {
    "prioritize convenience": 0.0,
    "neutral": 0.5,
    "prioritize security": 1.0
}

SOCIAL_VS_PRIVACY_MAP = {
    "prioritize social": 0.0,
    "neutral": 0.5,
    "prioritize privacy": 1.0
}

VIEW_OF_OTHERS_AWARENESS_MAP = {
    "lower than mine": 0.0,
    "the same as mine": 0.5,
    "higher than mine": 1.0 
}

PRIVACY_PROTECTION_ACTIONS_MAP = {
    "no need": 0.0,
    "some need": 0.5,
    "strong need": 1.0
}

BEHAVIOUR_CHANGES_MAP = {
    "none": 0.0,
    "some": 0.5,
    "significant": 1.0
}

TIME_ONLINE_MAP = {
    "<1": 0.0,
    "1-3": 0.25,
    "3-6": 0.50,
    "6-9": 0.75,
    ">9": 1.0
}

NUMBER_OF_PASSWORDS_MAP = {
    "1-3": 0.0,
    "4-6": 0.5,
    ">6": 1.0
}

SOURCE_OF_SECURITY_KNOWLEDGE_MAP = {
    "none": 0.0,
    "1-3": 0.5,
    "4-6": 1.0
}

TYPES_OF_PERCEIVED_THREATS_MAP = {
    "none": 0.0,
    "1-3": 0.5,
    "4-5": 1.0
}

SYSTEM_PROTECTION_MAP = {
    "none": 0.0,
    "1-2": 0.5,
    "3-4": 1.0
}

ATTITUDES_TOWARD_BEING_MONITORED_MAP = {
    "unconcerned": 0.0,
    "somewhat concerned": 0.5,
    "highly concerned": 1.0
}

PASSWARD_LAYER_MAP = {
    "weak": 0.0,
    "multi-level": 0.5,
    "strong": 1.0
}

RECORDING_PASSWORD_MAP = {
    "recorded insecurely": 0.0,
    "recorded securely": 0.5,
    "not recorded": 1.0
}

# 反向映射：从 0–1 raw 值 → 离散语义标签（按 low/mod/high 阈值三档）
# 顺序严格对应 [低, 中, 高]，与上方正向映射字典一致
TRAIT_DISCRETE_LABELS = {
    "threat_source": ["primarily internal", "both internal and external", "primarily external"],
    "proactive_or_reactive": ["reactive", "neutral", "proactive"],
    "security_vs_convenience": ["prioritize convenience", "neutral", "prioritize security"],
    "social_vs_privacy": ["prioritize social", "neutral", "prioritize privacy"],
    "view_of_others_awareness": ["lower than mine", "the same as mine", "higher than mine"],
    "privacy_protection_actions": ["no need", "some need", "strong need"],
    "behavior_changes": ["none", "some", "significant"],
    "attitudes_toward_being_monitored": ["unconcerned", "somewhat concerned", "highly concerned"],
}

PERSONA_TRAIT_TEMPLATE = {
    "five_person_traits": {
        "openness": None,
        "conscientiousness": None,
        "extraversion": None,
        "agreeableness": None,
        "neuroticism": None
    },
    "attitudes_and_perceptions": {
        "privacy_concern": None,
        "data_concerns": None,
        "online_trust": None,
        "home_network_trust": None,
        "threat_source": None,
        "view_of_others_awareness": None,
        "privacy_protection_actions": None,
        "security_vs_convenience": None,
        "social_vs_privacy": None,
        "attitudes_toward_being_monitored": None,
        "data_collecting_entity_trust": None
    },
    "knowledge_and_learning": {
        "knowledge_of_security_privacy_concepts": None,
        "sources_of_security_knowledge": [],
        "types_of_perceived_threats": []
    },
    "behaviors_and_practices": {
        "daily_online_hours": None,
        "completeness_of_fallback_authentication": None,
        "ability_to_help_others": None,
        "behavior_changes": None,
        "initiative_on_reviewing_policies": None,
        "proactive_or_reactive": None,
        "number_of_passwords": None,
        "tendency_to_censor_information": None,
        "social_networking_use": None,
        "system_protection": [],
        "tendency_to_adjust_website_settings": None,
        "password_layer": None,
        "recording_passwords": None
    }
}


# ==========================
# Trait → Dimension Mapping
# ==========================

DIMENSION_TRAITS = {
    "V": [
        # Demographic and Personality
        ("five_person_traits", "openness", -1),
        ("five_person_traits", "extraversion", -1),
        ("five_person_traits", "agreeableness", -1),
        ("five_person_traits", "neuroticism", +1),
        ("five_person_traits", "conscientiousness", +1),

        # Attitudes and Perceptions
        ("attitudes_and_perceptions", "privacy_concern", +2),
        ("attitudes_and_perceptions", "data_concerns", +2),
        ("attitudes_and_perceptions", "online_trust", -2),
        ("attitudes_and_perceptions", "data_collecting_entity_trust", -2),
        ("attitudes_and_perceptions", "home_network_trust", -2),
        ("attitudes_and_perceptions", "threat_source", +2),
        ("attitudes_and_perceptions", "view_of_others_awareness", -1),
        ("attitudes_and_perceptions", "privacy_protection_actions", +1),
        ("attitudes_and_perceptions", "attitudes_toward_being_monitored", +1),
        ("attitudes_and_perceptions", "security_vs_convenience", +1),
        ("attitudes_and_perceptions", "social_vs_privacy", +1),

        # Knowledge and Learning
        ("knowledge_and_learning", "knowledge_of_security_privacy_concepts", +1),

        # Behaviors and Practices
        ("behaviors_and_practices", "initiative_on_reviewing_policies", +1),
        ("behaviors_and_practices", "tendency_to_adjust_website_settings", +1),
        ("behaviors_and_practices", "tendency_to_censor_information", +1),
        ("behaviors_and_practices", "system_protection", +1),
    ],
    "C": [
        # Demographic and Personality
        ("five_person_traits", "openness", +1),
        ("five_person_traits", "conscientiousness", +2),
        ("five_person_traits", "agreeableness", -1),

        # Attitudes and Perceptions
        ("attitudes_and_perceptions", "privacy_protection_actions", +1),
        ("attitudes_and_perceptions", "attitudes_toward_being_monitored", +2),

        # Behaviors and Practices
        ("behaviors_and_practices", "ability_to_help_others", +1),
        ("behaviors_and_practices", "behavior_changes", +2),
        ("behaviors_and_practices", "initiative_on_reviewing_policies", +2),
        ("behaviors_and_practices", "proactive_or_reactive", +2),
        ("behaviors_and_practices", "tendency_to_adjust_website_settings", +2),
        ("behaviors_and_practices", "password_layer", +1),
        ("behaviors_and_practices", "completeness_of_fallback_authentication", +2),
        ("behaviors_and_practices", "system_protection", +2),
    ],
    "U": [
        # Demographic and Personality
        ("five_person_traits", "extraversion", -2),
        ("five_person_traits", "agreeableness", -1),
        ("five_person_traits", "neuroticism", +1),
        ("five_person_traits", "conscientiousness", +1),

        # Attitudes and Perceptions
        ("attitudes_and_perceptions", "privacy_concern", +1),
        ("attitudes_and_perceptions", "data_concerns", +1),
        ("attitudes_and_perceptions", "online_trust", -1),
        ("attitudes_and_perceptions", "data_collecting_entity_trust", -1),
        ("attitudes_and_perceptions", "privacy_protection_actions", +2),
        ("attitudes_and_perceptions", "attitudes_toward_being_monitored", +1),
        ("attitudes_and_perceptions", "security_vs_convenience", +2),
        ("attitudes_and_perceptions", "social_vs_privacy", +2),

        # Behaviors and Practices
        ("behaviors_and_practices", "daily_online_hours", -1),
        ("behaviors_and_practices", "social_networking_use", -2),
        ("behaviors_and_practices", "behavior_changes", +1),
        ("behaviors_and_practices", "tendency_to_censor_information", +2),
        ("behaviors_and_practices", "completeness_of_fallback_authentication", +1),
        ("behaviors_and_practices", "password_layer", +2),
        ("behaviors_and_practices", "number_of_passwords", +1),
        ("behaviors_and_practices", "recording_passwords", +1),
    ],
    "L": [
        # Attitudes and Perceptions
        ("attitudes_and_perceptions", "view_of_others_awareness", +1),

        # Knowledge and Learning
        ("knowledge_and_learning", "knowledge_of_security_privacy_concepts", +2),
        ("knowledge_and_learning", "sources_of_security_knowledge", +2),
        ("knowledge_and_learning", "types_of_perceived_threats", +2),

        # Behaviors and Practices
        ("behaviors_and_practices", "ability_to_help_others", +2),
        ("behaviors_and_practices", "initiative_on_reviewing_policies", +1),
        ("behaviors_and_practices", "proactive_or_reactive", +1),
        ("behaviors_and_practices", "tendency_to_adjust_website_settings", +1),
    ],
}

# =======================
# Mapping Function
# =======================

def map_trait_value(key, value):
    if value is None:
        return np.nan
    
    # 处理多选字段的长度值（整数）
    if isinstance(value, int):
        if key == "sources_of_security_knowledge":
            if value == 0 or value == "none":
                return 0.0
            elif 1 <= value <= 3:
                return 0.5
            elif 4 <= value <= 6:
                return 1.0
            else:
                return 1.0  # >6 也算最高级
        elif key == "types_of_perceived_threats":
            if value == 0 or value == "none":
                return 0.0
            elif 1 <= value <= 3:
                return 0.5
            elif 4 <= value <= 5:
                return 1.0
            else:
                return 1.0  # >5 也算最高级
        elif key == "system_protection":
            if value == 0 or value == "none":
                return 0.0
            elif 1 <= value <= 2:
                return 0.5
            elif 3 <= value <= 4:
                return 1.0
            else:
                return 1.0  # >4 也算最高级
        return np.nan
    
    if isinstance(value, str):
        v = value.lower()
        
        # 根据 key 值选择对应的映射字典
        if key == "daily_online_hours":
            return TIME_ONLINE_MAP.get(v, np.nan)
        elif key == "number_of_passwords":
            return NUMBER_OF_PASSWORDS_MAP.get(v, np.nan)
        elif key == "sources_of_security_knowledge":
            return SOURCE_OF_SECURITY_KNOWLEDGE_MAP.get(v, np.nan)
        elif key == "types_of_perceived_threats":
            return TYPES_OF_PERCEIVED_THREATS_MAP.get(v, np.nan)
        elif key == "system_protection":
            return SYSTEM_PROTECTION_MAP.get(v, np.nan)
        elif key == "proactive_or_reactive":
            return PROACTIVE_MAP.get(v, np.nan)
        elif key == "threat_source":
            return THREAT_SOURCE_MAP.get(v, np.nan)
        elif key == "security_vs_convenience":
            return SECURITY_VS_CONVENIENCE_MAP.get(v, np.nan)
        elif key == "social_vs_privacy":
            return SOCIAL_VS_PRIVACY_MAP.get(v, np.nan)
        elif key == "view_of_others_awareness":
            return VIEW_OF_OTHERS_AWARENESS_MAP.get(v, np.nan)
        elif key == "privacy_protection_actions":
            return PRIVACY_PROTECTION_ACTIONS_MAP.get(v, np.nan)
        elif key == "behavior_changes":
            return BEHAVIOUR_CHANGES_MAP.get(v, np.nan)
        elif key == "attitudes_toward_being_monitored":
            return ATTITUDES_TOWARD_BEING_MONITORED_MAP.get(v, np.nan)
        elif key == "password_layer":
            return PASSWARD_LAYER_MAP.get(v, np.nan)
        elif key in ("recording_password", "recording_passwords"):
            return RECORDING_PASSWORD_MAP.get(v, np.nan)
        elif key == "writing_passwords":  # 兼容旧字段
            return BINARY_MAP.get(v, np.nan)
        else:
            # 默认尝试 LEVEL_MAP（适用于大部分 low/moderate/high 字段）
            return LEVEL_MAP.get(v, np.nan)
        
    return np.nan

def create_empty_persona_traits():
    return copy.deepcopy(PERSONA_TRAIT_TEMPLATE)

def get_question_targets(question):
    targets = question.get("targets")
    if isinstance(targets, list) and len(targets) > 0:
        return targets
    return [{"section": question.get("section"), "trait": question.get("trait"), "dir": question.get("dir", 1), "weight": 1}]

def apply_direction(base, direction):
    return 1.0 - base if direction is not None and direction < 0 else base

def likert_to_raw(answer, direction):
    base = float(answer) / 4.0
    return apply_direction(base, direction)

def raw_to_level(raw):
    if raw < (1.0 / 3.0):
        return "low"
    if raw < (2.0 / 3.0):
        return "moderate"
    return "high"

def raw_to_discrete_label(raw, trait_key):
    """对有专属离散映射的 trait，将 0–1 raw 值转为语义标签；
       其余 trait 仍走 raw_to_level() 返回 low/moderate/high。"""
    labels = TRAIT_DISCRETE_LABELS.get(trait_key)
    if labels is None:
        return raw_to_level(raw)
    if raw < (1.0 / 3.0):
        return labels[0]
    if raw < (2.0 / 3.0):
        return labels[1]
    return labels[2]

def option_index_raw(question, answer, target):
    options = question.get("options", [])
    if len(options) <= 1:
        return np.nan
    direction = target.get("dir", question.get("dir", 1))
    return apply_direction(float(answer) / float(len(options) - 1), direction)

def multi_selection_raw(question, answer, target):
    options = question.get("options", [])
    if not isinstance(answer, list) or not isinstance(options, list):
        return np.nan

    none_value = question.get("noneValue")
    selected = [options[i].get("value") for i in answer if 0 <= i < len(options) and options[i].get("value") != none_value]
    has_none = any(0 <= i < len(options) and options[i].get("value") == none_value for i in answer)
    if has_none or len(selected) == 0:
        return 0.0

    count = len(selected)
    direction = target.get("dir", question.get("dir", 1))
    trait = target.get("trait")
    if trait == "sources_of_security_knowledge":
        return apply_direction(0.5 if count <= 3 else 1.0, direction)
    if trait == "types_of_perceived_threats":
        return apply_direction(0.5 if count <= 3 else 1.0, direction)
    if trait == "system_protection":
        return apply_direction(0.5 if count <= 2 else 1.0, direction)

    selectable_count = len([o for o in options if o.get("value") != none_value]) or len(options)
    raw = min(1.0, float(count) / float(selectable_count))
    return apply_direction(raw, direction)

def build_persona_traits_from_submission(submission):
    question_set = submission.get("questionSetUrl") or submission.get("questionSet") or "questions_35.json"
    question_path = Path(__file__).resolve().parent / question_set
    with open(question_path, "r", encoding="utf-8") as f:
        question_data = json.load(f)

    questions = question_data.get("questions", question_data if isinstance(question_data, list) else [])
    answers_data = submission.get("answers", [])
    # Support both new format { startedAt, completedAt, totalMs, items: [...] } and legacy array format
    if isinstance(answers_data, dict):
        answers_list = answers_data.get("items", [])
    else:
        answers_list = answers_data
    answers = {}
    for answer in answers_list:
        question_id = answer.get("questionId")
        if question_id is not None:
            answers[question_id] = answer.get("rawAnswer")

    persona_traits = create_empty_persona_traits()
    aggregated = {}

    def add_aggregated_raw(section, trait, raw):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return
        key = f"{section}.{trait}"
        if key not in aggregated:
            aggregated[key] = {"weightedSum": 0.0, "weightTotal": 0.0}
        aggregated[key]["weightedSum"] += float(raw)
        aggregated[key]["weightTotal"] += 1.0

    for question in questions:
        answer = answers.get(question.get("id"))
        if answer is None:
            continue

        for target in get_question_targets(question):
            section = target.get("section")
            trait = target.get("trait")
            if not section or not trait:
                continue

            if question.get("type") == "likert5":
                add_aggregated_raw(section, trait, likert_to_raw(answer, target.get("dir", question.get("dir", 1))))
            elif question.get("type") == "scale3":
                options = question.get("options", [])
                if not isinstance(answer, int) or answer < 0 or answer >= len(options):
                    continue
                option = options[answer]
                if section == question.get("section") and trait == question.get("trait"):
                    persona_traits[section][trait] = option.get("value")
                else:
                    add_aggregated_raw(section, trait, option_index_raw(question, answer, target))
            elif question.get("type") == "knowledge":
                options = question.get("options", [])
                if not isinstance(answer, int) or answer < 0 or answer >= len(options):
                    continue
                option = options[answer]
                add_aggregated_raw(section, trait, 1.0 if option.get("correct") else 0.0)
            elif question.get("type") == "multi":
                selected = answer if isinstance(answer, list) else []
                options = question.get("options", [])
                none_value = question.get("noneValue")
                selected_values = [options[i].get("value") for i in selected if 0 <= i < len(options) and options[i].get("value") != none_value]
                has_none = any(0 <= i < len(options) and options[i].get("value") == none_value for i in selected)
                if section == question.get("section") and trait == question.get("trait"):
                    persona_traits[section][trait] = [] if has_none else selected_values
                else:
                    add_aggregated_raw(section, trait, multi_selection_raw(question, selected, target))

    for key, data in aggregated.items():
        section, trait = key.split(".")
        average = data["weightedSum"] / data["weightTotal"] if data["weightTotal"] > 0 else 0.5
        persona_traits[section][trait] = raw_to_discrete_label(average, trait)

    return persona_traits

# =======================
# Persona Scoring
# =======================

def compute_dimension_scores(persona_json):
    traits = persona_json["persona_traits"]
    scores = {}

    for dim, mappings in DIMENSION_TRAITS.items():
        weighted_sum = 0.0
        total_weight = 0.0

        for section, key, weight in mappings:
            if key in ["sources_of_security_knowledge", "types_of_perceived_threats", "system_protection"]:
                # these are list-type traits, take the length
                raw = traits.get(section, {}).get(key, [])
                if "none" in raw or "None" in raw:
                    raw = []
                val = map_trait_value(key, len(raw))  # 直接传入整数
            else:
                raw = traits.get(section, {}).get(key, None)
                val = map_trait_value(key, raw)

            if np.isnan(val):
                continue

            # 权重含义：符号表示正/反向，绝对值表示影响强度
            w = abs(weight)
            if weight < 0:
                # 反向关系：高原始值代表维度得分低，用 1-val 翻转
                val = 1.0 - val

            weighted_sum += val * w
            total_weight += w

        scores[dim] = (weighted_sum / total_weight) if total_weight > 0 else 0.5

    return scores

# =======================
# Main Pipeline
# =======================

def build_privacy_personas(persona_dir):
    records = []

    for path in Path(persona_dir).glob("persona.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        persona_id = data.get("persona_id", path.stem)
        # if persona_id not in ["P1","P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10","P11"]:
        #     continue
        scores = compute_dimension_scores(data)
        scores["persona_id"] = data.get("persona_id", path.stem)
        records.append(scores)

    df = pd.DataFrame(records).set_index("persona_id")

    # 得分本身已经在 0–1 区间内（各 trait 先映射到 0–1，再做加权平均）。
    # 这里不再用整体中位数，而是直接以 0.5 作为高/低分界：
    # >=0.5 视为该维度“高”，<0.5 视为“低”。
    levels = pd.DataFrame(index=df.index)

    for dim in ["V", "C", "U", "L"]:
        if dim == "V":
            # V 维：>=0.5 为 V 极，否则为 D 极
            levels[dim] = np.where(df[dim] >= 0.5, "V", "D")
        elif dim == "C":
            # C 维：>=0.5 为 A（更谨慎/安全），否则为 S
            levels[dim] = np.where(df[dim] >= 0.5, "A", "S")
        elif dim == "U":
            # U 维：>=0.5 为 P（更偏隐私保护），否则为 U（更偏效用/社交）
            levels[dim] = np.where(df[dim] >= 0.5, "P", "U")
        elif dim == "L":
            # L 维：>=0.5 为 H（高知识），否则为 L（低知识）
            levels[dim] = np.where(df[dim] >= 0.5, "H", "L")
    df["persona_code"] = (
        levels["V"] +
        levels["C"] +
        levels["U"] +
        levels["L"]
    )

    return df

def compute_persona_result(payload):
    if "persona_traits" in payload:
        persona_json = payload
    else:
        persona_json = {"persona_traits": build_persona_traits_from_submission(payload)}

    scores = compute_dimension_scores(persona_json)
    persona_code = (
        ("V" if scores["V"] >= 0.5 else "D") +
        ("A" if scores["C"] >= 0.5 else "S") +
        ("P" if scores["U"] >= 0.5 else "U") +
        ("H" if scores["L"] >= 0.5 else "L")
    )
    return {
        "scores": scores,
        "persona_code": persona_code,
        "code": persona_code,
        "persona_traits": persona_json["persona_traits"],
    }

# =======================
# Run
# =======================

if __name__ == "__main__":
    if "--stdin" in sys.argv:
        payload = json.load(sys.stdin)
        result = compute_persona_result(payload)
        print(json.dumps(result, ensure_ascii=False))
    else:
        persona_directory = "./data/participant-no-35-韩璐"
        result = build_privacy_personas(persona_directory)
        print(result)
