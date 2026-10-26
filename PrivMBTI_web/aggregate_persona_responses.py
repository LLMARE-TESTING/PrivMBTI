"""
Aggregate task responses for each persona across four task types.

Run-level outputs are canonicalized *before* voting:
- assessment/contextual answers -> option IDs "1"-"5";
- knowledge answers -> valid option IDs such as "A"-"D";
- setting answers -> an option ID valid for that specific setting.

Exactly three run slots are expected. A final answer is retained when at least
two canonical answers agree; when no majority exists but at least one run
produced a valid answer, the earliest valid run is used (vote_type
'first_valid'). Missing, duplicate, invalid, or internally conflicting outputs
with no valid run produce no final answer and are retained in the
quality-control metadata instead of being silently dropped.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"d:\privacy\privacy persona\privacy-mbti - localhost版本\results\user_study_personas_with_rag_new_30%")
TASK_TYPES = ['assessment_task','contextual_task', 'knowledge_task', 'setting_task',]
OUTPUT_DIR = Path(r"d:\privacy\privacy persona\privacy-mbti - localhost版本\aggrerated_results\persona_responses_with_rag_new_30%")

# 日志记录：收集过程中跳过的 persona（文件数不为 3）以及缺失的 task_type
SKIPPED_IN_COLLECTION = []   # (persona_id, task_type, group, file_count)
MISSING_TASK_TYPES = []      # (persona_id, task_type)
AGGREGATION_ISSUES = []

EXPECTED_RUNS = 3
EXPECTED_RESULT_COUNTS = {
    'assessment_task': 30,
    'contextual_task': 50,
    'knowledge_task': 30,
    'setting_task': 172,
}


def _validate_vote_inputs(answers: list):
    """Return a structural vote error, or None when three run slots exist."""
    if len(answers) != EXPECTED_RUNS:
        return 'incomplete'
    return None


def majority_vote(answers: list) -> tuple:
    """Vote over three already-canonicalized categorical answers.

    Resolution order (each answer is a canonical ID, or None for a missing /
    invalid run):
      - 3 valid & identical       -> 'unanimous'
      - 2 valid & identical       -> 'majority'
      - 3 valid, all different    -> 'first_run'
      - < 3 valid but >= 1 valid  -> 'first_valid' (earliest valid run wins)
      - no valid answer           -> 'invalid'
    """
    vote_error = _validate_vote_inputs(answers)
    if vote_error:
        return None, vote_error

    valid_answers = [answer for answer in answers if answer is not None]
    if len(valid_answers) < EXPECTED_RUNS:
        if len(valid_answers) == 2 and valid_answers[0] == valid_answers[1]:
            return valid_answers[0], 'majority'
        if valid_answers:
            # No majority, but at least one run produced a valid answer.
            # Retain the earliest valid run instead of dropping the answer.
            return valid_answers[0], 'first_valid'
        return None, 'invalid'

    if answers[0] == answers[1] == answers[2]:
        return answers[0], 'unanimous'

    if answers[0] == answers[1] or answers[0] == answers[2]:
        return answers[0], 'majority'

    if answers[1] == answers[2]:
        return answers[1], 'majority'

    return answers[0], 'first_run'


def average_vote_likert(answers: list) -> tuple:
    """Vote over three canonical Likert IDs, averaging a three-way tie."""
    vote_error = _validate_vote_inputs(answers)
    if vote_error:
        return None, vote_error

    valid_answers = [answer for answer in answers if answer is not None]
    if len(valid_answers) < EXPECTED_RUNS:
        if len(valid_answers) == 2 and valid_answers[0] == valid_answers[1]:
            return valid_answers[0], 'majority'
        return None, 'invalid'

    if answers[0] == answers[1] == answers[2]:
        return answers[0], 'unanimous'

    if answers[0] == answers[1] or answers[0] == answers[2]:
        return answers[0], 'majority'

    if answers[1] == answers[2]:
        return answers[1], 'majority'

    try:
        nums = [int(answer) for answer in answers]
    except (ValueError, TypeError):
        return None, 'invalid'

    if any(number < 1 or number > 5 for number in nums):
        return None, 'invalid'

    return str(int(sum(nums) / EXPECTED_RUNS)), 'average'


def normalize_likert_answer(value):
    """Map 4, "4", "4. Agree", and similar values to the option ID "4"."""
    if value is None:
        return None

    text = str(value).strip()
    if text in {'1', '2', '3', '4', '5'}:
        return text

    match = re.match(r'^([1-5])(?:\s*[.\-:)]|\s+)', text)
    return match.group(1) if match else None


def _option_maps(response_options):
    """Return (ID -> full option, ID -> label) maps for dict/list options."""
    full_by_id = {}
    label_by_id = {}

    if isinstance(response_options, dict):
        for option_id, label in response_options.items():
            option_id = str(option_id).strip().upper()
            full_by_id[option_id] = f'{option_id}. {label}'
            label_by_id[option_id] = str(label).strip()
        return full_by_id, label_by_id

    for raw_option in response_options or []:
        raw_text = str(raw_option).strip()
        match = re.match(r'^([A-Za-z]+)\s*[.]\s*(.*)$', raw_text)
        if not match:
            continue
        option_id = match.group(1).upper()
        full_by_id[option_id] = raw_text
        label_by_id[option_id] = match.group(2).strip()

    return full_by_id, label_by_id


def normalize_choice_answer(value, response_options):
    """Resolve a categorical output to one valid option ID without fuzzy matching."""
    if value is None:
        return None

    full_by_id, label_by_id = _option_maps(response_options)
    if not full_by_id:
        return None

    text = str(value).strip()
    upper_text = text.upper()
    if upper_text in full_by_id:
        return upper_text

    folded = text.casefold()
    exact_full = [
        option_id for option_id, option in full_by_id.items()
        if option.casefold() == folded
    ]
    if len(exact_full) == 1:
        return exact_full[0]

    # Check the complete label before interpreting its first word as an ID.
    # This is important for values such as "Only Me".
    exact_label = [
        option_id for option_id, label in label_by_id.items()
        if label.casefold() == folded
    ]
    if len(exact_label) == 1:
        return exact_label[0]

    prefixed = re.match(r'^([A-Za-z]+)(?:\s*[.]|\s+)(.*)$', text)
    if prefixed:
        option_id = prefixed.group(1).upper()
        supplied_label = prefixed.group(2).strip()
        if option_id not in full_by_id:
            return None
        if not supplied_label:
            return option_id
        expected_label = label_by_id[option_id]
        return option_id if supplied_label.casefold() == expected_label.casefold() else None

    return None


def read_jsonl(filepath):
    """Read a JSONL file and return a list of parsed JSON objects."""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def collect_persona_files():
    """
    Build mapping: persona_id -> {task_type -> [list of available file paths]}.
    """
    persona_files = defaultdict(lambda: defaultdict(list))

    for task_type in TASK_TYPES:
        task_dir = BASE_DIR / task_type
        if not task_dir.exists():
            continue

        for group_dir in task_dir.iterdir():
            if not group_dir.is_dir():
                continue

            group_name = group_dir.name
            for persona_dir in group_dir.iterdir():
                if not persona_dir.is_dir():
                    continue

                persona_id = persona_dir.name  # e.g., "P30"
                json_files = sorted(persona_dir.glob(f"{persona_id}_*.json"))

                if json_files:
                    persona_files[persona_id][task_type] = [str(f) for f in json_files]

                if len(json_files) != EXPECTED_RUNS:
                    print(f"WARNING: {persona_id} in {task_type}/{group_name} has {len(json_files)} files (expected 3)")
                    SKIPPED_IN_COLLECTION.append((persona_id, task_type, group_name, len(json_files)))

    return persona_files


def _process_question_task(filepaths, task_type):
    """Process assessment, contextual, or knowledge questions by stable ID."""
    all_runs = [read_jsonl(fp) for fp in filepaths]
    run_count = len(all_runs)

    raw_answers = defaultdict(lambda: [None] * run_count)
    normalized_answers = defaultdict(lambda: [None] * run_count)
    run_presence = defaultdict(lambda: [False] * run_count)
    duplicate_runs = defaultdict(set)
    q_info = {}

    for run_idx, items in enumerate(all_runs):
        for item in items:
            task = item['task']
            q_num = task['question_number']

            if q_num not in q_info:
                q_info[q_num] = {
                    'question_number': q_num,
                    'construct': task.get('construct', ''),
                    'question': task['question'],
                    'question_type': task.get('question_type', ''),
                    'response_options': task.get('response_options', {}),
                }
                if task_type == 'knowledge_task':
                    q_info[q_num]['section'] = task.get('section', '')

            if run_presence[q_num][run_idx]:
                duplicate_runs[q_num].add(run_idx)
                normalized_answers[q_num][run_idx] = None
                continue

            run_presence[q_num][run_idx] = True
            raw_answer = item['response'].get('task_output')
            raw_answers[q_num][run_idx] = raw_answer

            if task_type in ('assessment_task', 'contextual_task'):
                normalized = normalize_likert_answer(raw_answer)
            else:
                normalized = normalize_choice_answer(
                    raw_answer,
                    task.get('response_options', {}),
                )
            normalized_answers[q_num][run_idx] = normalized

    results = []
    for q_num in sorted(q_info.keys()):
        answers = list(normalized_answers[q_num])
        issues = []
        if run_count != EXPECTED_RUNS:
            issues.append(f'expected_{EXPECTED_RUNS}_run_files_found_{run_count}')
        issues.extend(
            f'missing_run_{run_idx + 1}'
            for run_idx, present in enumerate(run_presence[q_num])
            if not present
        )
        issues.extend(
            f'duplicate_run_{run_idx + 1}'
            for run_idx in sorted(duplicate_runs[q_num])
        )
        issues.extend(
            f'invalid_run_{run_idx + 1}'
            for run_idx, (present, answer) in enumerate(
                zip(run_presence[q_num], answers)
            )
            if present and answer is None
        )

        if task_type in ('assessment_task', 'contextual_task'):
            final_answer, vote_type = average_vote_likert(answers)
        else:
            final_answer, vote_type = majority_vote(answers)

        result = {
            **q_info[q_num],
            'raw_answers_from_3_runs': list(raw_answers[q_num]),
            'answers_from_3_runs': answers,
            'final_answer': final_answer,
            'vote_type': vote_type,
            'quality_issues': issues,
        }

        if task_type == 'knowledge_task':
            correct_answer = normalize_choice_answer(
                next(
                    (
                        item['task'].get('correct_answer')
                        for items in all_runs
                        for item in items
                        if item['task']['question_number'] == q_num
                        and item['task'].get('correct_answer')
                    ),
                    None,
                ),
                q_info[q_num]['response_options'],
            )
            if correct_answer is None:
                issues.append('missing_or_invalid_correct_answer')
            result['correct_answer'] = correct_answer
            result['is_correct'] = (
                final_answer == correct_answer
                if final_answer is not None and correct_answer is not None
                else None
            )

        results.append(result)

    return results


def process_assessment_task(filepaths):
    """Process repeated Assessment outputs."""
    return _process_question_task(filepaths, 'assessment_task')


def process_contextual_task(filepaths):
    """Process repeated Contextual outputs."""
    return _process_question_task(filepaths, 'contextual_task')


def process_knowledge_task(filepaths):
    """Process repeated Knowledge outputs."""
    return _process_question_task(filepaths, 'knowledge_task')


# ---------------------------------------------------------------------------
# Known spelling corrections for setting names produced by the LLM.
# Applied automatically during aggregation so downstream comparisons
# always receive canonical names.
# ---------------------------------------------------------------------------
SETTING_NAME_CORRECTIONS = {
    # rednote task 1
    'Acticity Status':               'Activity Status',
    'Show me to people l may know':  'Show me to people I may know',
    'Hide my notes in lmage search': 'Hide my notes in Image search',
    'Recommend people l may know':   'Recommend people I may know',
}


def _correct_setting_name(name: str) -> str:
    """Return the canonical setting name, applying known corrections."""
    return SETTING_NAME_CORRECTIONS.get(name, name)


def extract_option_letter(option_str):
    """Extract an explicit option ID, without checking whether it is valid."""
    if option_str is None:
        return None
    text = str(option_str).strip()
    if re.fullmatch(r'[A-Za-z]+', text):
        return text.upper()
    match = re.match(r'^([A-Za-z]+)(?:\s*[.]|\s+)', text)
    if match:
        return match.group(1).upper()
    return None


def resolve_option_letter(option_value, available_options):
    """Resolve one setting output to a valid option ID for that setting."""
    return normalize_choice_answer(option_value, available_options)


def canonical_option_value(option_id, available_options):
    """Return the canonical full option text for a validated option ID."""
    if option_id is None:
        return None
    full_by_id, _ = _option_maps(available_options)
    return full_by_id.get(option_id)


def normalize_setting_task_output(task_output, source=''):
    """Return setting-task output as a list of setting dictionaries.

    Most records store ``task_output`` as a list. A few records were written
    with the list JSON-encoded a second time, so their value is a string such
    as '[{"setting_name": ...}]'. Parse that representation before merging.
    """
    if isinstance(task_output, str):
        try:
            task_output = json.loads(task_output)
        except json.JSONDecodeError as exc:
            location = f' in {source}' if source else ''
            raise ValueError(f'Invalid JSON string in setting task_output{location}') from exc

    if not isinstance(task_output, list):
        location = f' in {source}' if source else ''
        raise TypeError(
            f'Expected setting task_output to be a list{location}, '
            f'got {type(task_output).__name__}'
        )

    invalid_items = [i for i, setting in enumerate(task_output)
                     if not isinstance(setting, dict)]
    if invalid_items:
        location = f' in {source}' if source else ''
        raise TypeError(
            f'Expected setting task_output items to be objects{location}; '
            f'invalid item indexes: {invalid_items}'
        )

    return task_output


def process_setting_task(filepaths):
    """Process repeated Setting outputs by task number and canonical setting name."""
    all_runs = [read_jsonl(fp) for fp in filepaths]
    run_count = len(all_runs)

    task_outputs = defaultdict(lambda: [None] * run_count)
    task_presence = defaultdict(lambda: [False] * run_count)
    duplicate_task_runs = defaultdict(set)
    malformed_task_runs = defaultdict(set)
    task_info = {}

    for run_idx, items in enumerate(all_runs):
        for item in items:
            task = item['task']
            t_num = task['task_number']

            if t_num not in task_info:
                available_settings = task.get('available privacy settings', [])
                setting_names = []
                settings_available_options = {}
                for avs in available_settings:
                    setting_name = _correct_setting_name(avs['setting_name'])
                    if setting_name not in settings_available_options:
                        setting_names.append(setting_name)
                    settings_available_options[setting_name] = avs.get(
                        'setting_options', []
                    )

                task_info[t_num] = {
                    'task_number': t_num,
                    'app_name': task.get('app name', ''),
                    'app_description': task.get('app description', ''),
                    'setting_names': setting_names,
                    'settings_available_options': settings_available_options,
                }

            if task_presence[t_num][run_idx]:
                duplicate_task_runs[t_num].add(run_idx)
                task_outputs[t_num][run_idx] = None
                continue

            task_presence[t_num][run_idx] = True
            try:
                task_outputs[t_num][run_idx] = normalize_setting_task_output(
                    item['response'].get('task_output'),
                    f'{filepaths[run_idx]}, task {t_num}',
                )
            except (TypeError, ValueError):
                malformed_task_runs[t_num].add(run_idx)
                task_outputs[t_num][run_idx] = None

    results = []
    for t_num in sorted(task_info.keys()):
        info = task_info[t_num]
        expected_setting_names = info['setting_names']
        available_opts_map = info['settings_available_options']

        task_issues = []
        if run_count != EXPECTED_RUNS:
            task_issues.append(
                f'expected_{EXPECTED_RUNS}_run_files_found_{run_count}'
            )
        task_issues.extend(
            f'missing_task_run_{run_idx + 1}'
            for run_idx, present in enumerate(task_presence[t_num])
            if not present
        )
        task_issues.extend(
            f'duplicate_task_run_{run_idx + 1}'
            for run_idx in sorted(duplicate_task_runs[t_num])
        )
        task_issues.extend(
            f'malformed_task_run_{run_idx + 1}'
            for run_idx in sorted(malformed_task_runs[t_num])
        )

        per_run_settings = []
        unexpected_setting_names = []
        for run_idx, run_output in enumerate(task_outputs[t_num]):
            settings_by_name = defaultdict(list)
            for setting in run_output or []:
                raw_name = setting.get('setting_name')
                if not isinstance(raw_name, str):
                    unexpected_setting_names.append({
                        'run': run_idx + 1,
                        'setting_name': raw_name,
                    })
                    continue
                setting_name = _correct_setting_name(raw_name)
                if setting_name not in available_opts_map:
                    unexpected_setting_names.append({
                        'run': run_idx + 1,
                        'setting_name': raw_name,
                    })
                    continue
                settings_by_name[setting_name].append(setting)
            per_run_settings.append(settings_by_name)

        settings_results = []
        for setting_name in expected_setting_names:
            available_options = available_opts_map[setting_name]
            raw_answers = [None] * run_count
            answers = [None] * run_count
            reasonings = [''] * run_count
            issues = list(task_issues)

            for run_idx, settings_by_name in enumerate(per_run_settings):
                candidates = settings_by_name.get(setting_name, [])
                if not candidates:
                    issues.append(f'missing_setting_run_{run_idx + 1}')
                    continue
                if len(candidates) > 1:
                    issues.append(f'duplicate_setting_run_{run_idx + 1}')
                    continue

                setting = candidates[0]
                raw_answer = setting.get('setting_option')
                raw_answers[run_idx] = raw_answer
                reasonings[run_idx] = setting.get('reasoning', '')
                answers[run_idx] = resolve_option_letter(
                    raw_answer,
                    available_options,
                )
                if answers[run_idx] is None:
                    issues.append(f'invalid_option_run_{run_idx + 1}')

            final_option_letter, vote_type = majority_vote(answers)
            final_option = canonical_option_value(
                final_option_letter,
                available_options,
            )
            match_idx = next(
                (
                    run_idx for run_idx, answer in enumerate(answers)
                    if answer == final_option_letter
                ),
                None,
            )
            final_reasoning = (
                reasonings[match_idx] if match_idx is not None else ''
            )

            settings_results.append({
                'setting_name': setting_name,
                'raw_answers_from_3_runs': raw_answers,
                'answers_from_3_runs': answers,
                'final_option': final_option,
                'final_option_letter': final_option_letter,
                'final_reasoning': final_reasoning,
                'vote_type': vote_type,
                'quality_issues': issues,
            })

        results.append({
            'task_number': info['task_number'],
            'app_name': info['app_name'],
            'app_description': info['app_description'],
            'settings': settings_results,
            'quality_issues': task_issues,
            'unexpected_setting_names': unexpected_setting_names,
        })

    return results


def _flatten_task_results(task_type, task_results):
    """Yield individual question/setting results for counts and vote summaries."""
    if task_type == 'setting_task':
        for task in task_results:
            yield from task['settings']
    else:
        yield from task_results


def _record_quality_issues(persona_id, task_type, task_results):
    """Copy non-empty per-result QC metadata into the aggregation log."""
    if task_type == 'setting_task':
        for task in task_results:
            task_issues = task.get('quality_issues', [])
            if task_issues:
                AGGREGATION_ISSUES.append({
                    'persona': persona_id,
                    'task_type': task_type,
                    'task_number': task['task_number'],
                    'issues': task_issues,
                })
            if task.get('unexpected_setting_names'):
                AGGREGATION_ISSUES.append({
                    'persona': persona_id,
                    'task_type': task_type,
                    'task_number': task['task_number'],
                    'issues': ['unexpected_setting_names'],
                    'details': task['unexpected_setting_names'],
                })

            # Task-level issues are logged once above; retain only setting-specific
            # issues here to avoid repeating one missing run for every setting.
            task_issue_set = set(task_issues)
            for setting in task['settings']:
                setting_issues = [
                    issue for issue in setting.get('quality_issues', [])
                    if issue not in task_issue_set
                ]
                if setting_issues:
                    AGGREGATION_ISSUES.append({
                        'persona': persona_id,
                        'task_type': task_type,
                        'task_number': task['task_number'],
                        'setting_name': setting['setting_name'],
                        'issues': setting_issues,
                    })
        return

    for question in task_results:
        if question.get('quality_issues'):
            AGGREGATION_ISSUES.append({
                'persona': persona_id,
                'task_type': task_type,
                'question_number': question['question_number'],
                'issues': question['quality_issues'],
            })


def parse_args():
    """Parse optional paths so each simulation condition can be aggregated safely."""
    parser = argparse.ArgumentParser(
        description='Aggregate three repeated persona-simulation runs.',
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=BASE_DIR,
        help=f'Root directory containing task-type folders (default: {BASE_DIR})',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Directory for aggregated JSON files (default: {OUTPUT_DIR})',
    )
    return parser.parse_args()


def main():
    SKIPPED_IN_COLLECTION.clear()
    MISSING_TASK_TYPES.clear()
    AGGREGATION_ISSUES.clear()

    print("Collecting persona files...")
    persona_files = collect_persona_files()
    print(f"Found {len(persona_files)} unique personas")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for persona_id in sorted(persona_files.keys(), key=lambda x: int(x[1:])):
        task_files = persona_files[persona_id]
        print(f"\nProcessing {persona_id}...")

        # Determine group from file path (all task types should have same group)
        first_task_type = list(task_files.keys())[0]
        first_path = task_files[first_task_type][0]
        group_name = Path(first_path).parent.parent.name

        aggregated = {
            'persona': persona_id,
            'group': group_name,
            'responses': {},
        }

        for task_type in TASK_TYPES:
            if task_type not in task_files:
                print(f"  WARNING: {task_type} missing for {persona_id}")
                aggregated['responses'][task_type] = None
                MISSING_TASK_TYPES.append((persona_id, task_type))
                continue

            fpaths = task_files[task_type]
            if task_type == 'assessment_task':
                aggregated['responses'][task_type] = process_assessment_task(fpaths)
            elif task_type == 'contextual_task':
                aggregated['responses'][task_type] = process_contextual_task(fpaths)
            elif task_type == 'knowledge_task':
                aggregated['responses'][task_type] = process_knowledge_task(fpaths)
            elif task_type == 'setting_task':
                aggregated['responses'][task_type] = process_setting_task(fpaths)

            task_results = aggregated['responses'][task_type]

            # Count individual question/setting results and preserve all QC flags.
            if task_results:
                flat_results = list(
                    _flatten_task_results(task_type, task_results)
                )
                result_count = len(flat_results)
                expected_count = EXPECTED_RESULT_COUNTS[task_type]
                if result_count != expected_count:
                    AGGREGATION_ISSUES.append({
                        'persona': persona_id,
                        'task_type': task_type,
                        'issues': ['unexpected_result_count'],
                        'expected': expected_count,
                        'actual': result_count,
                    })
                _record_quality_issues(persona_id, task_type, task_results)

                all_votes = [result['vote_type'] for result in flat_results]
                vote_counts = Counter(all_votes)
                print(f"  {task_type}: {result_count} questions/settings, "
                      f"votes: unanimous={vote_counts.get('unanimous', 0)}, "
                      f"majority={vote_counts.get('majority', 0)}, "
                      f"first_run={vote_counts.get('first_run', 0)}, "
                      f"first_valid={vote_counts.get('first_valid', 0)}, "
                      f"average={vote_counts.get('average', 0)}, "
                      f"invalid={vote_counts.get('invalid', 0)}, "
                      f"incomplete={vote_counts.get('incomplete', 0)}")

        # Write output
        output_path = OUTPUT_DIR / f"{persona_id}_aggregated_responses.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)

        print(f"  -> Written to {output_path}")

    # 写入日志文件
    log_path = OUTPUT_DIR / "_aggregation_log.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total_personas": len(persona_files),
        "skipped_in_collection": [
            {"persona": p, "task_type": t, "group": g, "file_count": c}
            for p, t, g, c in SKIPPED_IN_COLLECTION
        ],
        "missing_task_types": [
            {"persona": p, "task_type": t}
            for p, t in MISSING_TASK_TYPES
        ],
        "aggregation_issue_count": len(AGGREGATION_ISSUES),
        "aggregation_issues": AGGREGATION_ISSUES,
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"  -> Aggregation log written to {log_path}")

    if MISSING_TASK_TYPES:
        print(f"\n⚠ WARNING: {len(MISSING_TASK_TYPES)} persona(s) have missing task types (output as null):")
        for p, t in MISSING_TASK_TYPES:
            print(f"  - {p}: {t}")

    if AGGREGATION_ISSUES:
        print(
            f"\n⚠ WARNING: {len(AGGREGATION_ISSUES)} aggregation quality "
            f"issue record(s); see {log_path}"
        )

    print(f"\nDone! Output files written to {OUTPUT_DIR}")


if __name__ == '__main__':
    args = parse_args()
    BASE_DIR = args.base_dir
    OUTPUT_DIR = args.output_dir
    main()
