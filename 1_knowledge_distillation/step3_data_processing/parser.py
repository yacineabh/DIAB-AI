import json
import os
from pathlib import Path
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_FILE = "diabetix_training_data.jsonl"
RAW_FOLDER  = "raw_batches"

REQUIRED_FIELDS    = {"profile", "input", "reasoning", "output", "category", "diabetes_type"}
VALID_CATEGORIES   = {"analysis", "advisory"}
VALID_TYPES        = {"Type 1", "Type 2"}
MIN_REASONING_LEN  = 150


def try_fix_json(line: str) -> str | None:
    """
    Attempts common auto-fixes before giving up on a line.
    """
    line = line.strip()

    # Remove markdown artifacts
    for prefix in ["```json", "```", "`"]:
        if line.startswith(prefix):
            line = line[len(prefix):]
    for suffix in ["```", "`"]:
        if line.endswith(suffix):
            line = line[:-len(suffix)]
    line = line.strip()

    # Remove leading number + dot (e.g. "1. {...")
    if line and line[0].isdigit():
        dot_pos = line.find(".")
        brace_pos = line.find("{")
        if dot_pos != -1 and brace_pos != -1 and dot_pos < brace_pos:
            line = line[brace_pos:]

    # Remove trailing comma
    if line.endswith(","):
        line = line[:-1]

    return line if line.startswith("{") else None


def validate_example(obj: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, reason_if_invalid).
    """
    # Required fields
    missing = REQUIRED_FIELDS - set(obj.keys())
    if missing:
        return False, f"missing fields: {missing}"

    # Category
    if obj["category"] not in VALID_CATEGORIES:
        return False, f"invalid category: '{obj['category']}'"

    # Diabetes type
    if obj["diabetes_type"] not in VALID_TYPES:
        return False, f"invalid diabetes_type: '{obj['diabetes_type']}'"

    # Reasoning length
    if len(str(obj["reasoning"])) < MIN_REASONING_LEN:
        return False, f"reasoning too short ({len(str(obj['reasoning']))} chars)"

    # Input must contain logs
    if "Glucose" not in str(obj["input"]):
        return False, "input field doesn't look like logs (no 'Glucose' found)"

    # Output must be valid JSON string
    try:
        parsed_output = json.loads(obj["output"])
    except (json.JSONDecodeError, TypeError):
        return False, "output field is not valid JSON string"

    # Output structure check per category
    if obj["category"] == "analysis":
        required_output_keys = {"pattern_observed", "frequency", "likely_cause", "suggestion"}
        missing_keys = required_output_keys - set(parsed_output.keys())
        if missing_keys:
            return False, f"output missing analysis keys: {missing_keys}"

    elif obj["category"] == "advisory":
        required_output_keys = {"meal_recommendations", "icr_assessment", "lifestyle_advice"}
        missing_keys = required_output_keys - set(parsed_output.keys())
        if missing_keys:
            return False, f"output missing advisory keys: {missing_keys}"

    return True, ""


def process_batch(filepath: str) -> tuple[list, list]:
    valid   = []
    skipped = []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    for i, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue

        # Try to fix common issues
        fixed = try_fix_json(line)
        if fixed is None:
            skipped.append((i+1, "not a JSON line after fix attempts", line[:80]))
            continue

        # Parse JSON
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError as e:
            skipped.append((i+1, f"JSON parse error: {e}", fixed[:80]))
            continue

        # Validate
        is_valid, reason = validate_example(obj)
        if not is_valid:
            skipped.append((i+1, reason, ""))
            continue

        valid.append(obj)

    return valid, skipped


def append_to_output(examples: list, output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        for obj in examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def print_batch_report(filename: str, valid: list, skipped: list):
    print(f"\n{'─'*50}")
    print(f"📄 {filename}")
    print(f"   ✅ Valid:   {len(valid)}")
    print(f"   ❌ Skipped: {len(skipped)}")
    if skipped:
        print("   Skipped reasons:")
        for line_num, reason, preview in skipped[:5]:
            print(f"     Line {line_num}: {reason}")
            if preview:
                print(f"       → {preview}")
    if valid:
        cats  = Counter(ex["category"] for ex in valid)
        types = Counter(ex["diabetes_type"] for ex in valid)
        print(f"   Categories:     {dict(cats)}")
        print(f"   Diabetes types: {dict(types)}")


def final_report(output_path: str):
    if not Path(output_path).exists():
        print("No output file found yet.")
        return

    examples = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\n{'='*50}")
    print(f"  FINAL DATASET REPORT")
    print(f"{'='*50}")
    print(f"  Total examples : {len(examples)}")

    cats  = Counter(ex["category"] for ex in examples)
    types = Counter(ex["diabetes_type"] for ex in examples)
    print(f"  Categories     : {dict(cats)}")
    print(f"  Diabetes types : {dict(types)}")

    reasoning_lens = [len(ex["reasoning"]) for ex in examples]
    if reasoning_lens:
        print(f"  Reasoning len  : min={min(reasoning_lens)}, "
              f"max={max(reasoning_lens)}, "
              f"avg={sum(reasoning_lens)//len(reasoning_lens)}")

    # Duplicate check
    seen = set()
    dupes = 0
    for ex in examples:
        key = ex["input"][:120]
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes:
        print(f"  ⚠️  Duplicates : {dupes} detected")
    else:
        print(f"  ✅ Duplicates  : none")

    # Readiness check
    print(f"\n{'─'*50}")
    analysis_count = cats.get("analysis", 0)
    advisory_count = cats.get("advisory", 0)
    t1_count = types.get("Type 1", 0)
    t2_count = types.get("Type 2", 0)

    checks = [
        (analysis_count >= 350,  f"Analysis examples: {analysis_count}/400"),
        (advisory_count >= 175,  f"Advisory examples: {advisory_count}/200"),
        (t1_count >= 270,        f"Type 1 examples:   {t1_count}/300"),
        (t2_count >= 270,        f"Type 2 examples:   {t2_count}/300"),
        (len(examples) >= 550,   f"Total examples:    {len(examples)}/600"),
    ]
    all_pass = True
    for passed, label in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {label}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🚀 Dataset is ready for fine-tuning!")
    else:
        print("  ⚠️  Dataset not yet complete — generate more batches.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":  
    if Path(OUTPUT_FILE).exists():
        print(f"Output file '{OUTPUT_FILE}' already exists. Removing it...")
        os.remove(OUTPUT_FILE)
        print(f"Output file '{OUTPUT_FILE}' removed.")
    batch_files = sorted(Path(RAW_FOLDER).glob("*.jsonl"))

    if not batch_files:
        print(f"No .txt files found in '{RAW_FOLDER}/' folder.")
        print("Save your Claude outputs as batch_01.txt, batch_02.txt, etc.")
    else:
        total_valid = 0
        for batch_path in batch_files:
            valid, skipped = process_batch(str(batch_path))
            append_to_output(valid, OUTPUT_FILE)
            print_batch_report(batch_path.name, valid, skipped)
            total_valid += len(valid)

        print(f"\n{'═'*50}")
        print(f"  Processed {len(batch_files)} batches → {total_valid} examples appended")

    final_report(OUTPUT_FILE)