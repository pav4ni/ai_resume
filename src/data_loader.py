"""
data_loader.py — Load candidates (JSONL stream or JSON array) and job description.

Design notes:
- load_candidates streams line-by-line to avoid loading 487 MB into RAM at once.
- Schema validation intentionally samples only the first 200 records even when
  validate=True, keeping startup well inside the 5-min budget.
- load_candidates_sample loads the pre-built 50-candidate JSON array (safe for dev/test).
- load_job_description returns raw markdown/text.
"""

import json
import os
from typing import Optional

# ---- schema validator (compiled once at module load) -----
_schema_validator = None

def _get_validator():
    """Compile and cache the Draft7Validator against candidate_schema.json."""
    global _schema_validator
    if _schema_validator is None:
        import jsonschema
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'candidate_schema.json')
        schema_path = os.path.normpath(schema_path)
        with open(schema_path, encoding='utf-8') as f:
            schema = json.load(f)
        _schema_validator = jsonschema.Draft7Validator(schema)
    return _schema_validator


def load_candidates(
    path: str = 'data/candidates.jsonl',
    limit: Optional[int] = None,
    validate: bool = False
) -> list:
    """
    Stream-read a JSONL file of candidate dicts.

    Args:
        path:     Path to the .jsonl file (relative to CWD).
        limit:    If set, stop after this many candidates.
        validate: If True, validate the first 200 records against the JSON Schema
                  and print a pass/fail summary.  Full-pool runs should use False.

    Returns:
        List of candidate dicts.
    """
    candidates = []
    validated = 0
    valid_count = 0
    invalid_count = 0
    VALIDATE_SAMPLE = 200  # only validate this many records to stay within budget

    validator = _get_validator() if validate else None

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidates.append(cand)

            if validate and validated < VALIDATE_SAMPLE:
                errors = list(validator.iter_errors(cand))
                if errors:
                    invalid_count += 1
                else:
                    valid_count += 1
                validated += 1

            if limit and len(candidates) >= limit:
                break

    if validate:
        print(f"Schema validation (first {VALIDATE_SAMPLE} records): "
              f"{valid_count} passed, {invalid_count} failed")

    return candidates


def load_candidates_sample(path: str = 'data/sample_candidates.json') -> list:
    """
    Load the 50-candidate JSON array (pre-built sample for dev/test).

    Returns:
        List of candidate dicts.
    """
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_job_description(path: str = 'data/job_description.md') -> str:
    """
    Load the job description as raw text.

    Returns:
        String with the full JD text.
    """
    with open(path, encoding='utf-8') as f:
        return f.read()


# ---- inline test --------------------------------------------------------
if __name__ == '__main__':
    print('=== data_loader self-test ===')

    # 1. Load sample
    sample = load_candidates_sample()
    print(f'Sample count: {len(sample)}')  # expected: 50
    print(f'First candidate_id: {sample[0]["candidate_id"]}')

    # 2. Validate sample candidates against schema
    #    Re-use load_candidates on the JSONL for schema test; for sample JSON
    #    we validate directly using the helper.
    import jsonschema
    validator = _get_validator()
    passed = 0
    failed = 0
    for cand in sample:
        errors = list(validator.iter_errors(cand))
        if errors:
            failed += 1
            print(f"  FAIL {cand['candidate_id']}: {[e.message for e in errors[:2]]}")
        else:
            passed += 1
    print(f'Schema check on 50-sample: {passed} passed, {failed} failed')

    # 3. Load JD
    jd = load_job_description()
    print(f'JD length: {len(jd)} chars')
    print(f'JD first 200 chars: {jd[:200]}')
