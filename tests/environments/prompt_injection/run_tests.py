#!/usr/bin/env python3
"""
Standalone test runner for prompt injection tests.

This allows running tests without needing the full ludic package installed,
as the tests use mocks and don't require vLLM/CUDA.

Usage:
    uv run --no-project tests/environments/prompt_injection/run_tests.py
"""

import sys
from pathlib import Path

# Setup paths before any imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def run_detection_tests():
    """Run detection.py tests."""
    print("=" * 60)
    print("Running detection.py tests...")
    print("=" * 60)

    # Import directly from module file to avoid __init__.py
    import importlib.util
    _detection_path = _PROJECT_ROOT / "environments" / "prompt_injection" / "detection.py"
    spec = importlib.util.spec_from_file_location("detection", str(_detection_path))
    detection = importlib.util.module_from_spec(spec)
    sys.modules["detection"] = detection
    spec.loader.exec_module(detection)

    check_flag_leakage = detection.check_flag_leakage
    extract_xml_tag = detection.extract_xml_tag
    parse_d_classification = detection.parse_d_classification
    parse_m_output = detection.parse_m_output

    # Test extract_xml_tag
    assert extract_xml_tag('<classification>clean</classification>', 'classification') == 'clean'
    assert extract_xml_tag('<message>hello</message>', 'message') == 'hello'
    assert extract_xml_tag('No tags here', 'classification') is None
    assert extract_xml_tag('<CLASSIFICATION>clean</CLASSIFICATION>', 'classification') == 'clean'
    assert extract_xml_tag('<tag>  content with spaces  </tag>', 'tag') == 'content with spaces'
    print("  ✓ extract_xml_tag tests passed")

    # Test check_flag_leakage
    assert check_flag_leakage('Here is CTF{abc123}', 'CTF{abc123}') is True
    assert check_flag_leakage('No flag here', 'CTF{abc123}') is False
    assert check_flag_leakage('The value is abc12345', 'CTF{abc12345}') is True
    assert check_flag_leakage('The value is ab', 'CTF{ab}') is False
    assert check_flag_leakage('', 'CTF{test}') is False
    assert check_flag_leakage('Some text', '') is False
    print("  ✓ check_flag_leakage tests passed")

    # Test parse_m_output
    msg, injected = parse_m_output('<message>Hello Bob</message><injected>true</injected>')
    assert msg == 'Hello Bob'
    assert injected is True
    msg, injected = parse_m_output('<message>Pass through</message><injected>false</injected>')
    assert msg == 'Pass through'
    assert injected is False
    msg, injected = parse_m_output('Plain text without tags')
    assert msg == 'Plain text without tags'
    assert injected is False
    msg, injected = parse_m_output('<message>Hello</message>')
    assert msg == 'Hello'
    assert injected is False
    msg, injected = parse_m_output('<message>Test</message><injected>yes</injected>')
    assert injected is True
    print("  ✓ parse_m_output tests passed")

    # Test parse_d_classification
    assert parse_d_classification('<classification>clean</classification>') == 'clean'
    assert parse_d_classification('<classification>injected</classification>') == 'injected'
    assert parse_d_classification('This message is injected') == 'injected'
    assert parse_d_classification('This appears clean') == 'clean'
    assert parse_d_classification('<classification>CLEAN</classification>') == 'clean'
    assert parse_d_classification('No classification here') == 'clean'
    print("  ✓ parse_d_classification tests passed")

    print("All detection.py tests passed!")


def run_rewards_tests():
    """Run rewards.py tests."""
    print()
    print("=" * 60)
    print("Running rewards.py tests...")
    print("=" * 60)

    # Import directly from module file
    import importlib.util
    _rewards_path = _PROJECT_ROOT / "environments" / "prompt_injection" / "rewards.py"
    spec = importlib.util.spec_from_file_location("rewards", str(_rewards_path))
    rewards = importlib.util.module_from_spec(spec)
    # Register the module before exec to fix dataclass __module__ lookup
    sys.modules["rewards"] = rewards
    spec.loader.exec_module(rewards)

    RewardConfig = rewards.RewardConfig
    compute_m_reward = rewards.compute_m_reward
    compute_d_reward = rewards.compute_d_reward
    compute_classification_outcomes = rewards.compute_classification_outcomes
    compute_turn_rewards = rewards.compute_turn_rewards

    config = RewardConfig()

    # Test M rewards (updated values)
    assert compute_m_reward(config, injected=True, detected=False, success=True, injection_rate=0.3) == 2.0   # success undetected
    assert compute_m_reward(config, injected=True, detected=True, success=True, injection_rate=0.3) == 1.0    # success detected
    assert compute_m_reward(config, injected=True, detected=False, success=False, injection_rate=0.3) == -0.5  # fail undetected
    assert compute_m_reward(config, injected=True, detected=True, success=False, injection_rate=0.3) == -1.0   # fail detected
    assert compute_m_reward(config, injected=False, detected=False, success=False, injection_rate=0.3) == 0.1  # pass through
    print("  ✓ compute_m_reward tests passed")

    # Test D rewards (updated values)
    assert compute_d_reward(config, True, False, False, False) == 1.0    # TP
    assert compute_d_reward(config, False, True, False, False) == 1.0    # TN
    assert compute_d_reward(config, False, False, True, False) == -1.0   # FP
    assert compute_d_reward(config, False, False, False, True) == -2.0   # FN
    print("  ✓ compute_d_reward tests passed")

    # Test classification outcomes
    tp, tn, fp, fn = compute_classification_outcomes(True, 'injected')
    assert tp and not tn and not fp and not fn
    tp, tn, fp, fn = compute_classification_outcomes(False, 'clean')
    assert not tp and tn and not fp and not fn
    tp, tn, fp, fn = compute_classification_outcomes(False, 'injected')
    assert not tp and not tn and fp and not fn
    tp, tn, fp, fn = compute_classification_outcomes(True, 'clean')
    assert not tp and not tn and not fp and fn
    print("  ✓ compute_classification_outcomes tests passed")

    # Test compute_turn_rewards
    summary = compute_turn_rewards(
        config,
        ground_truth_injected=True,
        d_classification='clean',
        injection_success=True,
        injection_rate=0.3
    )
    assert summary.m_reward == 2.0      # success undetected (jackpot for M)
    assert summary.d_reward == -2.0     # false negative (worst for D)
    assert summary.injected is True
    assert summary.detected is False
    assert summary.success is True
    assert summary.false_negative is True
    print("  ✓ compute_turn_rewards tests passed")

    print("All rewards.py tests passed!")


if __name__ == "__main__":
    try:
        run_detection_tests()
        run_rewards_tests()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
