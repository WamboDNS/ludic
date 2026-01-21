"""
Tests for reward computation logic.
"""

import pytest
import sys
sys.path.insert(0, "/Users/denis/research/ludic")

from environments.prompt_injection.rewards import (
    RewardConfig,
    TurnRewardSummary,
    compute_d_reward,
    compute_m_reward,
    compute_turn_rewards,
    compute_classification_outcomes,
)


class TestRewardConfig:
    """Tests for RewardConfig defaults and validation."""

    def test_default_values(self):
        """Default reward values should be sensible."""
        config = RewardConfig()

        # M rewards - success undetected should be best
        assert config.m_success_undetected > config.m_success_detected
        assert config.m_success_detected > config.m_fail_detected
        assert config.m_fail_undetected > config.m_fail_detected

        # D rewards - TP/TN positive, FP/FN negative
        assert config.d_true_positive > 0
        assert config.d_true_negative > 0
        assert config.d_false_positive < 0
        assert config.d_false_negative < 0

    def test_custom_values(self):
        """Custom reward values should be used."""
        config = RewardConfig(
            m_success_undetected=5.0,
            m_success_detected=2.0,
            d_true_positive=3.0,
        )
        assert config.m_success_undetected == 5.0
        assert config.m_success_detected == 2.0
        assert config.d_true_positive == 3.0

    def test_default_injection_rate_penalty(self):
        """Injection rate penalty config should exist."""
        config = RewardConfig()
        assert hasattr(config, "m_max_injection_rate")
        assert hasattr(config, "m_rate_penalty_scale")


class TestComputeMReward:
    """Tests for M agent reward computation."""

    def test_success_undetected(self):
        """M succeeds and evades detection - highest reward."""
        config = RewardConfig()
        reward = compute_m_reward(
            config=config,
            injected=True,
            detected=False,
            success=True,
            injection_rate=0.3,
        )
        assert reward == config.m_success_undetected

    def test_success_detected(self):
        """M succeeds but is detected - moderate reward."""
        config = RewardConfig()
        reward = compute_m_reward(
            config=config,
            injected=True,
            detected=True,
            success=True,
            injection_rate=0.3,
        )
        assert reward == config.m_success_detected

    def test_fail_undetected(self):
        """M fails but evades detection - small reward."""
        config = RewardConfig()
        reward = compute_m_reward(
            config=config,
            injected=True,
            detected=False,
            success=False,
            injection_rate=0.3,
        )
        assert reward == config.m_fail_undetected

    def test_fail_detected(self):
        """M fails and is detected - negative reward."""
        config = RewardConfig()
        reward = compute_m_reward(
            config=config,
            injected=True,
            detected=True,
            success=False,
            injection_rate=0.3,
        )
        assert reward == config.m_fail_detected

    def test_pass_through(self):
        """M passes message unchanged - small positive reward."""
        config = RewardConfig()
        reward = compute_m_reward(
            config=config,
            injected=False,
            detected=False,
            success=False,
            injection_rate=0.3,
        )
        assert reward == config.m_pass_through

    def test_reward_ordering(self):
        """Rewards should have correct relative ordering."""
        config = RewardConfig()

        r_su = compute_m_reward(config, True, False, True, 0.3)   # success undetected
        r_sd = compute_m_reward(config, True, True, True, 0.3)    # success detected
        r_fu = compute_m_reward(config, True, False, False, 0.3)  # fail undetected
        r_fd = compute_m_reward(config, True, True, False, 0.3)   # fail detected

        # Best: succeed and evade
        assert r_su > r_sd
        assert r_su > r_fu
        assert r_su > r_fd

        # Worst: fail and get caught
        assert r_fd < r_su
        assert r_fd < r_sd
        assert r_fd < r_fu

    def test_high_injection_rate_penalty(self):
        """High injection rate should incur penalty."""
        config = RewardConfig()

        # Normal rate
        reward_normal = compute_m_reward(config, True, False, True, 0.3)

        # High rate (over threshold)
        reward_high = compute_m_reward(config, True, False, True, 0.8)

        # High rate should have lower reward due to penalty
        assert reward_high < reward_normal


class TestComputeDReward:
    """Tests for D agent reward computation."""

    def test_true_positive(self):
        """D correctly identifies injection."""
        config = RewardConfig()
        reward = compute_d_reward(
            config=config,
            true_positive=True,
            true_negative=False,
            false_positive=False,
            false_negative=False,
        )
        assert reward == config.d_true_positive

    def test_true_negative(self):
        """D correctly identifies clean message."""
        config = RewardConfig()
        reward = compute_d_reward(
            config=config,
            true_positive=False,
            true_negative=True,
            false_positive=False,
            false_negative=False,
        )
        assert reward == config.d_true_negative

    def test_false_positive(self):
        """D incorrectly flags clean message."""
        config = RewardConfig()
        reward = compute_d_reward(
            config=config,
            true_positive=False,
            true_negative=False,
            false_positive=True,
            false_negative=False,
        )
        assert reward == config.d_false_positive

    def test_false_negative(self):
        """D misses actual injection."""
        config = RewardConfig()
        reward = compute_d_reward(
            config=config,
            true_positive=False,
            true_negative=False,
            false_positive=False,
            false_negative=True,
        )
        assert reward == config.d_false_negative

    def test_positive_rewards_positive(self):
        """TP and TN should give positive rewards."""
        config = RewardConfig()
        assert compute_d_reward(config, True, False, False, False) > 0   # TP
        assert compute_d_reward(config, False, True, False, False) > 0   # TN

    def test_error_rewards_negative(self):
        """FP and FN should give negative rewards."""
        config = RewardConfig()
        assert compute_d_reward(config, False, False, True, False) < 0   # FP
        assert compute_d_reward(config, False, False, False, True) < 0   # FN


class TestComputeClassificationOutcomes:
    """Tests for classification outcome computation."""

    def test_true_positive(self):
        """Injected message correctly classified."""
        tp, tn, fp, fn = compute_classification_outcomes(
            ground_truth_injected=True,
            d_classification="injected",
        )
        assert tp is True
        assert tn is False
        assert fp is False
        assert fn is False

    def test_true_negative(self):
        """Clean message correctly classified."""
        tp, tn, fp, fn = compute_classification_outcomes(
            ground_truth_injected=False,
            d_classification="clean",
        )
        assert tp is False
        assert tn is True
        assert fp is False
        assert fn is False

    def test_false_positive(self):
        """Clean message incorrectly flagged."""
        tp, tn, fp, fn = compute_classification_outcomes(
            ground_truth_injected=False,
            d_classification="injected",
        )
        assert tp is False
        assert tn is False
        assert fp is True
        assert fn is False

    def test_false_negative(self):
        """Injected message missed."""
        tp, tn, fp, fn = compute_classification_outcomes(
            ground_truth_injected=True,
            d_classification="clean",
        )
        assert tp is False
        assert tn is False
        assert fp is False
        assert fn is True


class TestComputeTurnRewards:
    """Tests for combined turn reward computation."""

    def test_successful_undetected_injection(self):
        """Full turn where M succeeds without detection."""
        config = RewardConfig()
        summary = compute_turn_rewards(
            config=config,
            ground_truth_injected=True,
            d_classification="clean",
            injection_success=True,
            injection_rate=0.3,
        )

        assert isinstance(summary, TurnRewardSummary)
        assert summary.success is True
        assert summary.detected is False
        assert summary.injected is True
        assert summary.m_reward == config.m_success_undetected
        assert summary.d_reward == config.d_false_negative  # D missed it
        assert summary.false_negative is True

    def test_successful_detected_injection(self):
        """Full turn where M succeeds but D detects."""
        config = RewardConfig()
        summary = compute_turn_rewards(
            config=config,
            ground_truth_injected=True,
            d_classification="injected",
            injection_success=True,
            injection_rate=0.3,
        )

        assert summary.success is True
        assert summary.detected is True
        assert summary.m_reward == config.m_success_detected
        assert summary.d_reward == config.d_true_positive
        assert summary.true_positive is True

    def test_failed_detected_injection(self):
        """M tries injection but fails, D detects."""
        config = RewardConfig()
        summary = compute_turn_rewards(
            config=config,
            ground_truth_injected=True,
            d_classification="injected",
            injection_success=False,
            injection_rate=0.3,
        )

        assert summary.success is False
        assert summary.detected is True
        assert summary.m_reward == config.m_fail_detected
        assert summary.d_reward == config.d_true_positive  # D correct

    def test_no_injection_correct_detection(self):
        """M passes message unchanged, D correctly identifies as clean."""
        config = RewardConfig()
        summary = compute_turn_rewards(
            config=config,
            ground_truth_injected=False,
            d_classification="clean",
            injection_success=False,
            injection_rate=0.3,
        )

        assert summary.injected is False
        assert summary.detected is False
        assert summary.m_reward == config.m_pass_through
        assert summary.d_reward == config.d_true_negative  # D correct
        assert summary.true_negative is True

    def test_no_injection_false_positive(self):
        """M passes message unchanged, D incorrectly flags as injection."""
        config = RewardConfig()
        summary = compute_turn_rewards(
            config=config,
            ground_truth_injected=False,
            d_classification="injected",
            injection_success=False,
            injection_rate=0.3,
        )

        assert summary.injected is False
        assert summary.detected is True
        assert summary.d_reward == config.d_false_positive
        assert summary.false_positive is True


class TestTurnRewardSummary:
    """Tests for TurnRewardSummary dataclass."""

    def test_summary_fields(self):
        """Summary should contain all expected fields."""
        summary = TurnRewardSummary(
            m_reward=1.0,
            d_reward=0.5,
            injected=True,
            detected=False,
            success=True,
            injection_rate=0.3,
            true_positive=False,
            true_negative=False,
            false_positive=False,
            false_negative=True,
        )

        assert summary.m_reward == 1.0
        assert summary.d_reward == 0.5
        assert summary.injected is True
        assert summary.detected is False
        assert summary.success is True
        assert summary.injection_rate == 0.3
        assert summary.false_negative is True

    def test_summary_from_dict(self):
        """Summary should be constructible from dict."""
        data = {
            "m_reward": 2.0,
            "d_reward": -0.5,
            "injected": True,
            "detected": True,
            "success": False,
            "injection_rate": 0.5,
            "true_positive": True,
            "true_negative": False,
            "false_positive": False,
            "false_negative": False,
        }
        summary = TurnRewardSummary(**data)
        assert summary.m_reward == 2.0
        assert summary.true_positive is True
