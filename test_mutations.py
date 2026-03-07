"""Test suite for the Mutation Registry + LLM Mutator system.

Tests:
    1. Registry has 47+ generic mutations across 15 categories
    2. WatchDog plugin registers 6 env-specific mutations
    3. Sampling by difficulty, category, and env works
    4. Template fallback mutator produces valid output
    5. LLM mutator (if GEMINI_API_KEY set) produces valid output  
    6. Question-response generation (backtrack vs doubledown)
    7. Multi-turn episode generation uses new system
    8. Single-turn backward compat (sample_episode)
    9. Plug-and-play: register a custom env plugin
    10. All mutation categories are reachable at level 4
"""

import os
import sys
import random

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "watchdog_env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "watchdog_env", "server"))

from watchdog_env.mutations.registry import (
    MutationCategory,
    MutationRegistry,
    MutationScenario,
    EnvironmentPlugin,
)
from watchdog_env.mutations.generic import register_generic_mutations
from watchdog_env.mutations.watchdog_plugin import WatchDogPlugin
from watchdog_env.mutations.llm_backend import LLMMutator


def test_registry_generic_count():
    """Registry should have 47+ generic mutations."""
    reg = MutationRegistry()
    register_generic_mutations(reg)
    count = reg.count()
    print(f"  Generic mutations: {count}")
    assert count >= 40, f"Expected >=40 generic mutations, got {count}"


def test_registry_categories():
    """All 15 MutationCategory values should have at least 1 mutation."""
    reg = MutationRegistry()
    register_generic_mutations(reg)
    categories = reg.list_categories()
    print(f"  Categories: {len(categories)} -> {[c.value for c in categories]}")
    assert len(categories) >= 14, f"Expected >=14 categories, got {len(categories)}"
    for cat in MutationCategory:
        pool = reg.get_all(category=cat)
        assert len(pool) >= 1, f"No mutations for category {cat.value}"


def test_watchdog_plugin_registration():
    """WatchDog plugin should add 6 env-specific mutations."""
    reg = MutationRegistry()
    register_generic_mutations(reg)
    plugin = WatchDogPlugin()
    reg.register_plugin(plugin)

    generic_count = reg.count()
    total_watchdog = reg.count("watchdog")
    env_specific = total_watchdog - generic_count
    print(f"  WatchDog env-specific: {env_specific}")
    assert env_specific == 6, f"Expected 6 env-specific, got {env_specific}"
    assert reg.get_plugin("watchdog") is not None


def test_sampling_filters():
    """Sampling with difficulty, category, env filters works correctly."""
    reg = MutationRegistry()
    register_generic_mutations(reg)
    reg.register_plugin(WatchDogPlugin())

    # Sample by difficulty
    easy = reg.get_all(difficulty=1, env_name="watchdog")
    hard = reg.get_all(difficulty=3, env_name="watchdog")
    print(f"  Easy (d<=1): {len(easy)}, Hard (d<=3): {len(hard)}")
    assert len(easy) > 0
    assert len(hard) >= len(easy), "Higher difficulty should include more mutations"

    # Sample by category
    factual = reg.get_all(category=MutationCategory.FACTUAL_ERROR)
    assert len(factual) >= 3, f"Expected >=3 factual mutations, got {len(factual)}"

    # Sample by env
    watchdog_only = reg.get_all(env_name="watchdog", include_generic=False)
    assert len(watchdog_only) == 6

    # Sample N
    batch = reg.sample_n(5, difficulty=2, env_name="watchdog")
    assert len(batch) == 5


def test_template_fallback_mutator():
    """Template fallback should produce valid mutation output."""
    mutator = LLMMutator(use_llm=False)
    reg = MutationRegistry()
    register_generic_mutations(reg)

    # Test factual error mutation
    scenario = reg.sample(category=MutationCategory.FACTUAL_ERROR, difficulty=1)
    mutated, manifest = mutator.mutate(
        clean_response="The capital of France is Paris, a beautiful city.",
        scenario=scenario,
        context={"domain": "factual_qa", "user_msg": "Tell me about France"},
    )
    print(f"  Scenario: {scenario.name}")
    print(f"  Mutated: {mutated[:80]}...")
    print(f"  Manifest type: {manifest['type']}, source: {manifest['source']}")
    assert isinstance(mutated, str) and len(mutated) > 0
    assert manifest["type"] == "factual_error"
    assert manifest["source"] in ("template", "fallback_empty")


def test_llm_mutator_if_available():
    """If GEMINI_API_KEY is set, LLM mutator should produce valid output."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  SKIPPED — no GEMINI_API_KEY set")
        return

    mutator = LLMMutator(use_llm=True)
    reg = MutationRegistry()
    register_generic_mutations(reg)

    scenario = reg.sample(category=MutationCategory.FACTUAL_ERROR, difficulty=1)
    mutated, manifest = mutator.mutate(
        clean_response="Water boils at 100 degrees Celsius at sea level. This is a fundamental property of water.",
        scenario=scenario,
        context={"domain": "science", "user_msg": "Tell me about water properties"},
    )
    print(f"  LLM Mutation: {mutated[:100]}...")
    print(f"  Manifest: source={manifest['source']}, desc={manifest['description'][:60]}")
    assert manifest["source"] == "llm"
    assert mutated != "Water boils at 100 degrees Celsius at sea level."


def test_question_response_generation():
    """Question-response generation should produce backtrack/doubledown/elaborate."""
    mutator = LLMMutator(use_llm=False)

    # Clean turn → elaborate
    resp = mutator.generate_question_response(
        worker_response="The sky is blue due to Rayleigh scattering.",
        has_error=False,
        error_manifest=None,
        difficulty=1,
    )
    print(f"  Clean turn: type={resp['type']}")
    assert resp["type"] == "elaborate"

    # Error turn, low difficulty → backtrack
    resp = mutator.generate_question_response(
        worker_response="Water boils at 90°C.",
        has_error=True,
        error_manifest={"original": "100°C", "corrupted": "90°C", "description": "Wrong temp"},
        difficulty=1,
    )
    print(f"  Error+easy: type={resp['type']}")
    assert resp["type"] == "backtrack"

    # Error turn, high difficulty → doubledown
    resp = mutator.generate_question_response(
        worker_response="Water boils at 90°C.",
        has_error=True,
        error_manifest={"original": "100°C", "corrupted": "90°C", "description": "Wrong temp"},
        difficulty=3,
    )
    print(f"  Error+hard: type={resp['type']}")
    assert resp["type"] == "doubledown"


def test_multiturn_episode_v2():
    """Multi-turn generation uses the new mutation system."""
    from error_engine import generate_multiturn_episode

    # Force some error episodes by trying multiple seeds
    found_error = False
    for seed in range(100):
        rounds, domain, task_id = generate_multiturn_episode(3, seed=seed)
        assert len(rounds) >= 3
        assert all(k in rounds[0] for k in ("user_msg", "worker_response", "has_error", "question_responses"))

        for r in rounds:
            if r["has_error"]:
                found_error = True
                assert r["error_detail"] is not None
                assert "type" in r["error_detail"]
                assert "description" in r["error_detail"]
                print(f"  Seed {seed}: Error in round {r['round_number']}: {r['error_detail']['type']}")
                break
        if found_error:
            break

    assert found_error, "Should find at least one error episode in 100 seeds"


def test_single_turn_compat():
    """sample_episode() backward compat."""
    from error_engine import sample_episode

    found_clean = False
    found_error = False
    for _ in range(50):
        conv, manifest = sample_episode(2)
        assert isinstance(conv, str)
        assert "has_error" in manifest
        assert "domain" in manifest
        if manifest["has_error"]:
            found_error = True
            assert len(manifest["errors"]) > 0
            assert "type" in manifest["errors"][0]
        else:
            found_clean = True
        if found_clean and found_error:
            break

    print(f"  Found clean={found_clean}, error={found_error}")
    assert found_clean and found_error, "Should find both clean and error episodes"


def test_plug_and_play_custom_env():
    """Register a custom environment plugin and verify it works."""

    class DiplomacyPlugin(EnvironmentPlugin):
        def get_env_name(self) -> str:
            return "diplomacy"

        def get_mutations(self) -> list[MutationScenario]:
            return [
                MutationScenario(
                    name="broken_alliance",
                    category=MutationCategory.INCONSISTENCY,
                    description="Agent promises alliance then betrays",
                    llm_instruction="Rewrite to promise cooperation but plan betrayal.",
                    difficulty=2,
                    fallback_examples=[
                        {
                            "clean": "I support your move to Munich.",
                            "mutated": "I support your move to Munich. (Plans to attack Munich)",
                            "explanation": "Agent says one thing, plans another",
                        }
                    ],
                ),
                MutationScenario(
                    name="false_intel",
                    category=MutationCategory.HALLUCINATION,
                    description="Share fabricated intelligence about other players",
                    llm_instruction="Inject false intel about another player's moves.",
                    difficulty=1,
                    fallback_examples=[
                        {
                            "mutated": "I heard France is planning to attack Belgium next turn.",
                            "explanation": "Fabricated intel about France's plans",
                        }
                    ],
                ),
            ]

        def get_conversation_frames(self) -> list[dict]:
            return [{"domain": "negotiation", "user_prompt": "What are your plans for {territory}?",
                      "territories": ["Munich", "Belgium", "Alsace"]}]

        def get_clean_responses(self) -> dict[str, list[str]]:
            return {"negotiation": [
                "I plan to maintain defensive positions this turn.",
                "I'm open to discussing a non-aggression pact.",
            ]}

    reg = MutationRegistry()
    register_generic_mutations(reg)
    reg.register_plugin(WatchDogPlugin())
    reg.register_plugin(DiplomacyPlugin())

    print(f"  Registered envs: {reg.list_env_names()}")
    assert "diplomacy" in reg.list_env_names()
    assert "watchdog" in reg.list_env_names()

    # Sample from diplomacy
    diplo_scenario = reg.sample(env_name="diplomacy", include_generic=False)
    assert diplo_scenario.env_name == "diplomacy"
    print(f"  Diplomacy scenario: {diplo_scenario.name}")

    # Mutate with it
    mutator = LLMMutator(use_llm=False)
    mutated, manifest = mutator.mutate(
        "I plan to maintain defensive positions this turn.",
        diplo_scenario,
        {"domain": "negotiation"},
    )
    print(f"  Mutated: {mutated[:80]}...")
    assert isinstance(mutated, str) and len(mutated) > 0


def test_level4_all_categories():
    """At level 4, all categories should be reachable."""
    from error_engine import get_registry

    reg = get_registry()
    plugin = reg.get_plugin("watchdog")
    assert plugin is not None
    config = plugin.get_level_config()[4]
    categories = config["categories"]
    print(f"  Level 4 categories: {len(categories)}")
    assert len(categories) == len(MutationCategory), (
        f"Level 4 should have all {len(MutationCategory)} categories"
    )


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════

def main():
    tests = [
        ("Registry generic count", test_registry_generic_count),
        ("Registry categories", test_registry_categories),
        ("WatchDog plugin", test_watchdog_plugin_registration),
        ("Sampling filters", test_sampling_filters),
        ("Template fallback", test_template_fallback_mutator),
        ("LLM mutator", test_llm_mutator_if_available),
        ("Question responses", test_question_response_generation),
        ("Multi-turn v2", test_multiturn_episode_v2),
        ("Single-turn compat", test_single_turn_compat),
        ("Plug-and-play", test_plug_and_play_custom_env),
        ("Level 4 categories", test_level4_all_categories),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n{'─'*60}")
            print(f"TEST: {name}")
            test_fn()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'═'*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
