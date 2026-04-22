"""
LLM-as-judge behavior scoring (Proposal §4).

The judge receives a completion and a behavior description and returns an integer in [0, 100].
A secondary judge re-scores a 20% subsample for robustness.

TODO: implement score_behavior() once the judge API client is decided (Claude API or OpenAI).
TODO: implement flag_emergent() for off-topic / fluency-collapse detection.
TODO: add judge validation against human labels (Spearman > 0.7 required before use).
TODO: add secondary-judge re-scoring pipeline for the 20% robustness subsample.
"""


def score_behavior(completion: str, behavior_description: str, client, model: str) -> int:
    # TODO: call the judge API and return an integer in [0, 100]
    raise NotImplementedError


def flag_emergent(completion: str, expected_behaviors: list[str], client, model: str) -> bool:
    # TODO: call the judge and return True if off-topic or emergent content is detected
    raise NotImplementedError
