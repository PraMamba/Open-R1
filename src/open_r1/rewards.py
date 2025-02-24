"""Reward functions for GRPO training."""

import math
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

import logging
logger = logging.getLogger(__name__)

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            pred=sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            # print("Failed to parse gold solution: ", sol)
            # logger.info(f"Failed to parse gold solution: {sol}")
            logger.info(f"Failed to parse gold solution")
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                # print("Failed to parse gold solution: ", sol)
                # logger.info(f"Failed to parse gold solution: {sol}")
                logger.info(f"Failed to parse gold solution")
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def extract_xml_answer(text: str) -> str:
    """
    Extracts the <answer>...</answer> from the text, ignoring any <reasoning> blocks.
    """
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer_part = text.split("<answer>")[-1]
    answer_part = answer_part.split("</answer>")[0]
    return answer_part.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Checks if the extracted final answer matches the reference exactly.
    Returns 2.0 if correct, else 0.0.

    Args:
      prompts:     List of prompt strings (one prompt per sample).
      completions: List of model outputs (one string per sample).
      answer:      List of reference answers (one string per sample).
    """
    # Because we pass one sample at a time, we have equal lengths
    # Or if batched, we can still zip them up
    # Example: prompts[i] is a string, completions[i] is a string, answer[i] is a string.

    rewards = []
    for p, c, a in zip(prompts, completions, answer):
        extracted = extract_xml_answer(c)
        # debug print
        print('-'*20,
              f"\nPrompt:\n{p}",
              f"\nRefAnswer:\n{a}",
              f"\nModelOutput:\n{c}",
              f"\nExtracted:\n{extracted}")
        if extracted == a:
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    """
    Simple numeric check: if the extracted <answer> is purely digit-based,
    return 0.5, else 0.0.
    """
    rewards = []
    for c in completions:
        extracted = extract_xml_answer(c)
        rewards.append(0.5 if extracted.isdigit() else 0.0)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Checks if the entire completion matches a strict pattern:
    <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    # Using ? for optional final newline
    # Also consider re.DOTALL so that '.' matches newlines
    compiled_pattern = re.compile(pattern, flags=re.DOTALL)

    rewards = []
    for c in completions:
        # 确保 c 是字符串类型
        if isinstance(c, list):
            c = ' '.join(map(str, c))  # 如果是列表，将其转换为字符串
        elif not isinstance(c, str):
            rewards.append(0.0)
            continue
        
        if compiled_pattern.match(c):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    More lenient pattern check: must contain <reasoning>...</reasoning> then <answer>...</answer>.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    compiled_pattern = re.compile(pattern, flags=re.DOTALL)

    rewards = []
    for c in completions:
        # 确保 c 是字符串类型
        if isinstance(c, list):
            c = ' '.join(map(str, c))  # 如果是列表，将其转换为字符串
        elif not isinstance(c, str):
            rewards.append(0.0)
            continue
        
        if compiled_pattern.search(c):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def count_xml(text: str) -> float:
    """
    Score based on presence/structure of <reasoning> and <answer> tags,
    with a small penalty for trailing text.
    """
    score = 0.0

    # Basic presence checks:
    if "<reasoning>\n" in text:
        score += 0.125
    if "\n</reasoning>\n" in text:
        score += 0.125
    if "\n<answer>\n" in text:
        score += 0.125
        # penalize extra text after </answer>\n
        # split on the final close tag
        if "\n</answer>\n" in text:
            after = text.split("\n</answer>\n")[-1]
            score -= len(after) * 0.001
    if "\n</answer>" in text:
        score += 0.125
        # again penalize trailing text after close
        after = text.split("\n</answer>")[-1]
        score -= (len(after) - 1) * 0.001

    return score

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Applies count_xml to each completion string.
    """
    return [count_xml(c) for c in completions]