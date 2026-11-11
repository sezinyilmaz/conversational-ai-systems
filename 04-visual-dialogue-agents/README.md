# Visual Dialogue Agents for GuessWhat?!

Two cooperating agents built on one small vision-language model play the [GuessWhat?!](https://arxiv.org/abs/1611.08481) game:

- the **Questioner** asks yes/no questions about an image to find a hidden object;
- the **Oracle**, which knows the target, answers *yes*, *no* or *n/a*.

Notebook: [`guesswhat_vlm_agents.ipynb`](guesswhat_vlm_agents.ipynb)

## Approach

- **Model:** `Qwen/Qwen3.5-0.8B` (image-text-to-text). Both roles share one set of weights and are separated by their prompts.
- **Grounding:** candidate objects are drawn on the image and also listed in text with their category and bounding box, normalised to Qwen's 0–999 coordinate scale. This lets the model refer to objects by index and position.
- **Questioner:** the free-form prompt produced unstable output, including broken and non-English questions. For evaluation I therefore switched to a constrained, template-based Questioner, with a rule-based selector that picks the final candidate.
- **Oracle:** a strict prompt asking for a single label, plus an answer normaliser that recovers *yes/no/n/a* from longer replies.
- **Decoding:** deterministic (`do_sample=False`) for stable, short outputs.
- **LoRA fine-tuning of the Oracle:** 88 supervised examples built from 40 games, with the evaluation games excluded. LoRA r = 8, α = 16, 22 steps. Training loss fell from ~10.6 to ~3.1.

## Results (20 games, 60 Oracle answers)

| Oracle | Game accuracy | Strict valid answers |
|---|---|---|
| Zero-shot prompt | 20% | 88.3% |
| LoRA fine-tuned | 35% | 41.7% |

## Interpretation and limitations

- Fine-tuning made the Oracle's answers **more useful for the game** but **worse at following the output format**. It often wrote "Based on the image…" instead of a single label. More label-only training examples or constrained decoding would be the next step.
- The evaluation is small: 20 games, so the jump from 20% to 35% is 4 to 7 games. Treat it as a pilot result, not a benchmark.
- The training log warns that PEFT was applied to an already-wrapped model, so adapters may have been stacked. A clean re-run should load the base model fresh before attaching LoRA.
