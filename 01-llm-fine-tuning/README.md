# Fine-tuning Gemma 3 (270M) for Everyday Conversation

Supervised fine-tuning of `google/gemma-3-270m-it` on short everyday dialogues, comparing zero-shot, full fine-tuning and LoRA.

Notebook: [`gemma3_sft_vs_lora.ipynb`](gemma3_sft_vs_lora.ipynb)

## Setup

- **Data:** [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), `everyday-conversations` subset. 5% of the training split is held out for validation (seed 42), and the 119-example test split is used for final evaluation.
- **Formatting:** conversations are rendered with Gemma's chat template.
- **Evaluation:** token-level F1 and corpus BLEU (SacreBLEU) between generated and reference replies, plus the best and worst examples by F1.

| | Full fine-tuning | LoRA |
|---|---|---|
| Trainer | TRL `SFTTrainer` | TRL `SFTTrainer` + PEFT |
| Trainable weights | all | adapters on `q/k/v/o_proj` (r = 8, α = 16, dropout 0.05) |
| Learning rate | 2e-5, cosine, 3% warmup | 2e-4, cosine |
| Schedule | 1 epoch, effective batch 16, max length 256, bf16, gradient checkpointing | 1 epoch |

## Results (test split)

| Model | Token F1 | BLEU |
|---|---|---|
| Zero-shot (base model) | 22.90 | 4.25 |
| **Full fine-tuning** | **31.72** | **11.05** |
| LoRA (see caveat) | 23.24 | 3.80 |

After full fine-tuning, replies match the dataset's short, friendly style much more closely. For example, to the reference *"You're welcome! Have a great time in Paris!"* the fine-tuned model answers *"You're welcome! Enjoy your trip to Paris."*

## Caveat on the LoRA result

The LoRA number is **not a valid comparison**. In the notebook, the LoRA trainer received the same `model` object that full fine-tuning had already updated in place. The resulting adapter was then merged onto a *freshly loaded base model*. So the adapter learned a correction relative to the fine-tuned weights but was applied to the original weights, which explains why its score sits near zero-shot.

The fix is to reload the base model before LoRA training (or train LoRA in a separate run) and then compare again.
