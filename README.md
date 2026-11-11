# Conversational AI Systems

Five projects covering the main components of a modern dialogue system: fine-tuning a small LLM, controlling an LLM agent with a dialogue manager, modelling multi-party turn-taking, multimodal dialogue agents and a spoken (speech-in, speech-out) pipeline.

Built during the *Conversational Agents & Dialogue Systems* module of the MSc Artificial Intelligence at Queen Mary University of London (2025–26).

| # | Project | What it shows | Key result |
|---|---|---|---|
| 01 | [LLM fine-tuning](01-llm-fine-tuning/) | Full supervised fine-tuning and LoRA on Gemma 3 (270M) for chat | Token-level F1 **22.9 → 31.7**, BLEU **4.3 → 11.1** over zero-shot |
| 02 | [Dialogue manager](02-dialogue-manager/) | Confidence-based clarification policy in front of a Llama 3.1 8B tool-calling agent | **11/12** correct clarification decisions on noisy and out-of-vocabulary input |
| 03 | [Next-speaker prediction](03-next-speaker-prediction/) | Hierarchical BiLSTM on the AMI meeting corpus, with ablations | **0.71 macro-F1** across 4 speakers (chance ≈ 0.25) |
| 04 | [Visual dialogue agents](04-visual-dialogue-agents/) | Questioner and Oracle agents with a vision-language model (Qwen3.5-0.8B), Oracle tuned with LoRA | Game accuracy 20% → 35% on a small 20-game evaluation |
| 05 | [Spoken dialogue](05-spoken-dialogue/) | Whisper ASR → LLM → text-to-speech, with Whisper model-size comparison | WER **13.7% → 9.4% → 7.1%** (tiny → base → small) |

Each folder has its own README covering the approach, results and limitations, plus the notebook with its original run outputs.

## Tech stack

PyTorch · Hugging Face Transformers · TRL (`SFTTrainer`) · PEFT / LoRA · OpenAI Whisper · Ollama (Llama 3.1 8B) · Qwen3 / Qwen3.5 · gTTS · scikit-learn · pandas

## Notes

- The notebooks started from module-provided lab templates (data loading, scaffolding and task descriptions). The implementations, experiments and analysis are mine.
- Datasets and model weights are not included. Every notebook downloads its public data (SmolTalk, AMI, GuessWhat?!, FLEURS) or explains where it comes from.
- Limitations are documented honestly in each project README. In particular, the LoRA comparison in project 01 and the retrieval step in project 05 have known issues.
