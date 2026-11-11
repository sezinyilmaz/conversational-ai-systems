# Spoken Dialogue Pipeline: Whisper → LLM → TTS

A speech-in, speech-out question-answering pipeline, plus a comparison of Whisper model sizes.

Notebook: [`spoken_dialogue_whisper.ipynb`](spoken_dialogue_whisper.ipynb)

## Pipeline

```
microphone (ipywebrtc) ──► Whisper ASR ──► retrieval + prompt ──► Qwen3-1.7B ──► gTTS ──► audio reply
                                     ▲                                │
                                     └──── dialogue history ◄─────────┘
```

- **ASR:** OpenAI Whisper (`base` in the live demo). Recorded WebM audio is passed to Whisper through ffmpeg.
- **Prompting:** the retrieval query combines the last few dialogue turns with the current question, so short follow-up questions keep their context. The system prompt asks for short answers, because the reply is spoken, and tells the model to say it does not know when the context lacks the answer.
- **Generation:** Qwen3-1.7B in bf16 with greedy decoding. Qwen's `<think>` blocks are stripped before speaking.
- **TTS:** gTTS, played back inside the notebook.

## Whisper model size vs. accuracy

Evaluated on 10 English utterances from the FLEURS test set (streamed to avoid downloading the full dataset), with Whisper's text normaliser applied before computing WER.

| Whisper model | WER |
|---|---|
| tiny | 13.68% |
| base | 9.43% |
| small | 7.08% |

Larger models make fewer word errors, at the cost of speed and memory. With only 10 utterances these numbers are indicative, not a benchmark.

## Known limitation: retrieval

The Contriever dense retriever loaded and indexed ~6M Wikipedia passages, but it returned unrelated passages for the test query. This points to a setup issue, possibly the tokenizer compatibility patch the retrieval code needed. To demonstrate the pipeline end to end, the notebook uses a simple title-based lookup instead.

**The retrieval step therefore does not work in this version.** The original demo also replaced the model's answer with a fixed sentence. That override has been removed here, and the answers it produced have been cleared from the outputs. Fixing the dense-retrieval setup is the obvious next step.
