# Dialogue Manager for Clarification under Noisy Input

A lightweight, rule-based dialogue manager (DM) placed in front of an LLM restaurant-booking agent. The DM decides **when to ask for clarification and how**, instead of leaving it to the prompt.

Notebook: [`dialogue_manager_clarification.ipynb`](dialogue_manager_clarification.ipynb)

## Why

LLM agents tend to guess when the user's input is vague, misspelled (for example from speech recognition) or outside the known options, and that is where hallucinated bookings come from. Here, every user turn first passes through the DM. The LLM only continues when the slot values are reliable.

## How it works

```
user text ──► slot normalisation ──► safe state update ──► clarification policy ──┬─► follow-up / clarification reply
                                                                                    └─► Llama 3.1 8B (Ollama) + JSON tool calls
                                                                                        (search / book / cancel against a small DB)
```

1. **Slot normalisation** for area, food and price range. Each slot is matched as *exact* (confidence 1.0), *substring* (0.9), *fuzzy* (difflib similarity over 1–3-gram spans, cut-off 0.72) or *unknown*. Unknown (mentioned but not recognisable) is kept separate from missing (not mentioned).
2. **Clarification policy**, in priority order:
   - unknown or out-of-vocabulary value → **explicit** clarification that lists the valid options
   - fuzzy match with confidence < 0.84 → **explicit** ("I think you said centre for area. Is that right?")
   - fuzzy match with confidence 0.84–0.94 → **implicit** (carry the value forward and ask the next question)
   - slot not mentioned → **follow-up** question
3. **Safe state update.** Only exact or substring matches are written to the dialogue state. Fuzzy values wait for confirmation, and a known value is never overwritten by an unknown one.
4. **Tool-calling agent.** Llama 3.1 8B (4-bit, served locally with Ollama) outputs one JSON action per turn. Search, booking and cancellation results come from the database rather than the model.

## Results

The notebook's evaluation suite has 12 cases in four groups: missing slots, ambiguous spellings, out-of-vocabulary values and simulated ASR noise.

| Group | Correct |
|---|---|
| Missing slot → follow-up | 3 / 3 |
| Ambiguous spelling → implicit | 1 / 2 |
| Out-of-vocabulary → explicit | 3 / 3 |
| Noisy ASR | 4 / 4 |
| **Overall** | **11 / 12 (91.7%)** |

The single miss is instructive. *"Find me something in the senter"* was expected to trigger implicit clarification, but its fuzzy score fell below the 0.84 threshold, so the DM asked explicitly. Meanwhile *"i want a restarant in the senter"* is expected to trigger explicit clarification. The same misspelling should get different treatment depending on context, which a single confidence threshold cannot capture.

## Limitations

- It relies on hand-written slot vocabularies and string similarity, so negation, self-corrections and paraphrases outside the vocabulary are not handled.
- Thresholds were set by hand on a small test set.
