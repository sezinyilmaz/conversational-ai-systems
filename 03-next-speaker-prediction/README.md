# Next-Speaker Prediction in Multi-Party Meetings

Predicting who will speak next in a four-person meeting, from the transcript and speaker history.

Notebook: [`next_speaker_bilstm.ipynb`](next_speaker_bilstm.ipynb)

## Data

- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) manual annotations, downloaded by the notebook. Words are merged into turns (same speaker, gap ≤ 0.5 s).
- 147 meetings with exactly four speakers, giving ~196K examples, each with an 8-turn history.
- Split **by meeting** (80/10/10, seed 42) so no meeting appears in more than one split: 161,108 / 13,389 / 21,662 examples.

## Model

A hierarchical BiLSTM:

1. **Utterance encoder:** word embeddings (128) → BiLSTM (128 per direction) → one vector per turn.
2. Each history turn's vector is concatenated with a **speaker embedding** (16-d).
3. **Context encoder:** a BiLSTM over the 8 history turns plus the current turn.
4. An optional **current-speaker embedding** is added, then an MLP classifies into 4 speakers.

Training: Adam (lr 3e-5), ReduceLROnPlateau on dev macro-F1, early stopping (patience 5), gradient clipping 1.0, batch size 16.

## Results and ablations

| Setting | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **Full model** (history + history speakers + current speaker) | 0.697 | **0.712** | **0.711** |
| History speakers only (no current speaker) | 0.666 | 0.686 | 0.685 |
| No speaker information | 0.259 | 0.272 | 0.259 |
| No dialogue history | 0.242 | 0.322 | 0.216 |

Chance level is about 0.25. The notebook's saved outputs are from the *history speakers only* run, while the other rows come from re-running it with the flags changed.

## What the ablations show

- **Who spoke before matters far more than what was said.** Without speaker information the model is at chance level, even though it still sees all the text.
- The sequence of previous speakers alone gets close to the full model (0.685 vs 0.711).
- Without history, the model almost never predicts one of the speakers, because the current utterance alone is too little signal.
- Errors cluster on very short turns ("uh", "yeah", fragments) and fast overlapping exchanges. Timing, pauses, prosody or gaze would likely help more than a bigger text model.
