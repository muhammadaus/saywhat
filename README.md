# saywhat

**Voice input for Hermes that asks instead of guessing.**

Dictation tools write down their best guess and move on. That works for "remind me to buy milk".
It doesn't work for "apply the kubectl manifest to the EKS cluster, then rerun the Terraform plan
with pgvector enabled", which Whisper hears as *"apply the **cubectal** manifest to the **x** cluster…
with **vector** enabled"*. An agent that acts on that guess runs the wrong command.

saywhat runs speech recognition locally and **measures how sure it is of every word**. When it isn't
sure, the agent asks you before acting. It offers whole sentences first, then individual words if
none of them is right. Your answers teach saywhat your vocabulary, so it asks less over time.

```
you (voice):  "Apply the kubectl manifest to the EKS cluster, then rerun the Terraform plan with pgvector enabled."
heard:        "Apply the cubectal manifest to the x cluster, … with vector enabled."   cubectal 54% · x 34% · vector 30%

agent ─ saywhat: Which did you say?
        ❯ Apply the kubectl manifest to the EKS cluster, … with pgvector enabled. (Recommended)
          Apply the cubectal manifest to the EKS cluster, … with pgvector enabled.
          Apply the kubectl manifest to the x cluster, … with pgvector enabled.
          None of these: let me pick word by word
          Other (type your answer)
```

Choose "word by word" and you get one question per uncertain word (`"x"` → `EKS · ECS · x`). Every
pick is learned. After three confirmations a mishearing is fixed silently. In testing, the same
recording went from three uncertain words to zero questions over a few turns.

## How it works

| Piece | What it does |
|---|---|
| **STT provider `saywhat`** | Local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with per-word probabilities. Words below the confidence threshold become *uncertain spans*. Each span gets ranked candidates from four sources: your learned mishearings; sound-alike matches against your dictionary and a seed lexicon of ~165 technical terms (letter-name aware, so "x" → EKS); a second **re-listen** pass biased toward those candidates; and the words exactly as heard. |
| **`pre_transcription` hook** | Puts your personal vocabulary into the recognizer prompt, for any STT provider. |
| **`pre_llm_call` hook** | When your message is a fresh dictation with uncertain words, tells the agent to confirm via `clarify` first: sentence choices, then word choices. Typed messages and confident dictation pass through untouched. |
| **`post_tool_call` hook** | Reads your `clarify` answers and learns *heard → meant* pairs. It only learns about words saywhat itself flagged, so a confused or prompt-injected model can't rewrite your dictionary. |
| **Practice mode** | Say a sentence you want to get across. saywhat compares what you meant with what a *neutral* listener (no personal vocabulary) heard, gives word-by-word verdicts and concrete tips ("spell it out: E. K. S."; "the 'pg' sound didn't come through"), plays a slow model pronunciation, and tracks your score across attempts. |
| **Speaking profile** | Practice history and your hardest words. Those words feed back into everyday dictation, so recognition adapts to *your* accent instead of asking you to lose it. |

## Install

```bash
hermes plugins install muhammadaus/saywhat --enable
hermes config set stt.provider saywhat
```

Then use Hermes voice mode as usual (default <kbd>Ctrl</kbd>+<kbd>B</kbd> to record). Voice messages on
Telegram, Discord, Slack and the other gateways go through the same provider.

The first transcription downloads the Whisper model you choose (`base` by default, ~150 MB).

## Use

**Talking to the agent:** nothing to learn. Speak, and when saywhat is unsure you get a choice.

**Practice:**

```
/saywhat practice Apply the kubectl manifest to the EKS cluster
   … record yourself (voice mode); each recording is scored and coached …
/saywhat practice stop
```

You can also just ask: *"I want to practise saying 'PostgreSQL replication slot'"* (the
`saywhat_practice` tool).

**Your dictionary:**

```
/saywhat                          status: terms, automatic fixes, hardest words
/saywhat teach engine x = nginx   always turn "engine x" into nginx
/saywhat forget nginx
```

Or ask the agent ("add Supabase to my voice dictionary"). It uses the `saywhat_dictionary` tool.
A fix the *model* teaches is offered first but only becomes automatic once you confirm it.

**Progress:** ask "how is my pronunciation improving?" (the `saywhat_progress` tool).

## Settings

`plugins.entries.saywhat.settings` in `config.yaml`, or the plugin settings form in Hermes Desktop:

| key | default | meaning |
|---|---|---|
| `whisper_model` | `base` | `tiny` … `large-v3`, `distil-large-v3`, `turbo` |
| `language` | auto | BCP-47 hint, e.g. `en`, `ja` |
| `device` | `auto` | `auto`, `cpu`, `cuda` |
| `threshold` | `0.6` | ask about words the recognizer is less sure of than this |
| `auto_correct_after` | `3` | confirmations before a mishearing is fixed silently |
| `max_candidates` | `4` | choices per uncertain word |
| `match_window_seconds` | `300` | how long a recording stays matched to your next message |

## Privacy and what it touches

- **Audio stays local.** Transcription runs on your machine with faster-whisper. saywhat makes no
  network calls of its own. On first use, faster-whisper downloads the model you picked from
  Hugging Face (`huggingface.co`) into the Hugging Face cache.
- **Dictated turns carry extra context to your LLM.** For an uncertain dictation, the confidence
  report and candidate sentences are added to your message, so they reach whichever model provider
  Hermes uses.
- **Model pronunciations use your configured Hermes TTS** (`text_to_speech`), which may be a cloud
  service; the practice sentence is sent to it. In an interactive terminal the audio is played with
  the system player (`afplay`, `ffplay`, `paplay` or `aplay`) as a short detached process. Nothing
  is played on headless gateway hosts.
- **Stored data** (profile-scoped, in Hermes `plugin-data`): your dictionary and learned
  mishearings, the most recent transcript with word timings (never audio), practice scores and
  hardest words. Remove a term with `/saywhat forget`.
- **Files read:** only an `audio_path` you or the agent pass to `saywhat_practice`, and only if it
  is an existing local audio file. URLs and other FFmpeg inputs are refused.
- No credentials, no telemetry, no shell commands besides the audio player above.

## Development

```bash
# tests import Hermes's transcription provider ABC, so run them from a Hermes checkout's venv
<hermes-agent>/venv/bin/python -m pip install pytest "faster-whisper>=1.1.0,<2"
<hermes-agent>/venv/bin/python -m pytest -q tests
hermes plugins validate . --install-deps
```

`tests/test_provider_audio.py` synthesizes real speech with macOS `say` and runs it through
faster-whisper. It skips elsewhere.

## License

MIT
