
# Hotel-Front-desk-Virtual-Assistant-with-Voice

# Hotel Front Desk AI — NLP Assignment 3


A domain-restricted, conversational AI system for hotel front-desk operations.  
Guests interact through a real-time chat interface powered by a locally running LLM (Qwen 2.5-3B via Ollama), now with **voice support** (Moonshine ASR for speech-to-text, Piper TTS for text-to-speech).

---

## Table of Contents

1. [Architecture Diagram](#architecture-diagram)
2. [Setup Instructions](#setup-instructions)
3. [Model Selection](#model-selection)
4. [Performance Benchmarks](#performance-benchmarks)
5. [Running the Benchmark Tests](#running-the-benchmark-tests)
6. [Known Limitations](#known-limitations)
7. [Voice Features & Troubleshooting](#voice-features--troubleshooting)

---

## Architecture Diagram

<img width="1309" height="618" alt="Screenshot 2026-03-06 004831" src="https://github.com/user-attachments/assets/249aa007-bd12-429f-ad13-f051feb3f591" />



### Data Flow (one turn)

```
User types or speaks message
  │
  ▼
Frontend (websocketService.js)
  sends JSON → { session_id, message } (text or transcript)
  │
  ▼
Backend routes.py
  1. Validate inputs (Pydantic + custom checks)
  2. memory_manager.get_history(session_id)
  3. memory_manager.get_active_context()   ← last 6 turns (12 messages)
  4. prompt_builder.build_prompt()         ← system prompt + history + user msg
  5. ollama_client.generate() / generate_stream()
  6. clean_greeting_from_response()        ← strip repeated hellos
  7. memory_manager.add_message()          ← persist both turns
  8. Return reply via WebSocket / REST (text)
  │
  ▼
Frontend renders streaming tokens in MessageDisplay
  │
  ▼
If voice enabled: TTS (Piper) streams audio chunks to frontend for playback
```

---


## Installation

### Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Python | 3.8+ | Backend runtime |
| Node.js | 16+ | Frontend build tool (Vite) |
| Ollama | Latest | Local LLM host |
| ffmpeg | Latest | Audio processing |
| Piper TTS | .onnx model | Local text-to-speech |

---

### 1. Install Ollama

- Download and install from [https://ollama.com](https://ollama.com)
- Verify it is running:
  ```bash
  ollama list
  ```

### 2. Create the custom model

- From the project root (where `Modelfile` lives):
  ```bash
  ollama create hotel-qwen -f Modelfile
  ollama list  # Should show: hotel-qwen
  ```

### 3. Backend Setup

- Install Python dependencies:
  ```bash
  cd backend
  pip install -r requirements.txt
  ```
- (Optional, for voice) Set Piper TTS model path (Windows example):
  ```powershell
  $env:PIPER_MODEL_PATH = "C:\\models\\en_US-lessac-medium.onnx"
  ```

### 4. Frontend Setup

- Install Node.js dependencies:
  ```bash
  cd frontend
  npm install
  ```

### 5. Additional Tools

- Ensure ffmpeg is installed and available in your PATH.
- Download a Piper TTS .onnx model and set the environment variable as above.

---

## Usage

### Start Backend
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Verify:
- REST health: [http://localhost:8000/health](http://localhost:8000/health)
- WebSocket endpoint: `ws://localhost:8000/ws/chat`
- Voice WebSocket endpoint: `ws://localhost:8000/ws/voice_chat`

### Start Frontend
```bash
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser. You should see a green **Connected** status badge and voice controls (Record/Upload Audio).

### (Optional) CLI testing
```bash
# From project root
python main.py
```
This runs a terminal-based conversation loop against the same backend modules — useful for model/prompt debugging without starting the web stack.

### Docker (alternative)
```bash
cd backend
docker-compose up --build
```
See [backend/DOCKER_README.md](backend/DOCKER_README.md) for details.

---

## Model Selection

### Why Qwen 2.5-3B?

| Consideration | Detail |
|---------------|--------|
| **Size** | 3 billion parameters — runnable on a mid-range laptop CPU with 8 GB RAM |
| **Language quality** | Strong instruction-following with coherent multi-turn dialogue |
| **Latency** | 1–5 s per response on CPU after the model is loaded (GPU is faster) |
| **Domain restriction** | Responds well to a hard system-prompt boundary; refuses off-topic queries reliably |
| **Local / offline** | No API keys, no data sent to the cloud — suitable for assignment/demo use |

### Alternatives considered

| Model | Parameters | Trade-off |
|-------|-----------|-----------|
| `llama3.2:1b` | 1B | Faster but weaker instruction-following |
| `mistral:7b` | 7B | Better quality, but requires ≥16 GB RAM |
| `qwen2.5:7b` | 7B | Better quality, higher hardware requirement |
| OpenAI GPT-4o | — | Best quality, but requires internet & paid API key |

`qwen2.5:3b` is the best balance of **quality, speed, and hardware accessibility** for a course assignment running on commodity hardware.

### Modelfile parameters explained

```
FROM qwen2.5:3b        # base checkpoint

PARAMETER num_ctx      1300   # max tokens in context window
PARAMETER num_predict   90   # max new tokens per response (keeps answers concise)
PARAMETER temperature   0.35   # higher = more varied, lower = more deterministic
PARAMETER top_p         0.82   # nucleus sampling threshold
PARAMETER num_thread      0   # 0 = use all available CPU threads
```

---

## Performance Benchmarks

The following results were measured on a **mid-range laptop (CPU-only inference)** running the full stack locally.

### Latency (5 sequential requests, single session)

| Metric | Value |
|--------|-------|
| Requests sent | 5 |
| Successes | 4 |
| Failures | 1 (cold-start timeout) |
| Min latency | 15 160 ms |
| Max latency | 64 172 ms |
| Mean latency | 27 931 ms |
| Std deviation | 20 375 ms |

> The first request timed out because Ollama was loading the model weights into RAM — this is a **one-time cold-start cost**. Requests 2–5 succeeded in 15–20 s each, which is typical for CPU-only qwen2.5:3b inference.

### Stress test (concurrent users)

| Concurrent users | Successes | Failures | Mean latency | Max latency | Min latency |
|-----------------|----------|---------|-------------|------------|------------|
| 2  | 2  | 0 | 14 207 ms | 18 028 ms | 10 387 ms |
| 4  | 4  | 0 | 21 582 ms | 35 156 ms |  8 657 ms |
| 6  | 6  | 0 | 33 238 ms | 55 611 ms |  9 587 ms |
| 8  | 8  | 0 | 36 834 ms | 66 883 ms |  8 128 ms |
| 10 | 10 | 0 | 48 568 ms | 89 120 ms |  8 617 ms |

> The system handled all 10 concurrent requests without a single failure. Mean latency grows linearly with concurrency — expected because Ollama serialises requests behind a single CPU inference thread. No crashes or dropped connections were observed at any level.

### Failure handling

| Edge case | Expected HTTP | Actual HTTP | Result |
|-----------|--------------|------------|--------|
| Empty message string | 422 | 422 | ✔ Pass |
| Whitespace-only message | 400 | 400 | ✔ Pass |
| Missing `message` field | 422 | 422 | ✔ Pass |
| Missing `session_id` field | 422 | 422 | ✔ Pass |
| Empty `session_id` | 422 | 400 | ✘ Minor mismatch — custom validator fires before Pydantic |
| Oversized message (10 001 chars) | 200 | 200 | ✔ Pass |
| SQL injection in message | 200 | 200 | ✔ Treated as plain text |
| JSON injection in `session_id` | 200 | 200 | ✔ Stored as plain string |
| GET on non-existent route | 404 | 404 | ✔ Pass |
| GET /health | 200 | 200 | ✔ Pass |
| GET /api/chat (wrong method) | 405 | 405 | ✔ Pass |
| Malformed JSON body | 422 | 422 | ✔ Pass |

> 11 of 12 tests passed. The one mismatch (empty `session_id` returning 400 instead of 422) is a minor ordering difference between the custom validator and Pydantic — the request is still correctly rejected.

---

## Running the Benchmark Tests

A self-contained test script is provided at [tests/benchmark_tests.py](tests/benchmark_tests.py).

### Install test dependencies

```bash
pip install requests httpx
```

### Run all three suites

```bash
# Backend must be running first
python tests/benchmark_tests.py
```

### Run individual suites

```bash
# Latency only (10 sequential messages)
python tests/benchmark_tests.py --latency --requests 10

# Stress test only (ramp to 20 concurrent users, step 4)
python tests/benchmark_tests.py --stress --max-users 20 --step 4

# Failure-handling only
python tests/benchmark_tests.py --failure
```

### What each suite measures

#### 1 · Latency Benchmarking
Sends `--requests` sequential messages to `/api/chat` using a single session  
and reports **min / max / mean / std-dev** response time in milliseconds.

```
  Metric                    Value
  ──────────────────────────────────
  Requests sent                 5
  Successes                     5
  Failures                      0
  Min latency (ms)           1823
  Max latency (ms)           4201
  Mean latency (ms)          2640
  Std dev (ms)                901
```

#### 2 · Stress Testing
Fires `N` simultaneous async requests at each concurrency level and prints  
a table of success count, failure count, mean/max/min latency.

```
  Users    Success    Fail   Mean(ms)     Max(ms)    Min(ms)
  ────────────────────────────────────────────────────────────
  2        ✔ 2        0      3012         3891       2133
  4        ✔ 4        0      6890         9203       4411
  6        ⚠ 5        1      13204        18902      5021
  10       ✘ 8        2      25103        42011      6301
```

#### 3 · Failure Handling
Sends 12 intentionally malformed requests and checks each returns the  
expected HTTP status code, verifying the API is hardened against bad input.

```
  Empty message string                          HTTP 422 (expected 422)  ✔
  Whitespace-only message                       HTTP 400 (expected 400)  ✔
  Missing 'message' field                       HTTP 422 (expected 422)  ✔
  ...
  Result: 12/12 failure-handling tests passed.
```

---



## Known Limitations

1. **Single-threaded LLM inference:** Ollama runs one request at a time on CPU. All concurrent requests queue behind each other, causing latency to grow linearly with the number of simultaneous users.
2. **In-memory session storage:** Conversation history is stored in process memory and is lost when the backend restarts. There is no persistent database.
3. **Domain restriction is prompt-only:** The hotel-only restriction is enforced through the system prompt. Prompt injection could bypass it; there is no secondary classifier or guardrail.

---

## Voice Features & Troubleshooting

### Voice Setup (Fully Local, No API Services)
The backend runs Moonshine ASR and Piper TTS locally. No external cloud APIs or microservices are needed.

**Install backend dependencies (including Moonshine/Piper):**

```bash
cd backend
pip install -r requirements.txt
```

**Set Piper environment variable (Windows example):**
```powershell
$env:PIPER_MODEL_PATH = "C:\\models\\en_US-lessac-medium.onnx"
```

**Voice Test Flow:**
1. Click **Record**, speak, then click **Stop**.
2. The transcript appears as a user message.
3. Assistant text streams token-by-token.
4. Audio response auto-plays as chunks arrive.
5. Use **Upload Audio** to test non-microphone input.

### Troubleshooting

**"Not Connected" Error:**
- Wait 2-3 seconds after page load
- Refresh the page
- Check backend is running: http://localhost:8000/health
- Check browser console (F12) for errors

**"Connection Lost" Error:**
- Click the **Reconnect** button
- Check if backend is still running
- Restart the backend server if needed

**No Response from Assistant:**
- Check Ollama is running: `ollama list`
- Check backend logs for errors about Ollama connection
- Test Ollama directly: `ollama run hotel-qwen "Hello, how are you?"`
- First response is slow (model loading)
- Check backend terminal for errors

**Slow First Response:**
- This is normal! The first message loads the model into memory (10-30 seconds). Subsequent messages are fast (1-3 seconds).

**Port Already in Use:**
- Free port 8000 or 3000 using `netstat` and `taskkill` (Windows) or `lsof` (Linux/Mac)

### Status Indicators

- 🟢 **Green dot + "Connected"** = All good!
- 🔴 **Red dot + "Disconnected"** = Backend not reachable
- **Typing...** = Assistant is generating response
- **Blinking cursor (▊)** = Response streaming in real-time
- **Timestamp** = Message completed

---

