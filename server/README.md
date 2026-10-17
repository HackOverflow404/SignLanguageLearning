# Sentence engine server

The server behind the phone app's sentence mode. A small open LLM writes a short
English practice sentence; the fail-closed gloss rule engine decides whether it
becomes an ASL gloss sequence. The LLM only ever touches English.

- `POST /sentence {"target": "want_2"}`: a sentence that uses the target sign,
  accepted by the engine and judged sensible by a second model (503 when none
  passed this time).
- `POST /gloss {"english": "I want water."}`: the engine's gloss sequence, or its
  refusal (`in_scope: false` with a `reason`).
- `GET /health`

Gloss sequences are rule-engine output, not yet reviewed by a fluent Deaf signer.

## How sentences are made

- **Writer** (`llm.py`): Qwen2.5-3B-Instruct, run with llama.cpp on the CPU. A
  grammar built from the engine's own lexicon restricts it to curriculum words
  (plus inflections and the function words the engine drops) and forces the
  target sign's word into the sentence.
- **Gate**: `sentence_prompts.sentence_prompt_maybe_llm`, unchanged from the
  desktop: only engine-accepted sentences survive.
- **Sense check**: Qwen2.5-1.5B-Instruct judges each survivor ("does this make
  sense as something a person would really say?", log-odds of yes over no); below
  `JUDGE_MIN` it is dropped. The engine checks structure, not meaning, so without
  this a grammar-forced small model's "My father is a school." would pass. On 304
  new hand-labelled outputs it separates sensible from nonsense with AUC 0.96; at
  the cutoff, 96% of what it passes is sensible.
- **Pool**: a background thread keeps 3 finished sentences per sign, so a request
  is usually answered instantly; an empty pool means generating on demand.

No API token or paid inference: both models run in-process.

## Code layout

`app.py` imports the repo's own engine modules. `build.sh` assembles the Docker
build context into `server/build/app/`: the server files, `curriculum.yaml`, and
`gloss_rules.py`, `sentence_prompts.py` and `_hf_client.py` copied unchanged (with
empty package `__init__`s, so none of the heavy desktop dependencies load).

```bash
server/build.sh
docker build -t aslcv-sentence server/build/app
docker run --rm aslcv-sentence pytest -q                    # server tests
docker run --rm -p 7860:7860 aslcv-sentence                 # local server
docker run --rm --cpus=2 aslcv-sentence python eval_generator.py --per-sign 3
```

Quality doesn't depend on the hardware, so evaluate on a GPU when one is around:
a CUDA build of llama-cpp-python in a throwaway venv, with `LLM_GPU_LAYERS=-1`.
Measure latency on the VM itself. Writer comparison (3 sentences per sign, all 60
signs, share on target and judged sensible): 0.5B 22%, 1.5B 35%, 3B 62%.

## Hosting: Oracle Cloud Always Free

A free ARM VM (2 cores, 12 GB) runs the server under Docker Compose behind Caddy,
which gets a Let's Encrypt certificate for the VM's IP as an
[sslip.io](https://sslip.io) name (e.g. `129-146-1-2.sslip.io`), since iOS needs
HTTPS. No domain is needed.

One-time console steps (cloud.oracle.com):

1. **Sign up.** The home region is permanent; free ARM capacity varies by region.
2. **Create an instance** (Compute → Instances → Create):
   - Image: Canonical Ubuntu 24.04.
   - Shape: Ampere `VM.Standard.A1.Flex`, **2 OCPUs, 12 GB memory**: the whole
     Always Free A1 allowance (1,500 OCPU-hours and 9,000 GB-hours a month,
     as of 2026-10). Anything larger is billed.
   - Boot volume: the default (~50 GB) is inside the 200 GB free block storage.
   - Networking: the VCN `aslcv-vcn`, made first with Networking → Virtual
     cloud networks → Start VCN Wizard → "Create VCN with Internet
     Connectivity" (creating it inside the instance form leaves the public IP
     switch disabled); its public subnet, "Assign a public IPv4 address" on.
   - SSH keys: upload a public key, e.g. `~/.ssh/id_ed25519_personal.pub`.
   - "Out of capacity" means no free ARM hosts right now: retry later or pick
     another availability domain. `server/deploy/launch_vm.sh` does the
     retrying: with the oci CLI configured (`uv tool install oci-cli`, an API
     key added under My profile → API keys, `~/.oci/config`) and the VCN
     made first, it creates this exact instance as soon as any availability
     domain has room, and refuses to run if an instance already exists.
3. **Open ports 80 and 443**: the instance's subnet → its Security List → Add
   Ingress Rules: source `0.0.0.0/0`, TCP, destination port `80`; again for `443`.

Then, from the repo:

```bash
export SSH_KEY=~/.ssh/id_ed25519_personal              # the key you uploaded
server/deploy/deploy.sh ubuntu@<public-ip> --setup   # first time: Docker + firewall
server/deploy/deploy.sh ubuntu@<public-ip>           # every later update
```

It builds the image on the VM (llama.cpp compiles for ARM, ~10 minutes the first
time), starts it, and waits for `https://<ip-with-dashes>.sslip.io/health`.
Logs: `ssh ubuntu@<ip> 'cd aslcv-server && docker compose logs -f app'`.

Oracle may reclaim an Always Free instance that is idle for 7 days (95th-percentile
CPU, network and memory all under 20%). Once every sign's pool is full this server
mostly idles, so a reclaimed VM is possible; `deploy.sh --setup` on a new one
rebuilds it.

Rate limits: 30 `/sentence` calls per client per 10 minutes and 1000 per day
(`PER_CLIENT_LIMIT`, `DAILY_LIMIT`).
