# How omlx-gate was built

Sources: the private development repository's git history (11 commits, all
on 2026-08-22), the code and tests in this repository, and the comments in
`pyproject.toml` and the modules. The earlier history of the gate inside the
oMLX fork is not in this repository; what is said about it below comes from
the extraction commit message. This public repository is a one-commit
snapshot.

## The problem

One Apple Silicon Mac serves local models to several clients on a local
network. Unified memory holds one or
two large models at a time. When several clients asked for big models at
once, requests queued until they timed out, or forced model swaps that cost
6-16 seconds each and could exhaust memory. oMLX's own memory guard then
answered with a low-level 507 that told the client nothing useful.

## First version

The gate was first written as a patch inside a fork of oMLX: three files
(`big_model_gate.py`, `big_model_gate_store.py`, `gate_logging.py`) and 29
tests. The fork was pinned at oMLX 0.6.0.dev1 and had fallen 168 commits
behind upstream, so every upstream improvement meant a rebase.

## Timeline (all 2026-08-22)

| Time | Step |
|------|------|
| 16:36 | **Extract.** The gate moved out of the fork into `omlx_gate/`, with oMLX as a pinned dependency. Proven on a scratch instance (a separate port, live server untouched): 29/29 tests pass against the same 0.6.0.dev1, the middleware attaches through `launch.py`, both startup tripwires fire, `/usage` answers, a 4B request passes ungated, the 4th model load in a minute gets `429 swap_rate_exceeded`, and SIGTERM releases the port. |
| 16:39 | **Re-pin to upstream.** oMLX is not on PyPI, so the pin is the `v0.6.2` git tag. Moving from 0.6.0.dev1 to v0.6.2 needed zero changes to `pool.py`: all four attach points exist upstream. Proven again on scratch against a clean environment, this time reading the real engine pool (not a mock). |
| 16:44 | **Cut over.** The live launch daemon was switched from the fork to `omlx-gate serve` with every flag unchanged; the fork was moved aside (not deleted) with a git bundle of both branches as a backup. Checks on the live endpoint after cutover: `/usage` policy block, wildcard bind, pool readable, 9 models discovered, small-model inference and a coding-agent request both succeed. |
| 17:30-20:13 | Seven notes commits recording measurements on the live stack: model load and decode speeds, prefix-cache hits on a coding-agent workload, tool-call reliability per model, and context ceilings. These measured numbers fed client settings elsewhere; they are not part of this package. |

## Key design decisions

- **Middleware, not a fork.** The only reason to fork oMLX was the gate. As
  ASGI middleware attached at launch, the gate needs no oMLX source changes,
  and an oMLX upgrade is a tag bump.
- **One coupling point.** Everything that reads oMLX internals lives in one
  function in `pool.py`, so an incompatible upstream change breaks one
  function, not the 864-line middleware. It never raises: an unreadable
  pool degrades `/usage` and residency checks instead of taking the server
  down.
- **`launch.py` does not reimplement `omlx serve`.** It imports oMLX's app,
  adds the middleware and hands the command line to oMLX's own CLI, so
  every serve flag and oMLX's own setup keep working.
- **Limits from measurement, not guesses.** Requests per minute come from
  an EWMA of real service time per model. Cold-load requests are excluded
  after one 66-second load-included sample produced a limit about 25 times
  too strict for a model that really answers in about 2.7 seconds.
- **Tell the client what to do.** Every gated response carries rate-limit
  headers and a suggested interval; 429s carry `Retry-After`; the 507 names
  the resident model and the swap commands; `/usage` states the full
  policy.
- **Pure ASGI.** No `BaseHTTPMiddleware`, so streamed tokens are not
  wrapped in an extra pipe layer.

## How AI was used

The extraction, re-pin and cutover were done in one session with Claude
Fable 5 as the coding agent (the commits carry its co-author line). The
working rule in the commit messages is "proven, not just written": each
step lists the checks that ran on a scratch instance before anything
touched the live server, and what was still open.

## Testing

29 pytest cases (24 test functions, some parametrised) cover size
classification, the rate formula, window expiry, each rejection with its
status code and headers, cold-load exclusion from service-time samples,
the swap guard and the `/usage` document. They run with FastAPI and httpx
only; oMLX is not needed.

## What did not work

- **Living on a fork.** 168 commits of upstream drift was the cost of
  keeping the gate inside oMLX; the extraction exists to end it.
- **Averaging every request.** Including cold loads in the service-time
  average made the published limit far too strict.
- **Running the interpreter directly from the launch daemon.** The bare
  interpreter could not import oMLX and the virtual environment's
  `python` is a symlink the daemon rejected; the installed console script
  was the one form that worked, proven with the exact daemon command on
  scratch first.
- **Hard restarts.** The notes record using a graceful stop rather than a
  forced restart, because of a known oMLX issue with forced restarts.

## Current state

v0.1.0, pinned to oMLX v0.6.2. The author's inference server was cut over
to it on 2026-08-22, and the code has not changed since
that day.
