# Engineering conventions

Durable practices for this codebase, not tied to one incident. Contrast
with `docs/decisions.md`, which is a chronological log of what happened
and why -- this doc is what to keep doing going forward. Add a new
section here when a decisions.md entry reveals a pattern worth repeating,
not every time something gets fixed.

## Prefer a real distributed Spark job over a driver-side Python loop, when the computation can be expressed that way (2026-09-17)

Gold notebooks 02/03 went through three attempts against real backfill
volume before landing here (full story in docs/decisions.md): a plain
`.collect()` hung the driver on memory; `toLocalIterator()` avoided that
but likely re-triggered the underlying shuffle per partition, for which
the standard fix (`.cache()`) turned out to be **unsupported on
serverless compute** (`PERSIST TABLE is not supported on serverless`,
worth remembering for any future in-memory-caching idea in this repo).
All three attempts shared the same flaw: routing 10M+ rows through a
single Python process on the driver, no matter how carefully chunked or
streamed, is fighting the tool.

The fix that actually worked: `silver_df.groupBy("token_id")
.applyInPandas(fn, schema=...)` -- push the per-group Python computation
onto the cluster as a real, distributed Spark operation. Each executor
handles a subset of tokens in parallel, calling into the same pure-Python
`gold_features.py` logic; no driver memory pressure, no caching, and
Spark's own groupBy shuffle guarantees every row for a token lands
together (no hand-rolled ordering/streaming logic to get right, and no
`.cache()` needed to make it fast). A `pandas.DataFrame` group has no
guaranteed row order, so the per-group function must sort locally first
-- cheap, since a group is one token's rows, not the whole table.

**When a notebook needs to run non-trivial per-group Python logic (not
just a `filter`/`join`/aggregation Spark already has a native op for)
against a dataset large enough that "collect it to the driver" is a real
question, reach for `groupBy(...).applyInPandas(...)` (or an equivalent
Spark-native distributed op) before reaching for `toLocalIterator()` or
`.collect()` plus a manual Python loop.** It scales with the cluster
instead of the driver, and its Tasks X/Y progress in the Databricks UI is
an *honest* signal of real completion -- unlike a driver-side loop, where
that same UI can show "done" while the real work continues invisibly
afterward (exactly what made the 02/03 stalls hard to diagnose the first
two times).

## Progress logging when work genuinely runs on the driver (2026-09-17)

The applyInPandas pattern above removes the need for hand-rolled progress
prints in Gold's case (Spark's own task progress covers it now), but
plenty of this codebase's work still legitimately runs as a driver-side
Python loop -- Bronze/Silver's chunked backfill and catch-up notebooks,
for one, where each chunk does real I/O (API calls, writes) that doesn't
map cleanly onto a single Spark job. **Any such loop over a collection
whose size isn't small and fixed must log progress periodically, not just
a final summary** -- this became a hard rule after repeated "is it stuck
or just slow" incidents in one session where a cell was "still running"
with zero signal either way.

- **Chunked jobs** (a bounded list of date-range/batch chunks, each doing
  real work) -- print `[i/total] <chunk description>: <result>` once per
  chunk. Already the standard in `00c_bronze_backfill_dune.py` and
  `01_silver_token_trades.py`'s catch-up mode; keep doing this for any new
  chunked backfill/catch-up notebook.
- **Tight loops over a large flat scan that genuinely has to run on the
  driver** -- print every N items (pick N so it fires at least a few
  times a minute at expected throughput) with a running count and elapsed
  time:

  ```python
  _PROGRESS_EVERY = 500_000
  _progress_start = time.time()
  for i, item in enumerate(large_iterable, start=1):
      ...
      if i % _PROGRESS_EVERY == 0:
          print(f"... {i:,} processed ({time.time() - _progress_start:.0f}s elapsed)")
  ```

Where a cheap independent total is available (a `.count()`, a known
row/chunk count), compare it against what the loop actually processed at
the end and print a `WARNING` if they don't match -- catches a silent
partial-scan, not just a slow one.

This isn't about making output noisier for its own sake -- it's the
difference between "check back in 5 minutes and see if the counter moved"
and "interrupt, add print statements, redeploy, re-run from scratch" to
answer the exact same question.

## Restart Python before importing local `src/` modules on a warm cluster/session (2026-09-18)

A notebook's `%pip install` cell only guarantees fresh code for
pip-installed packages. Anything imported via `sys.path.append("../src")`
(this repo's `features.*`/`backtest.*`/`models.*` modules) is a plain
Python import, not a pip package -- on an interactive cluster/session
that stays warm across notebook reruns, Python caches those modules in
`sys.modules` the first time they're imported and does **not** re-read
them from disk on a later run, even after a real code change lands and is
redeployed. This produced a genuinely confusing incident (see
decisions.md, 2026-09-18): a real fix (`winsorize_return`) landed on
disk, was independently confirmed present via a fresh import in a scratch
cell, and yet re-running the full notebook still produced byte-identical
output to before the fix -- because the notebook's own `import
models.entry_model` (etc.) was silently serving a cached pre-fix module
object from an earlier run in that same session, not the updated file.

**Any notebook that imports from this repo's `src/` should call
`dbutils.library.restartPython()` once, before those imports, not just
after `%pip install`.** This guarantees a clean interpreter -- and
therefore a clean `sys.modules` -- on every run, regardless of whether
anything pip-related changed. Cheap to do unconditionally; the failure
mode it prevents is expensive to diagnose (identical output across
multiple "redeploy and re-run" cycles looks exactly like the fix didn't
take effect, when it's actually a caching artifact one layer up).

## A dead Spark session needs a detach/re-attach, not just a re-run (2026-09-18)

If a notebook crashes mid-run with `INACTIVITY_TIMEOUT_AND_FAILED_TO_RESTORE`
(the compute session went idle long enough to be torn down -- e.g. a Dune
chunk paginating for minutes with zero Spark activity, see decisions.md
2026-09-18) or `NO_ACTIVE_SESSION` on a later cell, the notebook's `spark`
object is permanently stale for the rest of that attached session --
re-running cells from where it crashed will NOT create a fresh one.

**Detach and re-attach the cluster (or "Reconnect," whatever the UI calls
it for your compute type) first, then re-run the WHOLE notebook from the
top cell** -- not just the cell that failed. Every notebook-level variable
(`spark`, `client`, secrets fetched via `dbutils.secrets.get`, etc.) needs
fresh initialization after reattaching. This is safe to do freely for any
notebook that uses a resumable progress table (00c's pattern) -- already-
`success` chunks are skipped on the re-run, so nothing gets re-paid-for
except whatever chunk was actually in flight when the session died.
