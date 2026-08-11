# SSD streaming cache-policy gap experiment

This experiment records the routed-expert requests made by the repository's
canonical `ds4-bench` workload and compares two equal-capacity, cold caches:

- **Current:** the production global route-hotness policy, including its
  16-token decay, LRU tie-break, and protection of the current six experts.
- **Optimal:** batch-aware Belady MIN. It knows the future and evicts the
  resident expert whose next request is farthest away. This is an unattainable
  oracle, but it is the correct miss lower bound for the captured workload.

One miss reads one complete routed expert. For the checked-in Flash IQ2/Q2 GGUF
that is 7,077,888 bytes (6.75 MiB), so the output reports both expert slots and
GiB.

Run on a macOS host where Metal is available:

```sh
zsh experiments/ssd-cache-policy/run_experiment.zsh
```

The runner uses `speed-bench/promessi_sposi.txt`, pre-fills to 2,048 tokens,
then records the benchmark's 128-token decode probe. Long prefill uses the
batch path and is not included in the selected-ID trace; the comparison is
therefore specifically the resident-cache policy for decode. The trace should
contain 128 × 43 layer records and 33,024 individual expert accesses.

Outputs are written under `experiments/ssd-cache-policy/results/`:

- `cache_policy_results.csv`: misses, hit rates, gap, and avoidable I/O by size.
- `cache_policy_gap.svg`: current hit rate, oracle hit rate, and their gap.
- `promessi_2048_gen128.log`: live cache statistics for cross-checking.
- `promessi_2048_gen128_bench.csv`: the normal benchmark throughput row.

The 27 GiB sweep point is exactly 4,096 experts, matching the cache used while
recording. Its cold-cache simulated counters can therefore be checked against
the live summary in the log. The 73 GiB endpoint holds all 43 × 256 routed
experts and should reduce the policy gap to zero.

The selected-expert route is independent of cache contents, so one trace can
be replayed offline at every capacity. The cold start deliberately excludes
the built-in popularity preload from both policies; this isolates eviction
quality instead of mixing it with startup placement quality.

To repeat the run with another repository workload, set both `PROMPT` and a
distinct `OUT` directory. Multiple raw traces may also be passed to the
analyzer; each starts cold and their hit/miss counts are aggregated.

This is a miss-count and logical-I/O experiment, not a direct throughput
oracle. It intentionally does not model SSD latency overlap, read coalescing,
or temporary in-flight GPU protection. Check the simulated 27 GiB counters
against the live log before using small differences to justify a policy change.

The analyzer itself has no third-party dependencies. Its small deterministic
policy check can be run without a model:

```sh
python3 experiments/ssd-cache-policy/analyze_cache_policy.py --self-test
```
