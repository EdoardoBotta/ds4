#!/usr/bin/env python3
"""Compare DS4's streaming expert cache with an offline-optimal policy.

The Metal trace format is a headerless stream of six little-endian int32
expert IDs.  One record is written for each routed layer, in layer order.  The
simulator treats a layer's six experts as one atomic request: they must coexist
while that layer executes and therefore protect one another from eviction.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


GIB = 1024**3
INF_NEXT_USE = 1 << 60


@dataclass(frozen=True)
class Trace:
    path: Path
    layers: int
    experts: int
    selected: int
    groups: tuple[tuple[int, ...], ...]

    @property
    def tokens(self) -> int:
        return len(self.groups) // self.layers

    @property
    def accesses(self) -> int:
        return len(self.groups) * self.selected


@dataclass(frozen=True)
class PolicyStats:
    hits: int
    misses: int

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


def load_trace(path: Path, layers: int, experts: int, selected: int) -> Trace:
    raw = path.read_bytes()
    record_bytes = selected * 4
    if not raw or len(raw) % record_bytes:
        raise ValueError(
            f"{path}: size {len(raw)} is not a positive multiple of "
            f"{record_bytes} bytes"
        )
    records = len(raw) // record_bytes
    if records % layers:
        raise ValueError(
            f"{path}: {records} layer records is not divisible by {layers} layers"
        )

    values = struct.unpack(f"<{records * selected}i", raw)
    groups: list[tuple[int, ...]] = []
    for group_index in range(records):
        start = group_index * selected
        ids = tuple(values[start : start + selected])
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"{path}: duplicate expert in record {group_index}: {ids}"
            )
        for expert in ids:
            if not 0 <= expert < experts:
                raise ValueError(
                    f"{path}: expert {expert} in record {group_index} is outside "
                    f"0..{experts - 1}"
                )
        groups.append(ids)
    return Trace(path, layers, experts, selected, tuple(groups))


def group_keys(trace: Trace, group_index: int) -> tuple[int, ...]:
    layer = group_index % trace.layers
    return tuple(layer * trace.experts + expert for expert in trace.groups[group_index])


def simulate_current(trace: Trace, capacity: int, decay_tokens: int = 16) -> PolicyStats:
    """Mirror the Metal route-hotness policy for a cold resident cache."""
    if capacity < trace.selected:
        raise ValueError(f"capacity {capacity} cannot hold one {trace.selected}-expert request")

    universe = trace.layers * trace.experts
    hotness = [0] * universe
    last_used = [0] * universe
    version = [0] * universe
    resident = bytearray(universe)
    resident_keys: set[int] = set()
    victim_heap: list[tuple[int, int, int, int]] = []
    clock = 0
    decode_tokens = 0
    decay_at_token = 0
    hits = 0
    misses = 0

    def push(key: int) -> None:
        heapq.heappush(
            victim_heap,
            (hotness[key], last_used[key], version[key], key),
        )

    def rebuild_heap() -> None:
        victim_heap.clear()
        for key in resident_keys:
            version[key] += 1
            push(key)

    def evict(protected: set[int]) -> None:
        deferred: list[tuple[int, int, int, int]] = []
        victim = -1
        while victim_heap:
            item = heapq.heappop(victim_heap)
            _hot, _last, item_version, key = item
            if not resident[key] or item_version != version[key]:
                continue
            if key in protected:
                deferred.append(item)
                continue
            victim = key
            break
        for item in deferred:
            heapq.heappush(victim_heap, item)
        if victim < 0:
            raise RuntimeError("no evictable current-policy entry")
        resident[victim] = 0
        resident_keys.remove(victim)
        version[victim] += 1

    for group_index in range(len(trace.groups)):
        layer = group_index % trace.layers
        if layer == 0:
            decode_tokens += 1
            if decay_at_token == 0:
                decay_at_token = decode_tokens
            decayed = False
            while decode_tokens - decay_at_token >= decay_tokens:
                for key in range(universe):
                    hotness[key] >>= 1
                decay_at_token += decay_tokens
                decayed = True
            if decayed:
                rebuild_heap()

        keys = group_keys(trace, group_index)
        protected = set(keys)

        # Production records route hotness before looking up the selected set.
        for key in keys:
            hotness[key] += 1
            if resident[key]:
                version[key] += 1
                push(key)

        missing: list[int] = []
        for key in keys:
            if resident[key]:
                hits += 1
                clock += 1
                last_used[key] = clock
                version[key] += 1
                push(key)
            else:
                misses += 1
                missing.append(key)

        # The Metal loader protects the whole six-expert selection while it
        # reserves slots and loads all misses in parallel.
        for key in missing:
            if len(resident_keys) >= capacity:
                evict(protected)
            resident[key] = 1
            resident_keys.add(key)
            clock += 1
            last_used[key] = clock
            version[key] += 1
            push(key)

    return PolicyStats(hits, misses)


def simulate_optimal(trace: Trace, capacity: int) -> PolicyStats:
    """Batch-aware Belady MIN, the cold-cache offline miss lower bound."""
    if capacity < trace.selected:
        raise ValueError(f"capacity {capacity} cannot hold one {trace.selected}-expert request")

    universe = trace.layers * trace.experts
    future: list[list[int]] = [[] for _ in range(universe)]
    keys_by_group: list[tuple[int, ...]] = []
    for group_index in range(len(trace.groups)):
        keys = group_keys(trace, group_index)
        keys_by_group.append(keys)
        for key in keys:
            future[key].append(group_index)

    cursor = [0] * universe
    next_use = [INF_NEXT_USE] * universe
    version = [0] * universe
    resident = bytearray(universe)
    resident_keys: set[int] = set()
    # Negative next-use makes heapq return the farthest future request first.
    victim_heap: list[tuple[int, int, int]] = []
    hits = 0
    misses = 0

    def push(key: int) -> None:
        heapq.heappush(victim_heap, (-next_use[key], version[key], key))

    def evict(protected: set[int]) -> None:
        deferred: list[tuple[int, int, int]] = []
        victim = -1
        while victim_heap:
            item = heapq.heappop(victim_heap)
            neg_next, item_version, key = item
            if (
                not resident[key]
                or item_version != version[key]
                or -neg_next != next_use[key]
            ):
                continue
            if key in protected:
                deferred.append(item)
                continue
            victim = key
            break
        for item in deferred:
            heapq.heappush(victim_heap, item)
        if victim < 0:
            raise RuntimeError("no evictable optimal-policy entry")
        resident[victim] = 0
        resident_keys.remove(victim)
        version[victim] += 1

    for group_index, keys in enumerate(keys_by_group):
        protected = set(keys)
        missing: list[int] = []

        # Advance every requested key to its next request before choosing a
        # victim; this is the information available to the offline oracle.
        for key in keys:
            pos = cursor[key]
            if pos >= len(future[key]) or future[key][pos] != group_index:
                raise RuntimeError("future-use index is inconsistent")
            pos += 1
            cursor[key] = pos
            next_use[key] = future[key][pos] if pos < len(future[key]) else INF_NEXT_USE
            if resident[key]:
                hits += 1
                version[key] += 1
                push(key)
            else:
                misses += 1
                missing.append(key)

        for key in missing:
            if len(resident_keys) >= capacity:
                evict(protected)
            resident[key] = 1
            resident_keys.add(key)
            version[key] += 1
            push(key)

    return PolicyStats(hits, misses)


def simulate_all(traces: list[Trace], capacities: list[int], expert_bytes: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for capacity in capacities:
        current_hits = current_misses = optimal_hits = optimal_misses = 0
        for trace in traces:
            current = simulate_current(trace, capacity)
            optimal = simulate_optimal(trace, capacity)
            current_hits += current.hits
            current_misses += current.misses
            optimal_hits += optimal.hits
            optimal_misses += optimal.misses

        accesses = current_hits + current_misses
        if accesses != optimal_hits + optimal_misses:
            raise RuntimeError("policy access totals differ")
        if optimal_misses > current_misses:
            raise RuntimeError("offline optimum produced more misses than current policy")
        current_rate = current_hits / accesses if accesses else 0.0
        optimal_rate = optimal_hits / accesses if accesses else 0.0
        avoidable = current_misses - optimal_misses
        rows.append(
            {
                "capacity_experts": capacity,
                "capacity_gib": capacity * expert_bytes / GIB,
                "accesses": accesses,
                "current_hits": current_hits,
                "current_misses": current_misses,
                "current_hit_rate": current_rate,
                "optimal_hits": optimal_hits,
                "optimal_misses": optimal_misses,
                "optimal_hit_rate": optimal_rate,
                "hit_rate_gap_pp": (optimal_rate - current_rate) * 100.0,
                "avoidable_misses": avoidable,
                "avoidable_miss_pct": 100.0 * avoidable / current_misses if current_misses else 0.0,
                "avoidable_io_gib": avoidable * expert_bytes / GIB,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))


def write_svg(path: Path, rows: list[dict[str, float | int]], title: str) -> None:
    width, height = 1040, 650
    left, right, top, bottom = 92, 35, 92, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["capacity_gib"]) for row in rows]
    log_xs = [math.log2(max(x, 1e-9)) for x in xs]
    x_min, x_max = min(log_xs), max(log_xs)
    if x_min == x_max:
        x_max = x_min + 1.0

    def px(x: float) -> float:
        return left + (math.log2(max(x, 1e-9)) - x_min) / (x_max - x_min) * plot_w

    def py(percent: float) -> float:
        return top + (100.0 - percent) / 100.0 * plot_h

    current = [(px(float(r["capacity_gib"])), py(float(r["current_hit_rate"]) * 100.0)) for r in rows]
    optimal = [(px(float(r["capacity_gib"])), py(float(r["optimal_hit_rate"]) * 100.0)) for r in rows]
    gap = [(px(float(r["capacity_gib"])), py(float(r["hit_rate_gap_pp"]))) for r in rows]

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#263238}",
        ".grid{stroke:#dfe5e8;stroke-width:1}.axis{stroke:#607d8b;stroke-width:1.3}",
        ".current{fill:none;stroke:#1565c0;stroke-width:3}.optimal{fill:none;stroke:#2e7d32;stroke-width:3}",
        ".gap{fill:none;stroke:#ef6c00;stroke-width:3;stroke-dasharray:8 5}.dot{stroke:white;stroke-width:1.5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fafcfd"/>',
        f'<text x="{left}" y="36" font-size="22" font-weight="600">{escape(title)}</text>',
        f'<text x="{left}" y="62" font-size="13" fill="#546e7a">Cold start; current route-hotness LFU (16-token decay) vs batch-aware Belady MIN</text>',
    ]

    for tick in range(0, 101, 10):
        y = py(float(tick))
        out.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        out.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{tick}%</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')

    # Label powers of two and the final point on the logarithmic x axis.
    tick_values: list[float] = []
    lo_power = math.ceil(math.log2(min(xs)))
    hi_power = math.floor(math.log2(max(xs)))
    tick_values.extend(2.0**power for power in range(lo_power, hi_power + 1))
    if not tick_values or abs(tick_values[-1] - xs[-1]) / xs[-1] > 0.08:
        tick_values.append(xs[-1])
    for value in tick_values:
        x = px(value)
        out.append(f'<line class="axis" x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}"/>')
        label = f"{value:g}"
        out.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12">{label}</text>')

    out.extend(
        [
            f'<path class="current" d="{svg_path(current)}"/>',
            f'<path class="optimal" d="{svg_path(optimal)}"/>',
            f'<path class="gap" d="{svg_path(gap)}"/>',
        ]
    )
    for _cls, color, points in (
        ("current", "#1565c0", current),
        ("optimal", "#2e7d32", optimal),
        ("gap", "#ef6c00", gap),
    ):
        for x, y in points:
            out.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x, legend_y = left + 18, top + 20
    for offset, label, color, dashed in (
        (0, "Current hit rate", "#1565c0", ""),
        (190, "Optimal hit rate", "#2e7d32", ""),
        (375, "Gap (percentage points)", "#ef6c00", ' stroke-dasharray="8 5"'),
    ):
        x = legend_x + offset
        out.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 30}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dashed}/>')
        out.append(f'<text x="{x + 38}" y="{legend_y + 4}" font-size="12">{label}</text>')

    peak = max(rows, key=lambda row: float(row["hit_rate_gap_pp"]))
    peak_x = px(float(peak["capacity_gib"]))
    peak_y = py(float(peak["hit_rate_gap_pp"]))
    out.append(
        f'<text x="{peak_x + 8:.2f}" y="{max(top + 14, peak_y - 10):.2f}" font-size="12" '
        f'fill="#e65100">peak gap {float(peak["hit_rate_gap_pp"]):.2f} pp</text>'
    )
    out.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" text-anchor="middle" font-size="14">Expert cache capacity (GiB, log₂ scale)</text>')
    out.append(f'<text transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="14">Hit rate / hit-rate gap</text>')
    out.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def parse_cache_gib(spec: str, expert_bytes: int) -> list[int]:
    capacities: list[int] = []
    for item in spec.split(","):
        gib = float(item.strip())
        if not math.isfinite(gib) or gib <= 0:
            raise ValueError(f"invalid cache GiB value: {item!r}")
        capacity = int(gib * GIB // expert_bytes)
        if capacity <= 0:
            raise ValueError(f"cache size {gib:g} GiB holds no complete expert")
        capacities.append(capacity)
    return sorted(set(capacities))


def self_test() -> None:
    groups = ((0,), (1,), (2,), (0,), (1,), (2,))
    trace = Trace(Path("self-test"), 1, 3, 1, groups)
    current = simulate_current(trace, 2)
    optimal = simulate_optimal(trace, 2)
    assert current == PolicyStats(0, 6), current
    assert optimal == PolicyStats(2, 4), optimal
    assert simulate_current(trace, 3) == PolicyStats(3, 3)
    assert simulate_optimal(trace, 3) == PolicyStats(3, 3)
    print("cache-policy self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", type=Path, help="raw DS4 selected-ID trace(s)")
    parser.add_argument("--layers", type=int, default=43, help="routed layers per token (Flash: 43)")
    parser.add_argument("--experts", type=int, default=256, help="experts per layer (Flash: 256)")
    parser.add_argument("--selected", type=int, default=6, help="experts selected per routed layer")
    parser.add_argument("--expert-bytes", type=int, default=7_077_888, help="bytes in one complete routed expert")
    parser.add_argument(
        "--cache-gib",
        default="1,2,4,8,12,16,24,27,32,40,48,56,64,72,73",
        help="comma-separated cache capacities in GiB",
    )
    parser.add_argument("--csv", type=Path, default=Path("cache_policy_results.csv"))
    parser.add_argument("--svg", type=Path, default=Path("cache_policy_gap.svg"))
    parser.add_argument("--title", default="SSD streaming expert-cache policy gap")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.traces:
        parser.error("at least one trace is required (or use --self-test)")
    if args.layers <= 0 or args.experts <= 0 or args.selected <= 0 or args.expert_bytes <= 0:
        parser.error("layers, experts, selected, and expert-bytes must be positive")

    traces = [load_trace(path, args.layers, args.experts, args.selected) for path in args.traces]
    capacities = parse_cache_gib(args.cache_gib, args.expert_bytes)
    if capacities[0] < args.selected:
        parser.error("the smallest cache cannot hold one selected-expert group")

    for trace in traces:
        print(
            f"trace {trace.path}: {trace.tokens} tokens, {len(trace.groups)} layer groups, "
            f"{trace.accesses} expert accesses"
        )
    rows = simulate_all(traces, capacities, args.expert_bytes)
    write_csv(args.csv, rows)
    write_svg(args.svg, rows, args.title)

    print("cache_GiB  experts  current_hit  optimal_hit  gap_pp  avoidable_misses")
    for row in rows:
        print(
            f"{float(row['capacity_gib']):9.2f}  {int(row['capacity_experts']):7d}  "
            f"{float(row['current_hit_rate']):11.2%}  {float(row['optimal_hit_rate']):11.2%}  "
            f"{float(row['hit_rate_gap_pp']):6.2f}  {int(row['avoidable_misses']):16d}"
        )
    print(f"wrote {args.csv}")
    print(f"wrote {args.svg}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
