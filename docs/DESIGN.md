# vecq design notes

## Data model

A `Table` is a vector of `Column`s plus a row count. Each column holds exactly one physical array:

| Logical type | Storage | Notes |
|---|---|---|
| `INT64` | `std::vector<int64_t>` | NULL rows store 0 |
| `FLOAT64` | `std::vector<double>` | NULL rows store 0.0 |
| `STRING` | `std::vector<uint32_t>` codes + `std::vector<std::string>` dict | dictionary in first-appearance order |

NULLs are tracked in a separate validity byte vector (`0xFF` = present). The vector is **empty** when a column has no NULLs, and every operator checks `has_nulls()` so the common case pays nothing.

## CSV loading

```
mmap -> header -> sample-based inference -> split -> parallel parse -> promote? -> merge
```

- **Splitting.** The target is `threads * 8` chunks of at least 1 MB. If the body contains no `"` (checked with one `memchr`), each cut is the first `\n` after the target offset. Otherwise one sequential scan toggles an `in_quote` flag on every quote byte. An escaped `""` toggles twice, so it is handled for free. A cut is only made at a newline outside quotes.
- **Inference and promotion.** The first 4096 rows decide the initial types. During the full parse, any value that does not fit its column's type records the widest type it would need. After all chunks finish, columns that need widening are promoted and the whole file is re-parsed. There are at most two promotions per column, so at most three passes.
- **Dictionaries.** Each chunk builds a `string_view -> code` map whose views point into the mmap'd file (or into a chunk-owned deque, for unescaped quoted values). Merging walks the chunks in order to build the global dictionary, then remaps codes in parallel.

## Predicate evaluation

Predicates are bound once per query into a tree of `BPred` nodes:

| Source predicate | Bound form |
|---|---|
| `int_col op int` | `CmpI64` kernel |
| `int_col op 2.5` | rewritten exactly: `< 2.5` becomes `<= 2`, `> 2.5` becomes `>= 3`, `= 2.5` becomes constant false |
| `float_col op number` | `CmpF64` kernel |
| `str_col = 'x'` | `CmpU32` on the code of `'x'`, or constant false if `'x'` is not in the dictionary |
| `str_col < 'x'`, `LIKE`, `NOT LIKE` | per-dictionary-entry LUT, then `gather_u8` |
| `IS [NOT] NULL` | copy (or invert) the validity bytes |
| `AND` / `OR` | `mask_and` / `mask_or` of child masks |

After a leaf runs, the column's validity mask is ANDed in. Without `NOT`, treating NULL comparisons as false gives the same filter result as SQL three-valued logic. That is why general `NOT` is not supported yet: it would need a separate "unknown" mask.

## Kernels

All kernels take byte masks. The main NEON tricks:

- **Compare and narrow.** Eight `int64x2_t` compares produce sixteen 64-bit lane masks. Three levels of `vmovn` (64→32→16→8) pack them into one `uint8x16_t`, which is stored directly.
- **Mask widening for aggregates.** `vld1_u8` loads 8 mask bytes. `vmovl_s8`/`s16`/`s32` sign-extend them, which turns `0xFF` into an all-ones 64-bit lane. A plain AND with the value bits then zeroes unselected rows (0 is the additive identity), and `vbslq` substitutes ±inf for min/max.
- **Independent accumulators.** Sums and min/max keep four accumulators so successive `fadd`/`fmaxnm` instructions do not wait on each other.
- **NaN semantics.** Min/max use `vminnm`/`vmaxnm`, which ignore NaN operands. The scalar versions are written to match.

The portable versions use the same algorithms in plain C++ and avoid data-dependent branches. In particular, `m ? x : 0.0` is replaced by a bitwise select, because clang turns the ternary into a branch.

## Parallel execution

`parallel_for(tasks, threads, fn)` spawns workers that pull task ids from an atomic counter. Each task is one 4096-row batch. Per-worker state (scratch masks, selection vectors, accumulators, group tables) is indexed by worker id, so the hot loop has no sharing. After the scan, worker states are merged serially. Merge cost is proportional to the number of groups, not rows.

### Group tables

- **Dense:** used for a single `STRING` key. This is an `int32_t` array of size `|dict| + 1` (the last slot is NULL) that maps a code directly to a group id.
- **Hashed:** used otherwise. Keys are up to 4 `int64` slots (int value, dictionary code, or normalized double bits) plus a NULL bitmask, stored in an open-addressing linear-probing table at ≤ 50% load.

## What would make it faster

1. **Vectorized grouped aggregation.** For small group counts, keep per-group partial sums in registers or use radix-partitioned aggregation. Q3 and Q5 are dominated by the scalar `gids[k]` update loops.
2. **NEON tokenizer.** Find delimiters, quotes and newlines 16 bytes at a time with `vceqq_u8` and build bitmasks, simdjson-style. Single-threaded load is currently about 1.2 s for 434 MB.
3. **Late materialization for projections.** `SELECT * ... LIMIT` currently materializes `Value` variants row by row.
