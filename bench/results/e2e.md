| Query | sqlite | vecq neon 9T | vecq neon 1T | vecq scalar 1T | speedup vs sqlite (vecq neon 9T) |
|---|---|---|---|---|---|
| Q1 filter+count | 205.81 | 0.50 | 1.23 | 1.32 | 409.2x |
| Q2 filter+sum/avg | 305.13 | 1.73 | 5.20 | 12.85 | 176.9x |
| Q3 group by (5 groups) | 2009.71 | 4.67 | 37.51 | 39.22 | 430.4x |
| Q4 filter+group+top-k | 702.48 | 1.39 | 6.27 | 6.17 | 506.8x |
| Q5 string filter+2-key group | 2613.56 | 8.11 | 67.42 | 66.67 | 322.3x |

Median query time in ms over 7 runs. Load: sqlite `.import` 9.3 s; vecq neon 9T 0.22 s; vecq neon 1T 1.20 s; vecq scalar 1T 1.20 s.

Dataset: 10000000 rows, 433726102 bytes (434 MB). Threads: 9. Machine: Apple M5 Pro, macOS-27.0.1-arm64-arm-64bit, Apple clang version 21.0.0 (clang-2100.4.1.1), SQLite 3.54.0.
