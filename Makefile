# Parallelism defaults to half the cores (min 2) to keep the machine responsive.
NPROC := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
J ?= $(shell n=$(NPROC); h=$$((n / 2)); [ $$h -lt 2 ] && h=2; echo $$h)
BUILD ?= build
CMAKE_FLAGS ?=
CLANG_FORMAT ?= uvx clang-format@23.1.3
SOURCES := $(shell find include src tools tests bench -name '*.cpp' -o -name '*.h')

.PHONY: build test bench bench-kernels bench-e2e lint format sanitize demo clean

build:
	cmake -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	cmake --build $(BUILD) -j$(J)

test: build
	ctest --test-dir $(BUILD) --output-on-failure -j$(J)

bench: bench-kernels bench-e2e

bench-kernels: build
	./$(BUILD)/vecq_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
	  --benchmark_min_time=0.2s --benchmark_out=bench/results/kernels.json --benchmark_out_format=json
	python3 scripts/kernels_table.py > bench/results/kernels.md

bench-e2e: build
	python3 bench/e2e.py --rows 10000000 --runs 7 --threads $(J) --out bench/results

lint:
	$(CLANG_FORMAT) --dry-run --Werror $(SOURCES)

format:
	$(CLANG_FORMAT) -i $(SOURCES)

sanitize:
	cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVECQ_SANITIZE=address,undefined -DVECQ_BUILD_BENCH=OFF
	cmake --build build-asan -j$(J)
	UBSAN_OPTIONS=halt_on_error=1 ./build-asan/vecq_tests
	cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVECQ_SANITIZE=thread -DVECQ_BUILD_BENCH=OFF
	cmake --build build-tsan -j$(J)
	./build-tsan/vecq_tests

demo: build
	@mkdir -p bench/data
	@[ -f bench/data/demo.csv ] || ./$(BUILD)/vecq-gen 2000000 1 > bench/data/demo.csv
	@sh scripts/demo.sh $(BUILD)/vecq bench/data/demo.csv

clean:
	rm -rf $(BUILD) build-asan build-tsan
