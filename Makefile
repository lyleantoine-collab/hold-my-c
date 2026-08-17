CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
TARGET = bench_axpby_simd

all: $(TARGET)

$(TARGET): src/axpby_simd.c bench/bench_axpby_simd.c
	$(CC) $(CFLAGS) src/axpby_simd.c bench/bench_axpby_simd.c -o $(TARGET)

clean:
	rm -f $(TARGET) bench_axpby
