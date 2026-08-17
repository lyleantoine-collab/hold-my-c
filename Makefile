CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
TARGET = bench_axpby

all: $(TARGET)

$(TARGET): src/axpby.c bench/bench_axpby.c
	$(CC) $(CFLAGS) src/axpby.c bench/bench_axpby.c -o $(TARGET)

clean:
	rm -f $(TARGET)
