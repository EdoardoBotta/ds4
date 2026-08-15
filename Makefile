CC ?= cc

CFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra \
	-Wno-unused-function -Wno-unused-variable -fobjc-arc
LDLIBS := -lm -pthread -framework Foundation -framework Metal
METAL_SRCS := $(wildcard metal/*.metal)

.PHONY: all clean test

all: ds4

ds4: qwen36.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36.o: qwen36.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ $<

ds4_metal.o: ds4_metal.m ds4.h ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ $<

test: ds4
	./tests/smoke.sh

clean:
	rm -f ds4 qwen36.o ds4_metal.o
