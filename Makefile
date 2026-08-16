CC ?= cc

CFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra \
	-Wno-unused-function -Wno-unused-variable -fobjc-arc
LDLIBS := -lm -pthread -framework Foundation -framework Metal
METAL_SRCS := $(wildcard metal/*.metal)

.PHONY: all clean test

all: ds4 ds4-server

ds4: qwen36.o qwen_frontend.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-server: ds4
	ln -sf ds4 $@

qwen36.o: qwen36.c qwen_frontend.h ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_frontend.o: qwen_frontend.m qwen_frontend.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

ds4_metal.o: ds4_metal.m ds4.h ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ $<

test: ds4 ds4-server
	./tests/smoke.sh

clean:
	rm -f ds4 ds4-server qwen36.o qwen_frontend.o ds4_metal.o
