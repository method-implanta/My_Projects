.PHONY: all clean

all: distances

distances: main.c
	gcc -o distances main.c -O2 -fopenmp -lm

clean:
	rm -f distances