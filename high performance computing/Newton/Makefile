.PHONY: all clean

all: newton

newton: main.c
	gcc main.c -o newton -Ofast -mavx2 -march=native -flto -funroll-loops -fomit-frame-pointer -fopenmp -lm

clean:
	rm -f newton