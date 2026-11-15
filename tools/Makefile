# Command-line tools for trying the NPU on a device, outside the BEAM:
#   axrun: runs a model on files, compares outputs, times runs;
#   ocr:   reads the text in a raw RGB image.
# Cross-compile with CC set to the target's compiler (CC=<prefix>gcc make).

CFLAGS ?= -O2
CFLAGS += -Wall -Wextra -std=gnu11 -I../c_src

all: axrun ocr

axrun: axrun.c ../c_src/axnpu.c ../c_src/axnpu.h
	$(CC) $(CFLAGS) -o $@ axrun.c ../c_src/axnpu.c

ocr: ocr.c ../c_src/ocr.c ../c_src/ocr.h ../c_src/axnpu.c ../c_src/axnpu.h
	$(CC) $(CFLAGS) -o $@ ocr.c ../c_src/ocr.c ../c_src/axnpu.c -lm

clean:
	rm -f axrun ocr

.PHONY: all clean
