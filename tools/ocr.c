// SPDX-License-Identifier: Apache-2.0
/*
 * ocr DET.axmodel REC.axmodel KEYS.txt IMAGE.rgb WIDTH HEIGHT: the lines of
 * text in a raw RGB image, each with its box, the detector's score and the
 * recognizer's confidence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ocr.h"

#define MAX_LINES 512

static void *read_file(const char *path, size_t *size)
{
	FILE *file = fopen(path, "rb");
	void *data = NULL;
	long length;

	if (!file)
		return NULL;
	if (!fseek(file, 0, SEEK_END) && (length = ftell(file)) > 0 && !fseek(file, 0, SEEK_SET)) {
		data = malloc(length);
		if (data && fread(data, 1, length, file) != (size_t)length) {
			free(data);
			data = NULL;
		}
		*size = length;
	}
	fclose(file);
	return data;
}

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
	size_t det_size = 0, rec_size = 0, keys_size = 0, image_size = 0;
	struct ocr_line *lines = calloc(MAX_LINES, sizeof(*lines));
	void *det, *rec, *keys, *image;
	struct ocr *ocr;
	char err[256];
	int width, height, n, i;
	double start;

	if (argc != 7) {
		fprintf(stderr, "usage: ocr DET.axmodel REC.axmodel KEYS.txt IMAGE.rgb WIDTH HEIGHT\n");
		return 2;
	}
	det = read_file(argv[1], &det_size);
	rec = read_file(argv[2], &rec_size);
	keys = read_file(argv[3], &keys_size);
	image = read_file(argv[4], &image_size);
	width = atoi(argv[5]);
	height = atoi(argv[6]);
	if (!det || !rec || !keys || !image || !lines || width < 1 || height < 1 ||
	    image_size != (size_t)width * height * 3) {
		fprintf(stderr, "cannot read the models, the dictionary or a %dx%d RGB image\n", width,
			height);
		return 1;
	}

	start = now_ms();
	ocr = ocr_load(det, det_size, rec, rec_size, keys, keys_size, err, sizeof(err));
	if (!ocr) {
		fprintf(stderr, "%s\n", err);
		return 1;
	}
	printf("loaded in %.1f ms\n", now_ms() - start);

	start = now_ms();
	n = ocr_read(ocr, image, width, height, lines, MAX_LINES);
	if (n < 0) {
		fprintf(stderr, "reading failed: %d\n", n);
		return 1;
	}
	for (i = 0; i < n; i++)
		printf("[%4d,%4d - %4d,%4d] %.2f %.2f  %s\n", lines[i].x0, lines[i].y0, lines[i].x1,
		       lines[i].y1, lines[i].score, lines[i].confidence, lines[i].text);
	printf("%d lines in %.1f ms\n", n, now_ms() - start);
	ocr_free(ocr);
	return 0;
}
