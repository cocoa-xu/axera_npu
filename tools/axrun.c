// SPDX-License-Identifier: Apache-2.0
/*
 * axrun MODEL.axmodel [INPUT.bin ...] [-- REFERENCE.bin ...]: runs a model on
 * the open runtime, writes each output to axout-<n>.bin and, given reference
 * outputs (from the vendor's engine), says whether they match byte for byte.
 * AXRUN_LOOPS=N times N more runs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "axnpu.h"

static void *read_file(const char *path, size_t *size)
{
	FILE *file = fopen(path, "rb");
	void *data = NULL;
	long length;

	if (!file)
		return NULL;
	if (!fseek(file, 0, SEEK_END) && (length = ftell(file)) >= 0 && !fseek(file, 0, SEEK_SET)) {
		data = malloc(length ? length : 1);
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

static void describe(const char *what, int i, const struct axnpu_tensor *t)
{
	int d;

	printf("%s %d \"%s\": dtype %d [", what, i, t->name, t->dtype);
	for (d = 0; d < t->ndims; d++)
		printf("%s%lld", d ? ", " : "", (long long)t->dims[d]);
	printf("] %zu bytes\n", t->size);
}

int main(int argc, char **argv)
{
	struct axnpu_model *model;
	char err[256];
	int i, arg, ret, loops, ref = 0, mismatches = 0;
	size_t size;
	void *data;
	double start;

	if (argc < 2) {
		fprintf(stderr, "usage: axrun MODEL.axmodel [INPUT.bin ...] [-- REFERENCE.bin ...]\n");
		return 2;
	}
	data = read_file(argv[1], &size);
	start = now_ms();
	model = data ? axnpu_load(data, size, err, sizeof(err)) : NULL;
	if (!model) {
		fprintf(stderr, "%s: %s\n", argv[1], data ? err : "cannot read");
		return 1;
	}
	printf("loaded in %.1f ms\n", now_ms() - start);
	free(data);

	for (i = 0; i < axnpu_num_inputs(model); i++)
		describe("input", i, axnpu_input(model, i));
	for (i = 0; i < axnpu_num_outputs(model); i++)
		describe("output", i, axnpu_output(model, i));

	for (arg = 2, i = 0; arg < argc && strcmp(argv[arg], "--"); arg++, i++) {
		struct axnpu_tensor *t = axnpu_input(model, i);
		void *input = read_file(argv[arg], &size);

		if (!t || !input || size != t->size) {
			fprintf(stderr, "%s: want %zu bytes\n", argv[arg], t ? t->size : 0);
			return 1;
		}
		memcpy(t->data, input, size);
		free(input);
	}

	start = now_ms();
	ret = axnpu_run(model, 5000);
	printf("run: %d in %.2f ms\n", ret, now_ms() - start);

	loops = getenv("AXRUN_LOOPS") ? atoi(getenv("AXRUN_LOOPS")) : 0;
	if (!ret && loops > 0) {
		double best = 1e9, total = 0, took;

		for (i = 0; i < loops && !ret; i++) {
			start = now_ms();
			ret = axnpu_run(model, 5000);
			took = now_ms() - start;
			total += took;
			best = took < best ? took : best;
		}
		printf("%d runs: %d, mean %.3f ms, best %.3f ms\n", loops, ret, total / loops, best);
	}

	if (arg < argc)
		ref = arg + 1;
	for (i = 0; i < axnpu_num_outputs(model); i++) {
		struct axnpu_tensor *t = axnpu_output(model, i);
		char name[32];
		FILE *file;

		snprintf(name, sizeof(name), "axout-%d.bin", i);
		file = fopen(name, "wb");
		if (file) {
			fwrite(t->data, 1, t->size, file);
			fclose(file);
		}
		if (ref && ref + i < argc) {
			void *expected = read_file(argv[ref + i], &size);
			int same = expected && size == t->size && !memcmp(expected, t->data, size);

			printf("output %d vs %s: %s\n", i, argv[ref + i], same ? "identical" : "DIFFERENT");
			mismatches += !same;
			free(expected);
		}
	}
	axnpu_free(model);
	return ret || mismatches ? 1 : 0;
}
