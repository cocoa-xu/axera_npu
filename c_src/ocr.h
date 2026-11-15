/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Text detection and recognition with PP-OCR models (DB detector, CTC
 * recognizer) on the NPU, for screens: lines are axis-aligned boxes.
 */
#ifndef OCR_H
#define OCR_H

#include <stddef.h>
#include <stdint.h>

#define OCR_TEXT_SIZE 1024

struct ocr_line {
	int x0, y0, x1, y1;	/* inclusive, in the image */
	float score;		/* the detector's mean probability over the line */
	float confidence;	/* the recognizer's mean probability over its characters */
	char text[OCR_TEXT_SIZE];	/* UTF-8 */
};

struct ocr;

/*
 * Loads the detector and recognizer (.axmodel bytes) and the recognizer's
 * dictionary (one character per line); on failure NULL and a reason in err.
 */
struct ocr *ocr_load(const void *det, size_t det_size, const void *rec, size_t rec_size,
		     const char *keys, size_t keys_size, char *err, size_t err_size);

void ocr_free(struct ocr *ocr);

/*
 * The lines of text in an RGB image (width x height x 3 bytes), top to bottom
 * and left to right: how many (at most max), or a negative errno.
 */
int ocr_read(struct ocr *ocr, const uint8_t *rgb, int width, int height, struct ocr_line *lines,
	     int max);

#endif
