// SPDX-License-Identifier: Apache-2.0
/*
 * Detection (DB): the image is scaled into the detector's input, its
 * probability map thresholded, and each connected region becomes a box,
 * grown back to the text's full height ("unclip").
 *
 * Recognition: each box is cut out, scaled to the recognizer's height and
 * read with greedy CTC. A line wider than the recognizer's aspect ratio is
 * read in pieces, cut between letters near even cuts; a gap as wide as a
 * quarter of the text's height is a word space.
 */
#include "ocr.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "axnpu.h"

#define DET_THRESHOLD 0.3f
#define BOX_THRESHOLD 0.6f
#define UNCLIP_RATIO 1.5f
#define INK_THRESHOLD 96	/* summed |r - bg| + |g - bg| + |b - bg| */
#define MAX_BOXES 512
#define RUN_TIMEOUT_MS 5000

struct model {
	struct axnpu_model *npu;
	struct axnpu_tensor *input, *output;
	int height, width;	/* of the NHWC input */
};

struct image {
	const uint8_t *rgb;
	int width, height;
};

struct box {
	int x0, y0, x1, y1;
	float score;
};

struct ocr {
	struct model det, rec;
	char *keys_text;
	char **keys;	/* 1-based as the recognizer counts: blank, the dictionary, space */
	int nkeys;
	int *stack, *label;	/* the detector's map, for its regions */
	struct box boxes[MAX_BOXES];
};

static int fail(char *err, size_t size, const char *what)
{
	snprintf(err, size, "%s", what);
	return -1;
}

static int model_load(struct model *model, const void *data, size_t size, int output_dims,
		      char *err, size_t err_size)
{
	model->npu = axnpu_load(data, size, err, err_size);
	if (!model->npu)
		return -1;
	model->input = axnpu_input(model->npu, 0);
	model->output = axnpu_output(model->npu, 0);
	if (axnpu_num_inputs(model->npu) != 1 || model->input->dtype != AXNPU_UINT8 ||
	    model->input->ndims != 4 || model->input->dims[3] != 3 ||
	    model->output->dtype != AXNPU_FLOAT || model->output->ndims != output_dims)
		return fail(err, err_size, "not a PP-OCR model: inputs or outputs differ");
	model->height = (int)model->input->dims[1];
	model->width = (int)model->input->dims[2];
	return 0;
}

static const float *model_run(struct model *model)
{
	return axnpu_run(model->npu, RUN_TIMEOUT_MS) ? NULL : model->output->data;
}

/*
 * Bilinear resize of the image's rectangle [x0, x0 + w) x [y0, y0 + h) into
 * dst (dst_w x dst_h, a row stride of `stride` pixels).
 */
static void resize(const struct image *image, float x0, float y0, float w, float h, uint8_t *dst,
		   int dst_w, int dst_h, int stride)
{
	float sx = w / dst_w, sy = h / dst_h;
	int x, y, c;

	for (y = 0; y < dst_h; y++) {
		float fy = y0 + (y + 0.5f) * sy - 0.5f;
		int iy = (int)floorf(fy);
		float wy = fy - iy;
		int ya = iy < 0 ? 0 : iy >= image->height ? image->height - 1 : iy;
		int yb = iy + 1 < 0 ? 0 : iy + 1 >= image->height ? image->height - 1 : iy + 1;

		for (x = 0; x < dst_w; x++) {
			float fx = x0 + (x + 0.5f) * sx - 0.5f;
			int ix = (int)floorf(fx);
			float wx = fx - ix;
			int xa = ix < 0 ? 0 : ix >= image->width ? image->width - 1 : ix;
			int xb = ix + 1 < 0 ? 0 : ix + 1 >= image->width ? image->width - 1 : ix + 1;
			const uint8_t *p00 = image->rgb + ((size_t)ya * image->width + xa) * 3;
			const uint8_t *p01 = image->rgb + ((size_t)ya * image->width + xb) * 3;
			const uint8_t *p10 = image->rgb + ((size_t)yb * image->width + xa) * 3;
			const uint8_t *p11 = image->rgb + ((size_t)yb * image->width + xb) * 3;
			uint8_t *d = dst + ((size_t)y * stride + x) * 3;

			for (c = 0; c < 3; c++) {
				float top = p00[c] + (p01[c] - p00[c]) * wx;
				float bottom = p10[c] + (p11[c] - p10[c]) * wx;

				d[c] = (uint8_t)(top + (bottom - top) * wy + 0.5f);
			}
		}
	}
}

/* Connected regions of the thresholded map, as boxes in the image. */
static int detect(struct ocr *ocr, const struct image *image)
{
	struct model *det = &ocr->det;
	int w = det->width, h = det->height, n = 0, i;
	float scale = fminf((float)w / image->width, (float)h / image->height);
	int sw = (int)(image->width * scale), sh = (int)(image->height * scale);
	const float *prob;

	memset(det->input->data, 0, det->input->size);
	resize(image, 0, 0, (float)image->width, (float)image->height, det->input->data,
	       sw > 0 ? sw : 1, sh > 0 ? sh : 1, w);
	prob = model_run(det);
	if (!prob)
		return -EIO;

	memset(ocr->label, 0, sizeof(int) * w * h);
	for (i = 0; i < w * h && n < MAX_BOXES; i++) {
		int top = 0, x0 = w, y0 = h, x1 = -1, y1 = -1, count = 0;
		float sum = 0, bw, bh, grow;
		struct box *b = &ocr->boxes[n];

		if (ocr->label[i] || prob[i] <= DET_THRESHOLD)
			continue;
		ocr->label[i] = 1;
		ocr->stack[top++] = i;
		while (top) {
			int p = ocr->stack[--top], px = p % w, py = p / w, dx, dy;

			sum += prob[p];
			count++;
			x0 = px < x0 ? px : x0;
			x1 = px > x1 ? px : x1;
			y0 = py < y0 ? py : y0;
			y1 = py > y1 ? py : y1;
			for (dy = -1; dy <= 1; dy++)
				for (dx = -1; dx <= 1; dx++) {
					int qx = px + dx, qy = py + dy, q = qy * w + qx;

					if (qx < 0 || qy < 0 || qx >= w || qy >= h || ocr->label[q] ||
					    prob[q] <= DET_THRESHOLD)
						continue;
					ocr->label[q] = 1;
					ocr->stack[top++] = q;
				}
		}

		bw = (float)(x1 - x0 + 1);
		bh = (float)(y1 - y0 + 1);
		if (sum / count < BOX_THRESHOLD || bw < 3 || bh < 3)
			continue;
		grow = bw * bh * UNCLIP_RATIO / (2 * (bw + bh));
		b->x0 = (int)fmaxf(0, (x0 - grow) / scale);
		b->y0 = (int)fmaxf(0, (y0 - grow) / scale);
		b->x1 = (int)fminf((float)image->width - 1, (x1 + 1 + grow) / scale);
		b->y1 = (int)fminf((float)image->height - 1, (y1 + 1 + grow) / scale);
		b->score = sum / count;
		if (b->x1 - b->x0 >= 2 && b->y1 - b->y0 >= 2)
			n++;
	}
	return n;
}

static int by_position(const void *a, const void *b)
{
	const struct box *p = a, *q = b;
	int line = (p->y0 + p->y1) / 2 - (q->y0 + q->y1) / 2;

	if (abs(line) > (p->y1 - p->y0) / 2)
		return line;
	return p->x0 - q->x0;
}

static int pixel_is_ink(const struct image *image, int x, int y, const int bg[3])
{
	const uint8_t *p = image->rgb + ((size_t)y * image->width + x) * 3;

	return abs(p[0] - bg[0]) + abs(p[1] - bg[1]) + abs(p[2] - bg[2]) > INK_THRESHOLD;
}

/* The box's background: the mean of its corners, which unclip leaves empty. */
static void background(const struct image *image, const struct box *box, int bg[3])
{
	const int xs[2] = { box->x0, box->x1 }, ys[2] = { box->y0, box->y1 };
	int c, i, j;

	for (c = 0; c < 3; c++) {
		bg[c] = 0;
		for (i = 0; i < 2; i++)
			for (j = 0; j < 2; j++)
				bg[c] += image->rgb[((size_t)ys[j] * image->width + xs[i]) * 3 + c];
		bg[c] /= 4;
	}
}

static int column_has_ink(const struct image *image, const struct box *box, const int bg[3], int x)
{
	int y;

	for (y = box->y0; y <= box->y1; y++)
		if (pixel_is_ink(image, x, y, bg))
			return 1;
	return 0;
}

static int row_has_ink(const struct image *image, const struct box *box, const int bg[3], int y)
{
	int x;

	for (x = box->x0; x <= box->x1; x++)
		if (pixel_is_ink(image, x, y, bg))
			return 1;
	return 0;
}

/* Shrinks a piece to its rows with ink, keeping a margin of a sixth of them. */
static void trim_rows(const struct image *image, struct box *piece, const int bg[3])
{
	int y0 = piece->y0, y1 = piece->y1, margin;

	while (y0 < y1 && !row_has_ink(image, piece, bg, y0))
		y0++;
	while (y1 > y0 && !row_has_ink(image, piece, bg, y1))
		y1--;
	if (y1 - y0 < 4)
		return;
	margin = (y1 - y0 + 1) / 6 + 1;
	piece->y0 = y0 - margin < piece->y0 ? piece->y0 : y0 - margin;
	piece->y1 = y1 + margin > piece->y1 ? piece->y1 : y1 + margin;
}

/*
 * Where to cut a line near column ideal: the middle of the widest run of
 * empty columns within reach, or ideal itself when there is none. Sets
 * *space when the run is as wide as a word space.
 */
static int find_cut(const struct image *image, const struct box *box, const int bg[3], int ideal,
		    int reach, int *space)
{
	int x, run = 0, best = 0, cut = ideal, lo = ideal - reach, hi = ideal + reach;

	lo = lo < box->x0 ? box->x0 : lo;
	hi = hi > box->x1 ? box->x1 : hi;
	for (x = lo; x <= hi + 1; x++) {
		if (x <= hi && !column_has_ink(image, box, bg, x)) {
			run++;
			continue;
		}
		if (run > best || (run == best && run && abs(x - run / 2 - ideal) < abs(cut - ideal))) {
			best = run;
			cut = x - (run + 1) / 2;
		}
		run = 0;
	}
	*space = best * 4 >= box->y1 - box->y0 + 1;
	return cut;
}

/*
 * Removes leading, trailing and repeated spaces, and those next to multibyte
 * characters: CJK text and its full-width punctuation go without them.
 */
static void tidy_spaces(char *text)
{
	char *in = text, *out = text;

	for (; *in; in++) {
		if (*in == ' ' && (out == text || out[-1] == ' ' || (unsigned char)out[-1] >= 0x80 ||
				   (unsigned char)in[1] >= 0x80 || !in[1]))
			continue;
		*out++ = *in;
	}
	*out = '\0';
}

/* Reads one piece of a line onto the end of text; adds its confidence and characters. */
static int recognize_piece(struct ocr *ocr, const struct image *image, const struct box *box,
			   char *text, float *confidence, int *chars)
{
	struct model *rec = &ocr->rec;
	int bw = box->x1 - box->x0 + 1, bh = box->y1 - box->y0 + 1;
	int dst_w = (int)fminf((float)rec->width, ceilf((float)rec->height * bw / bh));
	int steps = (int)rec->output->dims[1], classes = (int)rec->output->dims[2], t, c, last = 0;
	size_t used = strlen(text);
	const float *prob;

	memset(rec->input->data, 127, rec->input->size);
	resize(image, (float)box->x0, (float)box->y0, (float)bw, (float)bh, rec->input->data,
	       dst_w > 0 ? dst_w : 1, rec->height, rec->width);
	prob = model_run(rec);
	if (!prob)
		return -EIO;

	for (t = 0; t < steps; t++) {
		const float *p = prob + (size_t)t * classes;
		int best = 0;

		for (c = 1; c < classes; c++)
			if (p[c] > p[best])
				best = c;
		if (best && best != last) {
			size_t len = strlen(ocr->keys[best]);

			if (used + len + 1 < OCR_TEXT_SIZE) {
				memcpy(text + used, ocr->keys[best], len + 1);
				used += len;
			}
			*confidence += p[best];
			(*chars)++;
		}
		last = best;
	}
	return 0;
}

static int recognize(struct ocr *ocr, const struct image *image, const struct box *box,
		     struct ocr_line *line)
{
	int bw = box->x1 - box->x0 + 1, bh = box->y1 - box->y0 + 1;
	float widest = (float)bh * ocr->rec.width / ocr->rec.height, confidence = 0;
	int pieces = (int)ceilf(bw / widest), k, chars = 0, start = box->x0, bg[3], ret;

	line->x0 = box->x0;
	line->y0 = box->y0;
	line->x1 = box->x1;
	line->y1 = box->y1;
	line->score = box->score;
	line->text[0] = '\0';
	if (bw <= widest * 1.1f)
		pieces = 1;
	background(image, box, bg);
	for (k = 1; k <= pieces && start <= box->x1; k++) {
		struct box piece = *box;
		int end = box->x1, space = 0;

		if (k < pieces)
			end = find_cut(image, box, bg, box->x0 + k * bw / pieces, bw / pieces / 4, &space);
		piece.x0 = start;
		piece.x1 = end < start ? start : end;
		trim_rows(image, &piece, bg);
		ret = recognize_piece(ocr, image, &piece, line->text, &confidence, &chars);
		if (ret)
			return ret;
		if (space && strlen(line->text) + 1 < OCR_TEXT_SIZE)
			strcat(line->text, " ");
		start = piece.x1 + 1;
	}
	tidy_spaces(line->text);
	line->confidence = chars ? confidence / chars : 0;
	return 0;
}

/* Splits the dictionary into lines (LF or CRLF) after a blank and before a space. */
static int load_keys(struct ocr *ocr, const char *keys, size_t size)
{
	char *line, *next;
	size_t i, lines = 1;

	ocr->keys_text = malloc(size + 1);
	if (!ocr->keys_text)
		return -1;
	memcpy(ocr->keys_text, keys, size);
	ocr->keys_text[size] = '\0';
	for (i = 0; i < size; i++)
		lines += keys[i] == '\n';
	ocr->keys = calloc(lines + 2, sizeof(*ocr->keys));
	if (!ocr->keys)
		return -1;

	ocr->keys[ocr->nkeys++] = "";
	for (line = ocr->keys_text; line; line = next) {
		size_t len;

		next = strchr(line, '\n');
		if (next)
			*next++ = '\0';
		len = strlen(line);
		if (len && line[len - 1] == '\r')
			line[--len] = '\0';
		if (len)
			ocr->keys[ocr->nkeys++] = line;
	}
	ocr->keys[ocr->nkeys++] = " ";
	return 0;
}

struct ocr *ocr_load(const void *det, size_t det_size, const void *rec, size_t rec_size,
		     const char *keys, size_t keys_size, char *err, size_t err_size)
{
	struct ocr *ocr = calloc(1, sizeof(*ocr));

	if (!ocr) {
		fail(err, err_size, "out of memory");
		return NULL;
	}
	if (model_load(&ocr->det, det, det_size, 4, err, err_size) ||
	    model_load(&ocr->rec, rec, rec_size, 3, err, err_size))
		goto fail;
	if (ocr->det.output->dims[2] != ocr->det.height || ocr->det.output->dims[3] != ocr->det.width) {
		fail(err, err_size, "the detector's map is not the size of its input");
		goto fail;
	}
	if (load_keys(ocr, keys, keys_size)) {
		fail(err, err_size, "out of memory");
		goto fail;
	}
	if (ocr->rec.output->dims[2] != ocr->nkeys) {
		snprintf(err, err_size, "the recognizer reads %lld characters, the dictionary has %d",
			 (long long)ocr->rec.output->dims[2] - 2, ocr->nkeys - 2);
		goto fail;
	}
	ocr->stack = malloc(sizeof(int) * ocr->det.width * ocr->det.height);
	ocr->label = malloc(sizeof(int) * ocr->det.width * ocr->det.height);
	if (!ocr->stack || !ocr->label) {
		fail(err, err_size, "out of memory");
		goto fail;
	}
	return ocr;
fail:
	ocr_free(ocr);
	return NULL;
}

void ocr_free(struct ocr *ocr)
{
	if (!ocr)
		return;
	axnpu_free(ocr->det.npu);
	axnpu_free(ocr->rec.npu);
	free(ocr->keys);
	free(ocr->keys_text);
	free(ocr->stack);
	free(ocr->label);
	free(ocr);
}

int ocr_read(struct ocr *ocr, const uint8_t *rgb, int width, int height, struct ocr_line *lines,
	     int max)
{
	struct image image = { rgb, width, height };
	int n, i, ret;

	if (width < 1 || height < 1)
		return -EINVAL;
	n = detect(ocr, &image);
	if (n < 0)
		return n;
	qsort(ocr->boxes, n, sizeof(ocr->boxes[0]), by_position);
	for (i = 0; i < n && i < max; i++) {
		ret = recognize(ocr, &image, &ocr->boxes[i], &lines[i]);
		if (ret)
			return ret;
	}
	return i;
}
