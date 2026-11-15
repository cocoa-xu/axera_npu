// SPDX-License-Identifier: Apache-2.0
/*
 * An open runtime for Pulsar2 models on the AX620E NPU.
 *
 * An .axmodel is ONNX: its graph holds a "neu mode" node whose attribute
 * npu_graph_info (JSON) names the initializer with the compiled program (a
 * .neu FlatBuffers blob, which the driver parses) and the initializers its
 * extra inputs take (the weights). The .neu lists the tensors of a task in
 * the order the driver wants their buffers: the graph's inputs, a 4-byte
 * "<tensor>_offset" (zero) for each graph input and output, the extra inputs,
 * then the graph's outputs.
 */
#define _FILE_OFFSET_BITS 64
#include "axnpu.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

/* ---- /dev/npu: the vendor's interface plus the open driver's memory */

#define NPU_INIT		0xc0046e00
#define NPU_CREATE_HANDLE	0xc0046e02
#define NPU_CREATE_TASK		0xc0046e03
#define NPU_WAIT_TASKS0		0xc0046e0e
#define NPU_DISTRIBUTION_ONE	0xaa55

struct npu_mem {
	uint64_t phys;
	uint32_t size;
	uint32_t flags;
};

#define NPU_MEM_CACHED		1
#define NPU_MEM_ALLOC		_IOWR('n', 0x40, struct npu_mem)

/* A finished task as the waiting ioctl returns it. */
struct npu_record {
	uint32_t task, unknown, result, callback, context;
};

/* A task's buffer: what the driver reads per tensor. */
struct npu_task_buffer {
	uint64_t phys;
	uint32_t virt;
	uint32_t size;
};

#define MAX_TENSORS 16
#define MAX_TASK 64
#define OFFSET_STRIDE 128

struct buffer {
	uint64_t phys;
	void *virt;
	size_t size;
};

/* What feeds each tensor of the NPU program. */
enum source { FROM_INPUT, FROM_OUTPUT, FROM_CONST, FROM_ZEROS };

struct slot {
	enum source source;
	int index;
	uint32_t size;
};

struct axnpu_model {
	int fd;
	uint32_t handle;
	int ninputs, noutputs, nconsts, nslots, nzeros;
	struct axnpu_tensor inputs[MAX_TENSORS], outputs[MAX_TENSORS];
	struct buffer in_buf[MAX_TENSORS], out_buf[MAX_TENSORS], consts[MAX_TENSORS], zeros;
	char const_names[MAX_TENSORS][64];
	struct slot slots[MAX_TASK];
	struct npu_task_buffer task[MAX_TASK];
};

static int fail(char *err, size_t size, const char *format, ...)
{
	va_list args;

	if (err && size) {
		va_start(args, format);
		vsnprintf(err, size, format, args);
		va_end(args);
	}
	return -1;
}

static int mem_alloc(int fd, size_t size, int cached, struct buffer *buffer)
{
	struct npu_mem mem = { .size = (uint32_t)size, .flags = cached ? NPU_MEM_CACHED : 0 };
	void *virt;

	if (ioctl(fd, NPU_MEM_ALLOC, &mem))
		return -1;
	virt = mmap(NULL, mem.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)mem.phys);
	if (virt == MAP_FAILED)
		return -1;
	buffer->phys = mem.phys;
	buffer->virt = virt;
	buffer->size = mem.size;
	return 0;
}

/* ---- Protocol Buffers, read in place */

struct pb {
	const uint8_t *p, *end;
};

struct pb_field {
	uint32_t num, wire;
	uint64_t value;		/* varint */
	struct pb sub;		/* length-delimited */
};

static int pb_varint(struct pb *b, uint64_t *v)
{
	int shift;

	*v = 0;
	for (shift = 0; b->p < b->end && shift < 64; shift += 7) {
		uint8_t c = *b->p++;

		*v |= (uint64_t)(c & 0x7f) << shift;
		if (!(c & 0x80))
			return 0;
	}
	return -1;
}

/* The next field: 1, or 0 at the end (and on malformed data). */
static int pb_next(struct pb *b, struct pb_field *f)
{
	uint64_t key, len;
	size_t skip;

	if (b->p >= b->end || pb_varint(b, &key))
		return 0;
	f->num = (uint32_t)(key >> 3);
	f->wire = key & 7;
	switch (f->wire) {
	case 0:
		return !pb_varint(b, &f->value);
	case 2:
		if (pb_varint(b, &len) || len > (uint64_t)(b->end - b->p))
			return 0;
		f->sub.p = b->p;
		f->sub.end = b->p + len;
		b->p += len;
		return 1;
	case 1:
	case 5:
		skip = f->wire == 1 ? 8 : 4;
		if ((size_t)(b->end - b->p) < skip)
			return 0;
		b->p += skip;
		return 1;
	default:
		return 0;
	}
}

static int pb_is(const struct pb_field *f, uint32_t num, uint32_t wire)
{
	return f->num == num && f->wire == wire;
}

static int pb_equals(const struct pb_field *f, const char *s)
{
	size_t len = strlen(s);

	return f->wire == 2 && (size_t)(f->sub.end - f->sub.p) == len && !memcmp(f->sub.p, s, len);
}

static int pb_string(const struct pb_field *f, char *out, size_t size)
{
	size_t len = f->sub.end - f->sub.p;

	if (f->wire != 2 || len >= size)
		return -1;
	memcpy(out, f->sub.p, len);
	out[len] = '\0';
	return 0;
}

/* ---- ONNX: ModelProto 7 graph; GraphProto 1 node, 5 initializer, 11 input, 12 output */

static size_t dtype_size(int dtype)
{
	switch (dtype) {
	case AXNPU_UINT8:
	case AXNPU_INT8:
		return 1;
	case AXNPU_UINT16:
	case AXNPU_INT16:
	case AXNPU_FLOAT16:
		return 2;
	case AXNPU_FLOAT:
	case AXNPU_INT32:
		return 4;
	case AXNPU_INT64:
		return 8;
	default:
		return 0;
	}
}

/* ValueInfoProto {1 name, 2 TypeProto {1 Tensor {1 elem_type, 2 shape {1 dim {1 value}}}}} */
static int value_info(struct pb info, struct axnpu_tensor *t)
{
	struct pb_field f, g, h, d, e;
	size_t size;
	int i;

	memset(t, 0, sizeof(*t));
	while (pb_next(&info, &f)) {
		if (pb_is(&f, 1, 2) && pb_string(&f, t->name, sizeof(t->name)))
			return -1;
		if (!pb_is(&f, 2, 2))
			continue;
		while (pb_next(&f.sub, &g)) {
			if (!pb_is(&g, 1, 2))
				continue;
			while (pb_next(&g.sub, &h)) {
				if (pb_is(&h, 1, 0))
					t->dtype = (int)h.value;
				if (!pb_is(&h, 2, 2))
					continue;
				while (pb_next(&h.sub, &d)) {
					if (!pb_is(&d, 1, 2))
						continue;
					if (t->ndims == AXNPU_MAX_DIMS)
						return -1;
					t->dims[t->ndims] = -1;
					while (pb_next(&d.sub, &e))
						if (pb_is(&e, 1, 0))
							t->dims[t->ndims] = (int64_t)e.value;
					t->ndims++;
				}
			}
		}
	}

	size = dtype_size(t->dtype);
	for (i = 0; i < t->ndims; i++) {
		if (t->dims[i] <= 0 || t->dims[i] > (1 << 28) || size > ((size_t)1 << 30) / t->dims[i])
			return -1;
		size *= (size_t)t->dims[i];
	}
	t->size = size;
	return t->name[0] && size ? 0 : -1;
}

/* TensorProto {8 name, 9 raw_data}: the initializer's bytes. */
static int initializer(struct pb graph, const char *name, struct pb *data)
{
	struct pb_field f, g;

	while (pb_next(&graph, &f)) {
		struct pb tensor = f.sub, raw = { NULL, NULL };
		int named = 0;

		if (!pb_is(&f, 5, 2))
			continue;
		while (pb_next(&tensor, &g)) {
			if (pb_is(&g, 8, 2))
				named = pb_equals(&g, name);
			if (pb_is(&g, 9, 2))
				raw = g.sub;
		}
		if (named && raw.p) {
			*data = raw;
			return 0;
		}
	}
	return -1;
}

/* The "neu mode" node's npu_graph_info: NodeProto {4 op_type, 5 AttributeProto {1 name, 4 s}}. */
static int graph_info(struct pb graph, struct pb *json)
{
	struct pb_field f, g, h;

	while (pb_next(&graph, &f)) {
		struct pb node = f.sub, attributes[16];
		int nattributes = 0, npu = 0, i;

		if (!pb_is(&f, 1, 2))
			continue;
		while (pb_next(&node, &g)) {
			if (pb_is(&g, 4, 2))
				npu = pb_equals(&g, "neu mode");
			if (pb_is(&g, 5, 2) && nattributes < 16)
				attributes[nattributes++] = g.sub;
		}
		for (i = 0; npu && i < nattributes; i++) {
			struct pb attribute = attributes[i], value = { NULL, NULL };
			int named = 0;

			while (pb_next(&attribute, &h)) {
				if (pb_is(&h, 1, 2))
					named = pb_equals(&h, "npu_graph_info");
				if (pb_is(&h, 4, 2))
					value = h.sub;
			}
			if (named && value.p) {
				*json = value;
				return 0;
			}
		}
	}
	return -1;
}

/* ---- JSON, only as far as npu_graph_info needs */

static const char *json_ws(const char *p, const char *end)
{
	while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r'))
		p++;
	return p;
}

/* Past the value at p, or NULL. */
static const char *json_skip(const char *p, const char *end, int depth)
{
	char close;

	p = json_ws(p, end);
	if (p >= end || depth > 32)
		return NULL;
	if (*p == '"') {
		for (p++; p < end; p++) {
			if (*p == '\\')
				p++;
			else if (*p == '"')
				return p + 1;
		}
		return NULL;
	}
	if (*p != '{' && *p != '[') {
		while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n')
			p++;
		return p;
	}
	close = *p == '{' ? '}' : ']';
	p = json_ws(p + 1, end);
	if (p < end && *p == close)
		return p + 1;
	for (;;) {
		if (close == '}') {
			p = json_skip(p, end, depth + 1);
			p = p ? json_ws(p, end) : NULL;
			if (!p || p >= end || *p != ':')
				return NULL;
			p++;
		}
		p = json_skip(p, end, depth + 1);
		p = p ? json_ws(p, end) : NULL;
		if (!p || p >= end)
			return NULL;
		if (*p == close)
			return p + 1;
		if (*p != ',')
			return NULL;
		p++;
	}
}

/* The value of key in the object at p, or NULL. */
static const char *json_get(const char *p, const char *end, const char *key)
{
	size_t len = strlen(key);

	p = p ? json_ws(p, end) : NULL;
	if (!p || p >= end || *p != '{')
		return NULL;
	p = json_ws(p + 1, end);
	while (p < end && *p == '"') {
		const char *name = p + 1, *value;
		int match;

		p = json_skip(p, end, 0);
		if (!p)
			return NULL;
		match = (size_t)(p - 1 - name) == len && !memcmp(name, key, len);
		p = json_ws(p, end);
		if (p >= end || *p != ':')
			return NULL;
		value = json_ws(p + 1, end);
		if (match)
			return value;
		p = json_skip(value, end, 0);
		p = p ? json_ws(p, end) : NULL;
		if (!p)
			return NULL;
		if (p < end && *p == ',')
			p = json_ws(p + 1, end);
	}
	return NULL;
}

/* Item index of the array at p, or NULL. */
static const char *json_item(const char *p, const char *end, int index)
{
	int i;

	p = p ? json_ws(p, end) : NULL;
	if (!p || p >= end || *p != '[')
		return NULL;
	p = json_ws(p + 1, end);
	for (i = 0; p && p < end && *p != ']'; i++) {
		if (i == index)
			return p;
		p = json_skip(p, end, 0);
		p = p ? json_ws(p, end) : NULL;
		if (p && p < end && *p == ',')
			p = json_ws(p + 1, end);
	}
	return NULL;
}

/* The string at p, which must not need unescaping. */
static int json_string(const char *p, const char *end, char *out, size_t size)
{
	const char *q = p ? json_skip(p, end, 0) : NULL;
	size_t len;

	if (!q || *p != '"')
		return -1;
	len = q - p - 2;
	if (len >= size || memchr(p + 1, '\\', len))
		return -1;
	memcpy(out, p + 1, len);
	out[len] = '\0';
	return 0;
}

/* ---- .neu (FlatBuffers): the task's tensors, inputs then outputs */

struct fb {
	const uint8_t *buf;
	size_t len;
};

static uint32_t fb_u32(const struct fb *fb, size_t at)
{
	return fb->buf[at] | fb->buf[at + 1] << 8 | fb->buf[at + 2] << 16 |
	       (uint32_t)fb->buf[at + 3] << 24;
}

static uint16_t fb_u16(const struct fb *fb, size_t at)
{
	return (uint16_t)(fb->buf[at] | fb->buf[at + 1] << 8);
}

static int fb_in(const struct fb *fb, size_t at, size_t size)
{
	return at <= fb->len && size <= fb->len - at;
}

/* A table field's position, or 0 when absent. */
static size_t fb_field(const struct fb *fb, size_t table, unsigned int field, size_t size)
{
	int32_t soff;
	size_t vt;
	uint16_t vsize, tsize, off;

	if (!fb_in(fb, table, 4))
		return 0;
	soff = (int32_t)fb_u32(fb, table);
	if ((int64_t)table - soff < 0 || !fb_in(fb, table - soff, 4))
		return 0;
	vt = table - soff;
	vsize = fb_u16(fb, vt);
	tsize = fb_u16(fb, vt + 2);
	if (4 + 2 * field + 2 > vsize || !fb_in(fb, vt, vsize) || !fb_in(fb, table, tsize))
		return 0;
	off = fb_u16(fb, vt + 4 + 2 * field);
	return off && off + size <= tsize ? table + off : 0;
}

/* Where the offset at `at` points, or 0. */
static size_t fb_follow(const struct fb *fb, size_t at)
{
	uint32_t rel;

	if (!at || !fb_in(fb, at, 4))
		return 0;
	rel = fb_u32(fb, at);
	return rel <= fb->len - at && fb_in(fb, at + rel, 4) ? at + rel : 0;
}

struct neu_tensor {
	char name[64];
	uint32_t size;
};

static int neu_tensors(const struct fb *fb, size_t io, unsigned int field, struct neu_tensor *out,
		       int *n)
{
	size_t vec = fb_follow(fb, fb_field(fb, io, field, 4)), i;
	uint32_t count;

	if (!vec)
		return 0;
	count = fb_u32(fb, vec);
	if (count > (uint32_t)(MAX_TASK - *n) || !fb_in(fb, vec + 4, (size_t)count * 4))
		return -1;
	for (i = 0; i < count; i++) {
		size_t table = fb_follow(fb, vec + 4 + 4 * i);
		size_t name = fb_follow(fb, fb_field(fb, table, 0, 4));
		size_t size = fb_field(fb, table, 2, 4);
		uint32_t len;

		if (!table || !name || !size)
			return -1;
		len = fb_u32(fb, name);
		if (len >= sizeof(out->name) || !fb_in(fb, name + 4, len))
			return -1;
		memcpy(out[*n].name, fb->buf + name + 4, len);
		out[*n].name[len] = '\0';
		out[*n].size = fb_u32(fb, size);
		(*n)++;
	}
	return 0;
}

/* ---- Models */

/* Where a model's parts lie in the .axmodel while it loads. */
struct parts {
	struct pb graph, neu;
	struct pb const_data[MAX_TENSORS];
	struct neu_tensor tensors[MAX_TASK];
	int ntensors;
};

static int find(const struct axnpu_tensor *tensors, int n, const char *name)
{
	int i;

	for (i = 0; i < n; i++)
		if (!strcmp(tensors[i].name, name))
			return i;
	return -1;
}

static int ends_with(const char *s, const char *suffix)
{
	size_t len = strlen(s), slen = strlen(suffix);

	return len > slen && !strcmp(s + len - slen, suffix);
}

/* The graph's inputs and outputs (ValueInfoProto 11 and 12). */
static int parse_io(struct axnpu_model *m, struct pb graph, char *err, size_t err_size)
{
	struct pb_field f;

	while (pb_next(&graph, &f)) {
		struct axnpu_tensor *t = NULL;

		if (pb_is(&f, 11, 2) || pb_is(&f, 12, 2)) {
			int *n = f.num == 11 ? &m->ninputs : &m->noutputs;

			if (*n == MAX_TENSORS)
				return fail(err, err_size, "too many graph inputs or outputs");
			t = f.num == 11 ? &m->inputs[(*n)++] : &m->outputs[(*n)++];
		}
		if (t && value_info(f.sub, t))
			return fail(err, err_size, "bad graph input or output");
	}
	return 0;
}

/* The first program's .neu blob and its extra inputs, as npu_graph_info names them. */
static int parse_program(struct axnpu_model *m, struct parts *parts, char *err, size_t err_size)
{
	struct pb json;
	const char *text, *end, *dotneu, *extra, *item;
	char key[128];
	int i;

	if (graph_info(parts->graph, &json))
		return fail(err, err_size, "no NPU program (\"neu mode\" node)");
	text = (const char *)json.p;
	end = (const char *)json.end;

	dotneu = json_item(json_get(text, end, "dotneus"), end, 0);
	if (json_string(json_get(dotneu, end, "neu_key"), end, key, sizeof(key)))
		return fail(err, err_size, "no neu_key in npu_graph_info");
	if (initializer(parts->graph, key, &parts->neu))
		return fail(err, err_size, "no initializer %s", key);

	extra = json_get(dotneu, end, "extra_inputs");
	for (i = 0; (item = json_item(extra, end, i)); i++) {
		if (m->nconsts == MAX_TENSORS ||
		    json_string(json_get(item, end, "name"), end, m->const_names[m->nconsts],
				sizeof(m->const_names[0])) ||
		    json_string(json_get(item, end, "const_data_key"), end, key, sizeof(key)))
			return fail(err, err_size, "bad extra input %d", i);
		if (initializer(parts->graph, key, &parts->const_data[m->nconsts]))
			return fail(err, err_size, "no initializer %s", key);
		m->nconsts++;
	}
	return 0;
}

/* What feeds each of the program's tensors, each the size the program says. */
static int plan(struct axnpu_model *m, const struct parts *parts, char *err, size_t err_size)
{
	struct fb fb = { parts->neu.p, (size_t)(parts->neu.end - parts->neu.p) };
	struct neu_tensor tensors[MAX_TASK];
	size_t root, io;
	int i, k, n = 0;

	root = fb.len >= 8 && fb_in(&fb, fb_u32(&fb, 0), 4) ? fb_u32(&fb, 0) : 0;
	io = fb_follow(&fb, fb_field(&fb, root, 2, 4));
	if (!io || neu_tensors(&fb, io, 0, tensors, &n) || neu_tensors(&fb, io, 1, tensors, &n))
		return fail(err, err_size, "bad NPU program");

	for (i = 0; i < n; i++) {
		const struct neu_tensor *t = &tensors[i];
		struct slot *s = &m->slots[m->nslots++];

		if ((k = find(m->inputs, m->ninputs, t->name)) >= 0) {
			*s = (struct slot){ FROM_INPUT, k, t->size };
			if (t->size != m->inputs[k].size)
				return fail(err, err_size, "input %s: %u bytes in the program, %zu in the graph",
					    t->name, t->size, m->inputs[k].size);
			continue;
		}
		if ((k = find(m->outputs, m->noutputs, t->name)) >= 0) {
			*s = (struct slot){ FROM_OUTPUT, k, t->size };
			if (t->size != m->outputs[k].size)
				return fail(err, err_size, "output %s: %u bytes in the program, %zu in the graph",
					    t->name, t->size, m->outputs[k].size);
			continue;
		}
		for (k = 0; k < m->nconsts && strcmp(m->const_names[k], t->name); k++)
			;
		if (k < m->nconsts) {
			*s = (struct slot){ FROM_CONST, k, t->size };
			if (t->size > (size_t)(parts->const_data[k].end - parts->const_data[k].p))
				return fail(err, err_size, "%s: %u bytes wanted", t->name, t->size);
		} else if (ends_with(t->name, "_offset") && t->size <= OFFSET_STRIDE) {
			*s = (struct slot){ FROM_ZEROS, m->nzeros++, t->size };
		} else {
			return fail(err, err_size, "unknown tensor %s", t->name);
		}
	}
	return 0;
}

/* The model on the NPU: its memory, its program, its task. */
static int attach(struct axnpu_model *m, const struct parts *parts, char *err, size_t err_size)
{
	uint32_t attr[5] = { NPU_DISTRIBUTION_ONE };
	uint32_t create[6];
	int i;

	m->fd = open("/dev/npu", O_RDWR | O_CLOEXEC);
	if (m->fd < 0)
		return fail(err, err_size, "/dev/npu: %s", strerror(errno));
	if (ioctl(m->fd, NPU_INIT, attr))
		return fail(err, err_size, "init: %s", strerror(errno));

	for (i = 0; i < m->nconsts; i++) {
		size_t size = parts->const_data[i].end - parts->const_data[i].p;

		if (mem_alloc(m->fd, size, 0, &m->consts[i]))
			return fail(err, err_size, "%s: no memory for %zu bytes", m->const_names[i], size);
		memcpy(m->consts[i].virt, parts->const_data[i].p, size);
	}
	for (i = 0; i < m->ninputs; i++) {
		if (mem_alloc(m->fd, m->inputs[i].size, 1, &m->in_buf[i]))
			return fail(err, err_size, "no memory for input %s", m->inputs[i].name);
		m->inputs[i].data = m->in_buf[i].virt;
	}
	for (i = 0; i < m->noutputs; i++) {
		if (mem_alloc(m->fd, m->outputs[i].size, 1, &m->out_buf[i]))
			return fail(err, err_size, "no memory for output %s", m->outputs[i].name);
		m->outputs[i].data = m->out_buf[i].virt;
	}
	if (mem_alloc(m->fd, (size_t)(m->nzeros + 1) * OFFSET_STRIDE, 0, &m->zeros))
		return fail(err, err_size, "no memory");
	memset(m->zeros.virt, 0, m->zeros.size);

	for (i = 0; i < m->nslots; i++) {
		const struct slot *s = &m->slots[i];
		const struct buffer *b = s->source == FROM_INPUT ? &m->in_buf[s->index] :
					 s->source == FROM_OUTPUT ? &m->out_buf[s->index] :
					 s->source == FROM_CONST ? &m->consts[s->index] : &m->zeros;
		size_t offset = s->source == FROM_ZEROS ? (size_t)s->index * OFFSET_STRIDE : 0;

		m->task[i].phys = b->phys + offset;
		m->task[i].virt = (uint32_t)(uintptr_t)((uint8_t *)b->virt + offset);
		m->task[i].size = s->size;
	}

	create[0] = (uint32_t)(uintptr_t)parts->neu.p;
	create[1] = (uint32_t)(parts->neu.end - parts->neu.p);
	create[2] = (uint32_t)(uintptr_t)&m->handle;
	create[3] = create[4] = create[5] = 0;
	if (ioctl(m->fd, NPU_CREATE_HANDLE, create))
		return fail(err, err_size, "the driver did not take the program: %s", strerror(errno));
	return 0;
}

static void unmap(struct buffer *b)
{
	if (b->virt)
		munmap(b->virt, b->size);
}

static void release(struct axnpu_model *m)
{
	int i;

	for (i = 0; i < MAX_TENSORS; i++) {
		unmap(&m->in_buf[i]);
		unmap(&m->out_buf[i]);
		unmap(&m->consts[i]);
	}
	unmap(&m->zeros);
	if (m->fd >= 0)
		close(m->fd);	/* the driver frees the program and the memory */
	free(m);
}

static struct axnpu_model *open_model(const void *axmodel, size_t size, int on_npu, char *err,
				      size_t err_size)
{
	struct pb model = { axmodel, (const uint8_t *)axmodel + size };
	struct axnpu_model *m = calloc(1, sizeof(*m));
	struct parts *parts = calloc(1, sizeof(*parts));
	struct pb_field f;
	int ret = -1;

	if (!m || !parts) {
		fail(err, err_size, "out of memory");
		goto out;
	}
	m->fd = -1;
	while (pb_next(&model, &f))
		if (pb_is(&f, 7, 2))
			parts->graph = f.sub;
	if (!parts->graph.p) {
		fail(err, err_size, "not an ONNX model");
		goto out;
	}
	ret = parse_io(m, parts->graph, err, err_size);
	if (!ret)
		ret = parse_program(m, parts, err, err_size);
	if (!ret)
		ret = plan(m, parts, err, err_size);
	if (!ret && on_npu)
		ret = attach(m, parts, err, err_size);
out:
	free(parts);
	if (ret && m) {
		release(m);
		m = NULL;
	}
	return m;
}

struct axnpu_model *axnpu_load(const void *axmodel, size_t size, char *err, size_t err_size)
{
	return open_model(axmodel, size, 1, err, err_size);
}

struct axnpu_model *axnpu_describe(const void *axmodel, size_t size, char *err, size_t err_size)
{
	return open_model(axmodel, size, 0, err, err_size);
}

void axnpu_free(struct axnpu_model *model)
{
	if (model)
		release(model);
}

int axnpu_num_inputs(const struct axnpu_model *model)
{
	return model->ninputs;
}

int axnpu_num_outputs(const struct axnpu_model *model)
{
	return model->noutputs;
}

struct axnpu_tensor *axnpu_input(struct axnpu_model *model, int index)
{
	return index >= 0 && index < model->ninputs ? &model->inputs[index] : NULL;
}

struct axnpu_tensor *axnpu_output(struct axnpu_model *model, int index)
{
	return index >= 0 && index < model->noutputs ? &model->outputs[index] : NULL;
}

int axnpu_run(struct axnpu_model *m, int timeout_ms)
{
	struct { uint32_t buffers, count; } io = { (uint32_t)(uintptr_t)m->task, (uint32_t)m->nslots };
	uint32_t id = 0;
	struct { uint32_t handle, io, io_start, ext, task_out, callback, context; } info = {
		m->handle, (uint32_t)(uintptr_t)&io, (uint32_t)m->nslots, 0, (uint32_t)(uintptr_t)&id, 0, 0,
	};
	struct { uint32_t info, size; } request = { (uint32_t)(uintptr_t)&info, sizeof(info) };
	uint64_t timeout = timeout_ms > 0 ? (uint64_t)timeout_ms : 5000;

	if (m->fd < 0)
		return -ENODEV;
	if (ioctl(m->fd, NPU_CREATE_TASK, &request))
		return -errno;
	for (;;) {
		uint8_t wait[0x60] = { 0 };
		struct npu_record record;
		uint32_t count, i;

		memcpy(wait + 0x58, &timeout, sizeof(timeout));
		if (ioctl(m->fd, NPU_WAIT_TASKS0, wait))
			return -errno;
		memcpy(&count, wait + 0x50, sizeof(count));
		for (i = 0; i < count && i < 4; i++) {
			memcpy(&record, wait + i * sizeof(record), sizeof(record));
			if (record.task == id)
				return record.result ? -EIO : 0;
		}
	}
}
