// SPDX-License-Identifier: GPL-2.0-only
/*
 * Axera AX620E NPU
 *
 * /dev/npu with the interface of the vendor's ax_npu.ko V3.0.0, so that the
 * vendor's NPU runtime (libax_engine.so, libax_interpreter.so) runs as it
 * is, plus ioctls of our own that hand out DMA memory to an open
 * libax_sys.so: uncached, or cached with explicit cache maintenance.
 *
 * The NPU runs a command stream the vendor's compiler made: the driver
 * copies it into DMA memory, writes the addresses of the task's buffers into
 * a table of runtime values at the start of the on-chip memory (OCM), copies
 * the model's small OCM parameters there, points the dispatcher's queues at
 * them and starts it. The sync manager interrupts when the task is done.
 *
 * This version drives one virtual NPU: all four execution units and the
 * whole OCM (the vendor's 0xaa55 distribution). Tasks run one at a time.
 * The NPU has no IOMMU and runs whatever the command stream says, so
 * /dev/npu is for trusted users only.
 */

#include <linux/completion.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/idr.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/kref.h>
#include <linux/kthread.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/mod_devicetable.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of_reserved_mem.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/unaligned.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>

/* Register blocks, from the start of the NPU's registers. */
#define NPU_EU(n)		((n) * 0x10000)
#define NPU_DISPATCH		0x40000
#define NPU_SYNC		0x50000
#define NPU_GLB			0x90000

#define EU_CONFIG		0x00
#define EU_CRC0			0x08
#define EU_CRC1			0x0c
#define EU_CC			0xcc
#define EU_ERROR		0xd0
#define EU_ERROR_CLR		0xd4
#define EU_IR_DROP		0xdc

#define D_OCM_BASE0		0x000
#define D_MCODE_BASE0		0x004
#define D_QUEUE0_BASE		0x100
#define D_QUEUE0_SIZE		0x104
#define D_QUEUE1_BASE		0x110
#define D_QUEUE1_SIZE		0x114
#define D_CONFIG		0x120
#define D_THROTTLE		0x124
#define D_TRIGGER		0x168
#define D_IDLE			0x190
#define D_ERROR			0x20c
#define D_ERROR_CLR		0x210

#define SM_ISP0			0x14
#define SM_ISP0_CONF		0x18
#define SM_ISP1			0x34
#define SM_ISP1_CONF		0x38
#define SM_IRQ_STATUS_CLR0	0x5c
#define SM_IRQ_MASK0		0x64
#define SM_IRQ_MASK1		0x68
#define SM_IRQ_STATUS0		0x74
#define SM_ERROR_CLR0		0x7c
#define SM_ERROR0		0x84
#define SM_EU_STATUS_CLR0	0x8c
#define SM_CONV_DELAY		0x1a4

#define G_CLK_MUX		0x00
#define G_RESETS		0x0c
#define G_AUTO_GATE		0x10
#define G_EB0_SET		0x64
#define G_EB0_CLR		0x68
#define G_EB1_SET		0x6c
#define G_EB1_CLR		0x70
#define G_RST_ASSERT		0x74
#define G_RST_DEASSERT		0x78

/* The vendor's interface: magic 'n', a pointer argument. */
#define NPU_INIT		0xc0046e00
#define NPU_DEINIT		0x00006e01
#define NPU_CREATE_HANDLE	0xc0046e02
#define NPU_CREATE_TASK		0xc0046e03
#define NPU_CANCEL_TASK		0x40046e05
#define NPU_DESTROY_HANDLE	0x40046e06
#define NPU_HAS_DYNAMIC_PARAMS	0xc0046e07
#define NPU_HARD_RESET		0x00006e08
#define NPU_GET_ATTR		0xc0046e09
#define NPU_GET_MODEL_TYPE	0xc0046e0a
#define NPU_WAIT_TASKS0		0xc0046e0e
#define NPU_WAIT_TASKS1		0xc0046e0f
#define NPU_GET_MODEL_CMM_INFO	0xc0046e12
#define NPU_GET_VNPU_NUM	0xc0046e16
#define NPU_GET_AFFINITY	0xc0046e17
#define NPU_SET_AFFINITY	0xc0046e18

/* Ours: DMA memory for libax_sys.so, mapped by mmap at its address. */
struct npu_mem {
	__u64 phys;
	__u32 size;
	__u32 flags;
};

#define NPU_MEM_CACHED		BIT(0)	/* alloc: a cached mapping */
#define NPU_MEM_TO_CPU		BIT(0)	/* sync: invalidate, else write back */

#define NPU_MEM_ALLOC		_IOWR('n', 0x40, struct npu_mem)
#define NPU_MEM_FREE		_IOW('n', 0x41, struct npu_mem)
#define NPU_MEM_SYNC		_IOW('n', 0x42, struct npu_mem)

#define NPU_DISTRIBUTION_ONE	0xaa55
#define NPU_MAX_MODEL		(64 << 20)
#define NPU_MAX_BUFFER		(256 << 20)
#define NPU_MAX_RING		(64 << 20)
#define NPU_MAX_IO		64
#define NPU_MAX_PARAMS		64
#define NPU_TASK_TIMEOUT	msecs_to_jiffies(5000)
#define NPU_TASK_FAILED		0x80060089

/*
 * Where a tensor or a parameter lives (the vendor's numbers; the model has a
 * flag): DDR ones get a runtime value.
 */
#define NPU_IN_DDR		0
#define NPU_IN_OCM		2
#define NPU_NO_RTV		0xffff

struct npu_io {
	char name[64];
	u32 ocm_offset;
	u32 size;
	u8 type;
	u16 rtv;
};

struct npu_param {
	u8 type;
	u16 rtv;
	u32 ocm_offset;
	u32 size;
	void *data;		/* OCM: copied there before each run */
	void *ddr;		/* DDR: a copy the NPU reads */
	dma_addr_t ddr_dma;
};

struct npu_queue {
	u8 id;
	u32 size;
	u32 ocm_offset;
};

struct npu_model {
	struct file *owner;
	void *mcode;
	dma_addr_t mcode_dma;
	size_t mcode_size;
	struct npu_queue queues[2];
	unsigned int nqueues;
	struct npu_io io[NPU_MAX_IO];	/* inputs, then outputs */
	unsigned int nio, ninputs;
	struct npu_param params[NPU_MAX_PARAMS];
	unsigned int nparams;
	u32 ocm_reserved;
	u16 ocm_base_rtv, ring_rtv;
	u32 ring_size;
};

/* Memory for user space: in the device's list while allocated, and mapped. */
struct npu_buffer {
	struct list_head node;
	struct kref ref;
	struct npu_dev *npu;
	struct file *owner;
	bool cached;
	void *cpu;		/* uncached */
	struct page *pages;	/* cached */
	dma_addr_t dma;
	size_t size;
};

/* A finished task, as the waiting ioctl hands it out. */
struct npu_record {
	u32 task;
	u32 unknown;
	u32 result;
	u32 callback;
	u32 context;
};

struct npu_task {
	struct list_head node;
	struct file *owner;
	u32 id;
	u32 handle;
	u64 phys[NPU_MAX_IO];
	u32 callback, context;
};

struct npu_client {
	struct npu_dev *npu;
	spinlock_t lock;
	struct npu_record records[64];
	unsigned int head, tail;
	wait_queue_head_t wait;
};

struct npu_dev {
	struct device *dev;
	struct miscdevice misc;
	void __iomem *regs, *ocm;
	size_t ocm_size;

	struct mutex lock;		/* models, buffers, the hardware */
	struct idr models;
	struct list_head buffers;
	bool hw_ready;

	void *ring;
	dma_addr_t ring_dma;
	size_t ring_size;

	spinlock_t tasks_lock;
	struct list_head tasks;
	wait_queue_head_t tasks_wait;
	struct task_struct *worker;
	u32 next_task;

	struct completion done;
	u32 irq_status, error_status;
};

static inline void npu_w(struct npu_dev *npu, u32 off, u32 val)
{
	writel(val, npu->regs + off);
}

static inline u32 npu_r(struct npu_dev *npu, u32 off)
{
	return readl(npu->regs + off);
}

/* The vendor's interface passes 32-bit user pointers. */
static void __user *npu_uptr(u32 addr)
{
	return (void __user *)(uintptr_t)addr;
}

/* An address as the dispatcher and the runtime values take it. */
static u32 npu_addr(u64 addr)
{
	return lower_32_bits(addr) >> 5 | upper_32_bits(addr) << 27;
}

/* ---- Hardware */

static void npu_hw_init(struct npu_dev *npu)
{
	u32 mux, mask0, mask1;
	unsigned int n;

	/* Clock sources: EU 3, global 3, timer 5 (gated while it changes). */
	npu_w(npu, NPU_GLB + G_EB0_CLR, BIT(1));
	mux = npu_r(npu, NPU_GLB + G_CLK_MUX) & ~0xff;
	npu_w(npu, NPU_GLB + G_CLK_MUX, mux | 3 | 3 << 3 | 5 << 5);
	npu_w(npu, NPU_GLB + G_EB0_SET, BIT(1));

	npu_w(npu, NPU_GLB + G_RST_DEASSERT, 0xfff);
	npu_w(npu, NPU_GLB + G_EB0_CLR, BIT(2));
	npu_w(npu, NPU_GLB + G_EB0_SET, ~BIT(2));
	npu_w(npu, NPU_GLB + G_EB1_CLR, 0x311);
	npu_w(npu, NPU_GLB + G_EB1_SET, ~0x311);
	npu_w(npu, NPU_GLB + G_AUTO_GATE, npu_r(npu, NPU_GLB + G_AUTO_GATE) & ~3);

	/* Out of reset the sync manager clears these before it takes them. */
	if (readl_poll_timeout(npu->regs + NPU_SYNC + SM_IRQ_MASK0, mask0, !mask0, 100, 100000) ||
	    readl_poll_timeout(npu->regs + NPU_SYNC + SM_IRQ_MASK1, mask1, !mask1, 100, 100000))
		dev_warn(npu->dev, "Interrupt masks stay set\n");
	npu_w(npu, NPU_SYNC + SM_IRQ_MASK0, 0xfffff000);
	npu_w(npu, NPU_SYNC + SM_IRQ_MASK1, 0xfffff000);

	for (n = 0; n < 4; n++) {
		npu_w(npu, NPU_EU(n) + EU_CC, 0);
		npu_w(npu, NPU_EU(n) + EU_CRC0, 0);
		npu_w(npu, NPU_EU(n) + EU_CRC1, 0);
		npu_w(npu, NPU_EU(n) + EU_CONFIG, 0x3f);
	}
	npu_w(npu, NPU_DISPATCH + D_CONFIG, 0x2000);

	/* The vendor's defaults: IR drop 100/0x50, no throttle, ISP sync. */
	npu_w(npu, NPU_EU(0) + EU_IR_DROP, 100 << 5);
	npu_w(npu, NPU_EU(1) + EU_IR_DROP, 100 << 5);
	npu_w(npu, NPU_SYNC + SM_CONV_DELAY, 0x50);
	npu_w(npu, NPU_DISPATCH + D_THROTTLE, 0);
	npu_w(npu, NPU_SYNC + SM_ISP0, 0xffffffff);
	npu_w(npu, NPU_SYNC + SM_ISP0_CONF, 16 << 4 | 16 << 9 | 8);
	npu_w(npu, NPU_SYNC + SM_ISP1, 0xffffffff);
	npu_w(npu, NPU_SYNC + SM_ISP1_CONF, 16 << 4 | 16 << 9 | 8);

	npu->hw_ready = true;
	dev_dbg(npu->dev, "Ready: resets %#x, mux %#x\n", npu_r(npu, NPU_GLB + G_RESETS),
		npu_r(npu, NPU_GLB + G_CLK_MUX));
}

static void npu_hw_reset(struct npu_dev *npu)
{
	npu_w(npu, NPU_GLB + G_RST_ASSERT, 0xfff);
	udelay(10);
	npu->hw_ready = false;
}

static irqreturn_t npu_irq_normal(int irq, void *data)
{
	struct npu_dev *npu = data;
	u32 status = npu_r(npu, NPU_SYNC + SM_IRQ_STATUS0);

	npu_w(npu, NPU_SYNC + SM_IRQ_STATUS_CLR0, status);
	npu->irq_status = status;
	complete(&npu->done);
	return IRQ_HANDLED;
}

/*
 * Bits 0-3 of the error status name the execution units with an error code,
 * bit 4 the dispatcher; each code is cleared where it is read.
 */
static irqreturn_t npu_irq_error(int irq, void *data)
{
	struct npu_dev *npu = data;
	u32 status = npu_r(npu, NPU_SYNC + SM_ERROR0);
	unsigned int n;

	npu_w(npu, NPU_SYNC + SM_ERROR_CLR0, status);
	for (n = 0; n < 4; n++) {
		if (!(status & BIT(n)))
			continue;
		dev_err(npu->dev, "EU%u error %#x\n", n, npu_r(npu, NPU_EU(n) + EU_ERROR) & 0xffff);
		npu_w(npu, NPU_EU(n) + EU_ERROR_CLR, 1);
	}
	if (status & BIT(4)) {
		dev_err(npu->dev, "Dispatcher error %#x\n", npu_r(npu, NPU_DISPATCH + D_ERROR));
		npu_w(npu, NPU_DISPATCH + D_ERROR_CLR, 1);
	}
	npu->error_status = status ? status : ~0U;
	complete(&npu->done);
	return IRQ_HANDLED;
}

/* ---- Models: the .neu FlatBuffers blob, every offset checked */

struct fb {
	const u8 *buf;
	size_t len;
};

struct fb_table {
	const struct fb *fb;
	size_t pos, vt;
	u16 vsize, tsize;
};

static bool fb_in(const struct fb *fb, size_t pos, size_t size)
{
	return pos <= fb->len && size <= fb->len - pos;
}

static u32 fb_u32(const struct fb *fb, size_t pos)
{
	return get_unaligned_le32(fb->buf + pos);
}

/* Where the offset stored at `at` points to, or 0. */
static size_t fb_follow(const struct fb *fb, size_t at)
{
	u32 rel = fb_u32(fb, at);

	if (rel > fb->len - at || !fb_in(fb, at + rel, 4))
		return 0;
	return at + rel;
}

static int fb_table_at(const struct fb *fb, size_t pos, struct fb_table *t)
{
	s32 soff;

	if (!fb_in(fb, pos, 4))
		return -EINVAL;
	soff = (s32)fb_u32(fb, pos);
	if ((s64)pos - soff < 0 || !fb_in(fb, pos - soff, 4))
		return -EINVAL;
	t->fb = fb;
	t->pos = pos;
	t->vt = pos - soff;
	t->vsize = get_unaligned_le16(fb->buf + t->vt);
	t->tsize = get_unaligned_le16(fb->buf + t->vt + 2);
	if (t->vsize < 4 || !fb_in(fb, t->vt, t->vsize) || !fb_in(fb, pos, t->tsize))
		return -EINVAL;
	return 0;
}

/* A field's offset in its table, or 0 when it is absent. */
static size_t fb_field(const struct fb_table *t, unsigned int field, size_t size)
{
	size_t slot = 4 + 2 * field;
	u16 off;

	if (slot + 2 > t->vsize)
		return 0;
	off = get_unaligned_le16(t->fb->buf + t->vt + slot);
	if (!off || off + size > t->tsize)
		return 0;
	return off;
}

static u32 fb_get_u32(const struct fb_table *t, unsigned int field)
{
	size_t off = fb_field(t, field, 4);

	return off ? fb_u32(t->fb, t->pos + off) : 0;
}

static u8 fb_get_u8(const struct fb_table *t, unsigned int field)
{
	size_t off = fb_field(t, field, 1);

	return off ? t->fb->buf[t->pos + off] : 0;
}

static size_t fb_target(const struct fb_table *t, unsigned int field)
{
	size_t off = fb_field(t, field, 4);

	return off ? fb_follow(t->fb, t->pos + off) : 0;
}

static int fb_get_table(const struct fb_table *t, unsigned int field, struct fb_table *out)
{
	size_t target = fb_target(t, field);

	return target ? fb_table_at(t->fb, target, out) : -ENOENT;
}

/* A vector's elements: their start and number, each `size` bytes. */
static int fb_get_vector(const struct fb_table *t, unsigned int field, size_t size,
			 size_t *start, u32 *count)
{
	size_t target = fb_target(t, field);

	if (!target)
		return -ENOENT;
	*count = fb_u32(t->fb, target);
	*start = target + 4;
	if (*count > t->fb->len / max_t(size_t, size, 1) || !fb_in(t->fb, *start, *count * size))
		return -EINVAL;
	return 0;
}

static int fb_vector_table(const struct fb_table *t, size_t start, u32 i, struct fb_table *out)
{
	size_t target = fb_follow(t->fb, start + 4 * i);

	return target ? fb_table_at(t->fb, target, out) : -EINVAL;
}

static int fb_get_string(const struct fb_table *t, unsigned int field, const char **s, u32 *len)
{
	size_t start;
	int ret = fb_get_vector(t, field, 1, &start, len);

	if (ret)
		return ret;
	if (!fb_in(t->fb, start, *len + 1) || t->fb->buf[start + *len])
		return -EINVAL;
	*s = (const char *)t->fb->buf + start;
	return 0;
}

static bool fb_string_eq(const struct fb_table *t, unsigned int field, const char *name)
{
	const char *s;
	u32 len;

	return !fb_get_string(t, field, &s, &len) && len == strlen(name) && !memcmp(s, name, len);
}

/* The runtime value index of a name, from the rtv entries {0 name, 1 index}. */
static int npu_rtv_index(const struct fb_table *rtv, const char *name, u16 *index)
{
	struct fb_table entry;
	size_t start;
	u32 count, i, value;

	if (fb_get_vector(rtv, 2, 4, &start, &count))
		return -ENOENT;
	for (i = 0; i < count; i++) {
		if (fb_vector_table(rtv, start, i, &entry))
			return -EINVAL;
		if (fb_string_eq(&entry, 0, name)) {
			value = fb_get_u32(&entry, 1);
			if (value >= NPU_NO_RTV)
				return -EINVAL;
			*index = value;
			return 0;
		}
	}
	return -ENOENT;
}

/* A tensor or a parameter: {0 name, 1 OCM offset, 2 size, 3 in OCM}. */
static int npu_parse_io(const struct fb_table *t, const struct fb_table *rtv, struct npu_io *io)
{
	const char *name;
	u32 len;

	if (fb_get_string(t, 0, &name, &len) || len >= sizeof(io->name))
		return -EINVAL;
	memcpy(io->name, name, len);
	io->name[len] = '\0';
	io->ocm_offset = fb_get_u32(t, 1);
	io->size = fb_get_u32(t, 2);
	io->type = fb_get_u8(t, 3) ? NPU_IN_OCM : NPU_IN_DDR;
	io->rtv = NPU_NO_RTV;
	if (io->type == NPU_IN_DDR && npu_rtv_index(rtv, io->name, &io->rtv))
		return -EINVAL;
	return 0;
}

static int npu_parse_tensors(struct npu_model *model, const struct fb_table *io,
			     const struct fb_table *rtv, unsigned int field)
{
	struct fb_table t;
	size_t start;
	u32 count, i;

	if (fb_get_vector(io, field, 4, &start, &count))
		return 0;
	if (count > NPU_MAX_IO - model->nio)
		return -E2BIG;
	for (i = 0; i < count; i++) {
		if (fb_vector_table(io, start, i, &t) ||
		    npu_parse_io(&t, rtv, &model->io[model->nio]))
			return -EINVAL;
		model->nio++;
	}
	return 0;
}

/* A parameter's data from the root's {0 bytes, 1 name} tables. */
static int npu_param_data(const struct fb_table *root, const char *name, const u8 **data,
			  u32 *len)
{
	struct fb_table t;
	size_t start, bytes;
	u32 count, i;

	if (fb_get_vector(root, 4, 4, &start, &count))
		return -ENOENT;
	for (i = 0; i < count; i++) {
		if (fb_vector_table(root, start, i, &t))
			return -EINVAL;
		if (fb_string_eq(&t, 1, name)) {
			if (fb_get_vector(&t, 0, 1, &bytes, len))
				return -EINVAL;
			*data = root->fb->buf + bytes;
			return 0;
		}
	}
	return -ENOENT;
}

/*
 * The model's parameters: small ones the NPU wants in OCM, copied there
 * before each run, and ones it reads from DDR, copied once into DMA memory.
 */
static int npu_parse_params(struct npu_dev *npu, struct npu_model *model,
			    const struct fb_table *root, const struct fb_table *io,
			    const struct fb_table *rtv)
{
	struct npu_param *param;
	struct fb_table t;
	struct npu_io info;
	size_t start;
	const u8 *data;
	u32 count, i, len;

	if (fb_get_vector(io, 3, 4, &start, &count))
		return 0;
	if (count > NPU_MAX_PARAMS)
		return -E2BIG;

	for (i = 0; i < count; i++) {
		if (fb_vector_table(io, start, i, &t) || npu_parse_io(&t, rtv, &info))
			return -EINVAL;
		if (npu_param_data(root, info.name, &data, &len) || !len || len > info.size)
			return -EINVAL;

		param = &model->params[model->nparams++];
		param->type = info.type;
		param->rtv = info.rtv;
		param->ocm_offset = info.ocm_offset;
		param->size = len;

		if (info.type == NPU_IN_OCM) {
			if (len % 4 || info.ocm_offset % 4 || info.ocm_offset > npu->ocm_size ||
			    len > npu->ocm_size - info.ocm_offset)
				return -EINVAL;
			param->data = kmemdup(data, len, GFP_KERNEL);
			if (!param->data)
				return -ENOMEM;
		} else {
			param->ddr = dma_alloc_coherent(npu->dev, len, &param->ddr_dma, GFP_KERNEL);
			if (!param->ddr)
				return -ENOMEM;
			memcpy(param->ddr, data, len);
		}
	}
	return 0;
}

static int npu_parse_model(struct npu_dev *npu, struct npu_model *model, const struct fb *fb)
{
	struct fb_table root, soc, mcode, rtv, io, t;
	size_t start;
	u32 count, i, usage, reserved;
	int ret;

	if (fb->len < 8 || fb_table_at(fb, fb_u32(fb, 0), &root))
		return -EINVAL;
	if (fb_get_u8(&root, 9) != 1) {
		dev_dbg(npu->dev, "Control mode %u\n", fb_get_u8(&root, 9));
		return -EINVAL;
	}
	if (!fb_get_table(&root, 1, &soc) && fb_get_u8(&soc, 0) != 0) {
		dev_dbg(npu->dev, "SoC type %u\n", fb_get_u8(&soc, 0));
		return -EINVAL;
	}

	/* The command stream and its queues {0 id, 1 size, 2 OCM offset}. */
	if (fb_get_table(&root, 5, &mcode) || fb_get_vector(&mcode, 2, 8, &start, &count) || !count)
		return -EINVAL;
	model->mcode_size = (size_t)count * 8;
	model->mcode = dma_alloc_coherent(npu->dev, model->mcode_size, &model->mcode_dma,
					  GFP_KERNEL);
	if (!model->mcode)
		return -ENOMEM;
	memcpy(model->mcode, fb->buf + start, model->mcode_size);

	if (fb_get_vector(&mcode, 1, 4, &start, &count) || !count || count > 2)
		return -EINVAL;
	for (i = 0; i < count; i++) {
		struct npu_queue *q = &model->queues[i];

		if (fb_vector_table(&mcode, start, i, &t))
			return -EINVAL;
		q->id = fb_get_u8(&t, 0);
		q->size = fb_get_u32(&t, 1);
		q->ocm_offset = fb_get_u32(&t, 2);
		if (q->id > 1 || q->size < 32 || q->ocm_offset > npu->ocm_size ||
		    q->size > npu->ocm_size - q->ocm_offset)
			return -EINVAL;
	}
	model->nqueues = count;

	/* Runtime values {0 OCM usage, 1 OCM reserved, 2 entries}. */
	if (fb_get_table(&root, 6, &rtv))
		return -EINVAL;
	usage = fb_get_u32(&rtv, 0);
	reserved = fb_get_u32(&rtv, 1);
	if (usage > npu->ocm_size || reserved > npu->ocm_size - usage)
		return -ENOSPC;
	model->ocm_reserved = ALIGN(reserved, 32);
	if (npu_rtv_index(&rtv, "_ocm_base", &model->ocm_base_rtv))
		model->ocm_base_rtv = NPU_NO_RTV;

	/* IO {0 inputs, 1 outputs, 2 ring buffer, 3 params}. */
	if (fb_get_table(&root, 2, &io))
		return -EINVAL;
	ret = npu_parse_tensors(model, &io, &rtv, 0);
	model->ninputs = model->nio;
	if (!ret)
		ret = npu_parse_tensors(model, &io, &rtv, 1);
	if (ret)
		return ret;

	model->ring_rtv = NPU_NO_RTV;
	if (!fb_get_vector(&io, 2, 4, &start, &count) && count) {
		const char *name;
		u32 len;
		char buf[64];

		if (fb_vector_table(&io, start, 0, &t) || fb_get_string(&t, 0, &name, &len) ||
		    len >= sizeof(buf))
			return -EINVAL;
		memcpy(buf, name, len);
		buf[len] = '\0';
		if (npu_rtv_index(&rtv, buf, &model->ring_rtv))
			return -EINVAL;
		model->ring_size = fb_get_u32(&t, 2);
		if (model->ring_size > NPU_MAX_RING)
			return -E2BIG;
	}

	return npu_parse_params(npu, model, &root, &io, &rtv);
}

static void npu_free_model(struct npu_dev *npu, struct npu_model *model)
{
	unsigned int i;

	for (i = 0; i < model->nparams; i++) {
		struct npu_param *param = &model->params[i];

		kfree(param->data);
		if (param->ddr)
			dma_free_coherent(npu->dev, param->size, param->ddr, param->ddr_dma);
	}
	if (model->mcode)
		dma_free_coherent(npu->dev, model->mcode_size, model->mcode, model->mcode_dma);
	kfree(model);
}

/* The DDR ring buffer, as large as the largest model needs. */
static int npu_ensure_ring(struct npu_dev *npu, size_t size)
{
	void *ring;
	dma_addr_t dma;

	if (size <= npu->ring_size)
		return 0;
	ring = dma_alloc_coherent(npu->dev, size, &dma, GFP_KERNEL);
	if (!ring)
		return -ENOMEM;
	if (npu->ring)
		dma_free_coherent(npu->dev, npu->ring_size, npu->ring, npu->ring_dma);
	npu->ring = ring;
	npu->ring_dma = dma;
	npu->ring_size = size;
	return 0;
}

static int npu_create_handle(struct npu_dev *npu, struct file *file, void __user *arg)
{
	struct { u32 data, size, handle, attr, name, name_len; } req;
	struct npu_model *model;
	struct fb fb;
	void *blob;
	int ret, id;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;
	if (!req.size || req.size > NPU_MAX_MODEL)
		return -EINVAL;

	blob = vmalloc(req.size);
	model = kzalloc(sizeof(*model), GFP_KERNEL);
	if (!blob || !model) {
		ret = -ENOMEM;
		goto out;
	}
	if (copy_from_user(blob, npu_uptr(req.data), req.size)) {
		ret = -EFAULT;
		goto out;
	}

	fb.buf = blob;
	fb.len = req.size;
	model->owner = file;
	ret = npu_parse_model(npu, model, &fb);
	if (ret) {
		dev_info(npu->dev, "Model not taken: %d\n", ret);
		goto out;
	}

	mutex_lock(&npu->lock);
	ret = npu_ensure_ring(npu, model->ring_size);
	id = ret ? ret : idr_alloc(&npu->models, model, 1, 0, GFP_KERNEL);
	mutex_unlock(&npu->lock);
	if (id < 0) {
		ret = id;
		goto out;
	}

	if (put_user(id, (u32 __user *)npu_uptr(req.handle))) {
		mutex_lock(&npu->lock);
		idr_remove(&npu->models, id);
		mutex_unlock(&npu->lock);
		ret = -EFAULT;
		goto out;
	}

	dev_dbg(npu->dev, "Model %d: mcode %zu bytes, %u IO, %u params, ring %u\n", id,
		model->mcode_size, model->nio, model->nparams, model->ring_size);
	vfree(blob);
	return 0;

out:
	if (model)
		npu_free_model(npu, model);
	vfree(blob);
	return ret;
}

static int npu_destroy_handle(struct npu_dev *npu, struct file *file, u32 handle)
{
	struct npu_model *model;

	mutex_lock(&npu->lock);
	model = idr_find(&npu->models, handle);
	if (model && model->owner == file)
		idr_remove(&npu->models, handle);
	else
		model = NULL;
	mutex_unlock(&npu->lock);

	if (!model)
		return -EINVAL;
	npu_free_model(npu, model);
	return 0;
}

/* ---- Memory for libax_sys.so */

static void npu_buffer_release(struct kref *ref)
{
	struct npu_buffer *buffer = container_of(ref, struct npu_buffer, ref);
	struct device *dev = buffer->npu->dev;

	if (buffer->cached)
		dma_free_pages(dev, buffer->size, buffer->pages, buffer->dma, DMA_BIDIRECTIONAL);
	else
		dma_free_coherent(dev, buffer->size, buffer->cpu, buffer->dma);
	kfree(buffer);
}

static void npu_buffer_put(struct npu_buffer *buffer)
{
	kref_put(&buffer->ref, npu_buffer_release);
}

/* The allocated buffer holding [phys, phys + size), with the lock held. */
static struct npu_buffer *npu_find_buffer(struct npu_dev *npu, u64 phys, u64 size)
{
	struct npu_buffer *buffer;

	list_for_each_entry(buffer, &npu->buffers, node)
		if (phys >= buffer->dma && size <= buffer->size &&
		    phys - buffer->dma <= buffer->size - size)
			return buffer;
	return NULL;
}

static int npu_mem_alloc(struct npu_dev *npu, struct file *file, void __user *arg)
{
	struct npu_buffer *buffer;
	struct npu_mem req;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;
	if (!req.size || req.size > NPU_MAX_BUFFER)
		return -EINVAL;

	buffer = kzalloc(sizeof(*buffer), GFP_KERNEL);
	if (!buffer)
		return -ENOMEM;
	buffer->npu = npu;
	buffer->owner = file;
	buffer->size = PAGE_ALIGN(req.size);
	buffer->cached = req.flags & NPU_MEM_CACHED;
	if (buffer->cached) {
		buffer->pages = dma_alloc_pages(npu->dev, buffer->size, &buffer->dma,
						DMA_BIDIRECTIONAL, GFP_KERNEL);
		/* Write the zeroes back: the CPU may not have lines the NPU writes under. */
		if (buffer->pages)
			dma_sync_single_for_device(npu->dev, buffer->dma, buffer->size,
						   DMA_BIDIRECTIONAL);
	} else {
		buffer->cpu = dma_alloc_coherent(npu->dev, buffer->size, &buffer->dma, GFP_KERNEL);
	}
	if (!buffer->pages && !buffer->cpu) {
		kfree(buffer);
		return -ENOMEM;
	}
	kref_init(&buffer->ref);

	mutex_lock(&npu->lock);
	list_add(&buffer->node, &npu->buffers);
	mutex_unlock(&npu->lock);

	req.phys = buffer->dma;
	req.size = buffer->size;
	return copy_to_user(arg, &req, sizeof(req)) ? -EFAULT : 0;
}

static int npu_mem_free(struct npu_dev *npu, struct file *file, void __user *arg)
{
	struct npu_buffer *buffer;
	struct npu_mem req;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;

	mutex_lock(&npu->lock);
	buffer = npu_find_buffer(npu, req.phys, 1);
	if (buffer && (buffer->dma != req.phys || buffer->owner != file))
		buffer = NULL;
	if (buffer)
		list_del(&buffer->node);
	mutex_unlock(&npu->lock);

	if (!buffer)
		return -EINVAL;
	npu_buffer_put(buffer);
	return 0;
}

static void npu_sync(struct npu_dev *npu, struct npu_buffer *buffer, u64 phys, u32 size,
		     bool to_cpu)
{
	if (!buffer->cached || !size)
		return;
	if (to_cpu)
		dma_sync_single_for_cpu(npu->dev, phys, size, DMA_FROM_DEVICE);
	else
		dma_sync_single_for_device(npu->dev, phys, size, DMA_TO_DEVICE);
}

static int npu_mem_sync(struct npu_dev *npu, void __user *arg)
{
	struct npu_buffer *buffer;
	struct npu_mem req;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;

	mutex_lock(&npu->lock);
	buffer = npu_find_buffer(npu, req.phys, req.size);
	if (buffer)
		npu_sync(npu, buffer, req.phys, req.size, req.flags & NPU_MEM_TO_CPU);
	mutex_unlock(&npu->lock);
	return buffer ? 0 : -EINVAL;
}

static void npu_vma_open(struct vm_area_struct *vma)
{
	struct npu_buffer *buffer = vma->vm_private_data;

	kref_get(&buffer->ref);
}

static void npu_vma_close(struct vm_area_struct *vma)
{
	npu_buffer_put(vma->vm_private_data);
}

static const struct vm_operations_struct npu_vm_ops = {
	.open = npu_vma_open,
	.close = npu_vma_close,
};

/* A buffer is mapped whole, at its physical address as the offset. */
static int npu_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct npu_client *client = file->private_data;
	struct npu_dev *npu = client->npu;
	u64 phys = (u64)vma->vm_pgoff << PAGE_SHIFT;
	struct npu_buffer *buffer;
	int ret = -EINVAL;

	mutex_lock(&npu->lock);
	buffer = npu_find_buffer(npu, phys, vma->vm_end - vma->vm_start);
	if (buffer && buffer->dma == phys) {
		vma->vm_pgoff = 0;
		if (buffer->cached)
			ret = dma_mmap_pages(npu->dev, vma, buffer->size, buffer->pages);
		else
			ret = dma_mmap_coherent(npu->dev, vma, buffer->cpu, buffer->dma,
						buffer->size);
		if (!ret) {
			vma->vm_private_data = buffer;
			vma->vm_ops = &npu_vm_ops;
			kref_get(&buffer->ref);
		}
	}
	mutex_unlock(&npu->lock);
	return ret;
}

/* ---- Tasks */

static int npu_create_task(struct npu_dev *npu, struct file *file, void __user *arg)
{
	struct { u32 info, size; } req;
	struct { u32 handle, io, io_start, ext, task_out, callback, context; } info;
	struct { u32 buffers, count; } io;
	struct { u64 phys; u32 virt, size; } buf;
	struct npu_model *model;
	struct npu_task *task;
	unsigned int i, nio;

	if (copy_from_user(&req, arg, sizeof(req)) ||
	    copy_from_user(&info, npu_uptr(req.info), sizeof(info)) ||
	    copy_from_user(&io, npu_uptr(info.io), sizeof(io)))
		return -EFAULT;

	mutex_lock(&npu->lock);
	model = idr_find(&npu->models, info.handle);
	nio = model ? model->nio : 0;
	mutex_unlock(&npu->lock);
	if (!model || io.count < nio || io.count > NPU_MAX_IO)
		return -EINVAL;

	task = kzalloc(sizeof(*task), GFP_KERNEL);
	if (!task)
		return -ENOMEM;
	for (i = 0; i < nio; i++) {
		if (copy_from_user(&buf, npu_uptr(io.buffers) + i * sizeof(buf), sizeof(buf))) {
			kfree(task);
			return -EFAULT;
		}
		task->phys[i] = buf.phys;
	}
	task->owner = file;
	task->handle = info.handle;
	task->callback = info.callback;
	task->context = info.context;

	spin_lock(&npu->tasks_lock);
	task->id = ++npu->next_task;
	list_add_tail(&task->node, &npu->tasks);
	spin_unlock(&npu->tasks_lock);

	if (info.task_out && put_user(task->id, (u32 __user *)npu_uptr(info.task_out)))
		dev_dbg(npu->dev, "Task id not written back\n");
	wake_up(&npu->tasks_wait);
	return 0;
}

/*
 * Each DDR tensor must lie in memory this driver handed out; cached memory is
 * written back before the run and invalidated for the outputs after it.
 */
static int npu_task_buffers(struct npu_dev *npu, struct npu_model *model, struct npu_task *task,
			    struct npu_buffer **buffers)
{
	unsigned int i;

	for (i = 0; i < model->nio; i++) {
		const struct npu_io *io = &model->io[i];

		buffers[i] = NULL;
		if (io->type != NPU_IN_DDR)
			continue;
		buffers[i] = npu_find_buffer(npu, task->phys[i], max_t(u32, io->size, 1));
		if (!buffers[i]) {
			dev_dbg(npu->dev, "Task %u: %s at %#llx is not ours\n", task->id, io->name,
				task->phys[i]);
			return -EINVAL;
		}
		if (buffers[i]->cached)
			dma_sync_single_for_device(npu->dev, task->phys[i], io->size,
						   i < model->ninputs ? DMA_TO_DEVICE :
									DMA_FROM_DEVICE);
	}
	return 0;
}

/* Runtime values at the start of OCM, which the NPU sees at address 0. */
static void npu_load(struct npu_dev *npu, struct npu_model *model, struct npu_task *task)
{
	unsigned int i;

	if (model->ocm_base_rtv != NPU_NO_RTV)
		writel(npu_addr(model->ocm_reserved), npu->ocm + model->ocm_base_rtv * 4);
	for (i = 0; i < model->nio; i++)
		if (model->io[i].type == NPU_IN_DDR)
			writel(npu_addr(task->phys[i]), npu->ocm + model->io[i].rtv * 4);
	if (model->ring_rtv != NPU_NO_RTV)
		writel(npu_addr(npu->ring_dma), npu->ocm + model->ring_rtv * 4);
	for (i = 0; i < model->nparams; i++) {
		const struct npu_param *param = &model->params[i];

		if (param->type == NPU_IN_OCM)
			memcpy_toio(npu->ocm + param->ocm_offset, param->data, param->size);
		else
			writel(npu_addr(param->ddr_dma), npu->ocm + param->rtv * 4);
	}
}

static int npu_run(struct npu_dev *npu, struct npu_task *task)
{
	struct npu_buffer *buffers[NPU_MAX_IO];
	struct npu_model *model;
	u32 idle, mask = 0;
	unsigned int i;
	int ret;

	mutex_lock(&npu->lock);
	model = idr_find(&npu->models, task->handle);
	ret = model ? npu_task_buffers(npu, model, task, buffers) : -ENOENT;
	if (ret)
		goto out;
	if (!npu->hw_ready)
		npu_hw_init(npu);

	ret = readl_poll_timeout(npu->regs + NPU_DISPATCH + D_IDLE, idle, (idle & 5) == 5, 10, 500000);
	if (ret) {
		dev_err(npu->dev, "Dispatcher busy: %#x\n", idle);
		npu_hw_reset(npu);
		goto out;
	}
	npu_w(npu, NPU_SYNC + SM_EU_STATUS_CLR0, 0xffff);
	npu_load(npu, model, task);

	npu_w(npu, NPU_DISPATCH + D_OCM_BASE0, 0);
	npu_w(npu, NPU_DISPATCH + D_MCODE_BASE0, npu_addr(model->mcode_dma));
	for (i = 0; i < model->nqueues; i++) {
		const struct npu_queue *q = &model->queues[i];

		npu_w(npu, NPU_DISPATCH + (q->id ? D_QUEUE1_BASE : D_QUEUE0_BASE), q->ocm_offset >> 5);
		npu_w(npu, NPU_DISPATCH + (q->id ? D_QUEUE1_SIZE : D_QUEUE0_SIZE), (q->size >> 5) - 1);
		mask |= q->id ? BIT(2) : BIT(0);
	}

	reinit_completion(&npu->done);
	npu->irq_status = 0;
	npu->error_status = 0;
	npu_w(npu, NPU_DISPATCH + D_TRIGGER, mask);

	if (!wait_for_completion_timeout(&npu->done, NPU_TASK_TIMEOUT)) {
		dev_err(npu->dev, "Task %u timed out: idle %#x, error %#x\n", task->id,
			npu_r(npu, NPU_DISPATCH + D_IDLE), npu_r(npu, NPU_DISPATCH + D_ERROR));
		ret = -ETIMEDOUT;
	} else if (npu->error_status || npu->irq_status == ~0U) {
		dev_err(npu->dev, "Task %u failed: %#x %#x\n", task->id, npu->error_status,
			npu->irq_status);
		ret = -EIO;
	}
	if (ret) {
		npu_hw_reset(npu);
		goto out;
	}

	for (i = model->ninputs; i < model->nio; i++)
		if (buffers[i] && buffers[i]->cached)
			dma_sync_single_for_cpu(npu->dev, task->phys[i], model->io[i].size,
						DMA_FROM_DEVICE);
out:
	mutex_unlock(&npu->lock);
	return ret;
}

static void npu_finish(struct npu_task *task, int result)
{
	struct npu_client *client = task->owner->private_data;

	spin_lock(&client->lock);
	if (client->tail - client->head < ARRAY_SIZE(client->records)) {
		client->records[client->tail++ % ARRAY_SIZE(client->records)] = (struct npu_record){
			.task = task->id,
			.result = result ? NPU_TASK_FAILED : 0,
			.callback = task->callback,
			.context = task->context,
		};
	}
	spin_unlock(&client->lock);
	wake_up(&client->wait);
}

static int npu_worker(void *data)
{
	struct npu_dev *npu = data;
	struct npu_task *task;

	while (!kthread_should_stop()) {
		wait_event_interruptible(npu->tasks_wait,
					 !list_empty(&npu->tasks) || kthread_should_stop());
		spin_lock(&npu->tasks_lock);
		task = list_first_entry_or_null(&npu->tasks, struct npu_task, node);
		if (task)
			list_del(&task->node);
		spin_unlock(&npu->tasks_lock);
		if (!task)
			continue;

		npu_finish(task, npu_run(npu, task));
		fput(task->owner);
		kfree(task);
	}
	return 0;
}

static int npu_wait_tasks(struct npu_client *client, void __user *arg)
{
	struct npu_record records[4];
	u64 timeout_ms;
	u32 count = 0;
	long left;

	if (copy_from_user(&timeout_ms, arg + 0x58, sizeof(timeout_ms)))
		return -EFAULT;
	left = wait_event_interruptible_timeout(client->wait, client->head != client->tail,
						msecs_to_jiffies(min_t(u64, timeout_ms, 60000)));
	if (left < 0)
		return left;

	spin_lock(&client->lock);
	while (client->head != client->tail && count < ARRAY_SIZE(records))
		records[count++] = client->records[client->head++ % ARRAY_SIZE(client->records)];
	spin_unlock(&client->lock);

	if (!count)
		return -ETIMEDOUT;
	if (copy_to_user(arg, records, count * sizeof(records[0])) ||
	    copy_to_user(arg + 0x50, &count, sizeof(count)))
		return -EFAULT;
	return 0;
}

/* ---- The device file */

static int npu_open(struct inode *inode, struct file *file)
{
	struct npu_dev *npu = container_of(file->private_data, struct npu_dev, misc);
	struct npu_client *client;

	client = kzalloc(sizeof(*client), GFP_KERNEL);
	if (!client)
		return -ENOMEM;
	client->npu = npu;
	spin_lock_init(&client->lock);
	init_waitqueue_head(&client->wait);
	file->private_data = client;
	return 0;
}

static int npu_release(struct inode *inode, struct file *file)
{
	struct npu_client *client = file->private_data;
	struct npu_dev *npu = client->npu;
	struct npu_buffer *buffer, *next;
	struct npu_model *model;
	LIST_HEAD(freed);
	int id;

	mutex_lock(&npu->lock);
	idr_for_each_entry(&npu->models, model, id) {
		if (model->owner == file) {
			idr_remove(&npu->models, id);
			npu_free_model(npu, model);
		}
	}
	list_for_each_entry_safe(buffer, next, &npu->buffers, node)
		if (buffer->owner == file)
			list_move(&buffer->node, &freed);
	mutex_unlock(&npu->lock);

	list_for_each_entry_safe(buffer, next, &freed, node)
		npu_buffer_put(buffer);
	kfree(client);
	return 0;
}

static long npu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct npu_client *client = file->private_data;
	struct npu_dev *npu = client->npu;
	void __user *uarg = (void __user *)arg;
	u32 req[5] = { 0 };
	int ret;

	switch (cmd) {
	case NPU_INIT:
		if (copy_from_user(req, uarg, sizeof(req)))
			return -EFAULT;
		return req[0] == NPU_DISTRIBUTION_ONE ? 0 : -EOPNOTSUPP;
	case NPU_DEINIT:
		return 0;
	case NPU_GET_VNPU_NUM:
		return put_user(1, (u32 __user *)uarg);
	case NPU_GET_ATTR:
		req[0] = NPU_DISTRIBUTION_ONE;
		return copy_to_user(uarg, req, sizeof(req)) ? -EFAULT : 0;
	case NPU_CREATE_HANDLE:
		return npu_create_handle(npu, file, uarg);
	case NPU_DESTROY_HANDLE:
		return npu_destroy_handle(npu, file, (u32)arg);
	case NPU_GET_MODEL_TYPE:
	case NPU_HAS_DYNAMIC_PARAMS:
		if (copy_from_user(req, uarg, 8))
			return -EFAULT;
		return put_user(0, (u32 __user *)npu_uptr(req[1]));
	case NPU_GET_MODEL_CMM_INFO:
		if (copy_from_user(req, uarg, 8))
			return -EFAULT;
		memset(&req[2], 0, 12);
		return copy_to_user(npu_uptr(req[1]), &req[2], 12) ? -EFAULT : 0;
	case NPU_GET_AFFINITY:
		if (copy_from_user(req, uarg, 8))
			return -EFAULT;
		req[1] = 1;
		return copy_to_user(uarg, req, 8) ? -EFAULT : 0;
	case NPU_SET_AFFINITY:
		return 0;
	case NPU_CREATE_TASK:
		get_file(file);
		ret = npu_create_task(npu, file, uarg);
		if (ret)
			fput(file);
		return ret;
	case NPU_WAIT_TASKS0:
	case NPU_WAIT_TASKS1:
		return npu_wait_tasks(client, uarg);
	case NPU_CANCEL_TASK:
		return 0;
	case NPU_HARD_RESET:
		mutex_lock(&npu->lock);
		npu_hw_reset(npu);
		mutex_unlock(&npu->lock);
		return 0;
	case NPU_MEM_ALLOC:
		return npu_mem_alloc(npu, file, uarg);
	case NPU_MEM_FREE:
		return npu_mem_free(npu, file, uarg);
	case NPU_MEM_SYNC:
		return npu_mem_sync(npu, uarg);
	default:
		dev_dbg(npu->dev, "ioctl %#x not supported\n", cmd);
		return -ENOTTY;
	}
}

static const struct file_operations npu_fops = {
	.owner = THIS_MODULE,
	.open = npu_open,
	.release = npu_release,
	.unlocked_ioctl = npu_ioctl,
	.mmap = npu_mmap,
};

/* ---- Probe */

static void npu_release_memory(void *data)
{
	of_reserved_mem_device_release(data);
}

static int npu_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct npu_dev *npu;
	struct resource *res;
	int irq, ret;

	npu = devm_kzalloc(dev, sizeof(*npu), GFP_KERNEL);
	if (!npu)
		return -ENOMEM;
	npu->dev = dev;
	mutex_init(&npu->lock);
	idr_init(&npu->models);
	INIT_LIST_HEAD(&npu->buffers);
	spin_lock_init(&npu->tasks_lock);
	INIT_LIST_HEAD(&npu->tasks);
	init_waitqueue_head(&npu->tasks_wait);
	init_completion(&npu->done);

	npu->regs = devm_platform_ioremap_resource_byname(pdev, "regs");
	if (IS_ERR(npu->regs))
		return PTR_ERR(npu->regs);
	res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "ocm");
	npu->ocm = devm_ioremap_resource(dev, res);
	if (IS_ERR(npu->ocm))
		return PTR_ERR(npu->ocm);
	npu->ocm_size = resource_size(res);

	ret = dma_coerce_mask_and_coherent(dev, DMA_BIT_MASK(32));
	if (ret)
		return ret;
	ret = of_reserved_mem_device_init(dev);
	if (ret && ret != -ENODEV)
		return dev_err_probe(dev, ret, "Failed to take the NPU's memory\n");
	if (!ret) {
		ret = devm_add_action_or_reset(dev, npu_release_memory, dev);
		if (ret)
			return ret;
	}

	irq = platform_get_irq_byname(pdev, "normal0");
	if (irq < 0)
		return irq;
	ret = devm_request_irq(dev, irq, npu_irq_normal, 0, "npu", npu);
	if (ret)
		return ret;
	irq = platform_get_irq_byname(pdev, "error0");
	if (irq < 0)
		return irq;
	ret = devm_request_irq(dev, irq, npu_irq_error, 0, "npu-error", npu);
	if (ret)
		return ret;

	npu->worker = kthread_run(npu_worker, npu, "npu");
	if (IS_ERR(npu->worker))
		return PTR_ERR(npu->worker);

	npu->misc.minor = MISC_DYNAMIC_MINOR;
	npu->misc.name = "npu";
	npu->misc.fops = &npu_fops;
	npu->misc.parent = dev;
	ret = misc_register(&npu->misc);
	if (ret) {
		kthread_stop(npu->worker);
		return ret;
	}

	platform_set_drvdata(pdev, npu);
	return 0;
}

static void npu_remove(struct platform_device *pdev)
{
	struct npu_dev *npu = platform_get_drvdata(pdev);

	misc_deregister(&npu->misc);
	kthread_stop(npu->worker);
	if (npu->hw_ready)
		npu_hw_reset(npu);
	if (npu->ring)
		dma_free_coherent(npu->dev, npu->ring_size, npu->ring, npu->ring_dma);
	idr_destroy(&npu->models);
}

static const struct of_device_id npu_of_match[] = {
	{ .compatible = "axera,ax620e-npu" },
	{ }
};
MODULE_DEVICE_TABLE(of, npu_of_match);

static struct platform_driver npu_driver = {
	.probe = npu_probe,
	.remove = npu_remove,
	.driver = {
		.name = "ax620e-npu",
		.of_match_table = npu_of_match,
	},
};
module_platform_driver(npu_driver);

MODULE_DESCRIPTION("Axera AX620E NPU with the vendor's /dev/npu interface");
MODULE_LICENSE("GPL");
