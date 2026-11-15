/* SPDX-License-Identifier: Apache-2.0 */
/*
 * An open runtime for Pulsar2 models (.axmodel) on the AX620E NPU, through
 * the open /dev/npu driver (kernel/ax620e-npu.c): no vendor libraries.
 *
 * A model owns DMA buffers for its inputs and outputs: fill the inputs, run,
 * read the outputs. The driver keeps the CPU caches in step around a run.
 * A model is not thread-safe; separate models run concurrently and the NPU
 * takes their tasks one at a time.
 */
#ifndef AXNPU_H
#define AXNPU_H

#include <stddef.h>
#include <stdint.h>

#define AXNPU_MAX_DIMS 8

/* ONNX TensorProto data types. */
enum axnpu_dtype {
	AXNPU_FLOAT = 1,
	AXNPU_UINT8 = 2,
	AXNPU_INT8 = 3,
	AXNPU_UINT16 = 4,
	AXNPU_INT16 = 5,
	AXNPU_INT32 = 6,
	AXNPU_INT64 = 7,
	AXNPU_FLOAT16 = 10,
};

struct axnpu_tensor {
	char name[64];
	int dtype;
	int ndims;
	int64_t dims[AXNPU_MAX_DIMS];
	size_t size;	/* bytes */
	void *data;	/* the DMA buffer; NULL for a described model */
};

struct axnpu_model;

/* Loads a model onto the NPU; on failure NULL and a reason in err. */
struct axnpu_model *axnpu_load(const void *axmodel, size_t size, char *err, size_t err_size);

/* Reads a model's inputs and outputs without the NPU; it cannot run. */
struct axnpu_model *axnpu_describe(const void *axmodel, size_t size, char *err, size_t err_size);

void axnpu_free(struct axnpu_model *model);

int axnpu_num_inputs(const struct axnpu_model *model);
int axnpu_num_outputs(const struct axnpu_model *model);
struct axnpu_tensor *axnpu_input(struct axnpu_model *model, int index);
struct axnpu_tensor *axnpu_output(struct axnpu_model *model, int index);

/* Runs the model on its input buffers: 0, or a negative errno. */
int axnpu_run(struct axnpu_model *model, int timeout_ms);

#endif
