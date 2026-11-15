// SPDX-License-Identifier: Apache-2.0
/*
 * The NIF of AxeraNPU: models and OCR pipelines as resources, used one call
 * at a time each (a mutex), on dirty I/O schedulers since a call waits for
 * the NPU.
 */
#include <erl_nif.h>
#include <errno.h>
#include <string.h>

#include "axnpu.h"
#include "ocr.h"

#define RUN_TIMEOUT_MS 5000
#define MAX_LINES 512

struct model_resource {
	struct axnpu_model *model;
	ErlNifMutex *lock;
};

struct ocr_resource {
	struct ocr *ocr;
	ErlNifMutex *lock;
};

static ErlNifResourceType *model_type, *ocr_type;
static ERL_NIF_TERM atom_ok, atom_error, atom_f, atom_u, atom_s;

static void model_dtor(ErlNifEnv *env, void *obj)
{
	struct model_resource *r = obj;

	(void)env;
	axnpu_free(r->model);
	if (r->lock)
		enif_mutex_destroy(r->lock);
}

static void ocr_dtor(ErlNifEnv *env, void *obj)
{
	struct ocr_resource *r = obj;

	(void)env;
	ocr_free(r->ocr);
	if (r->lock)
		enif_mutex_destroy(r->lock);
}

static int load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info)
{
	(void)priv;
	(void)info;
	model_type = enif_open_resource_type(env, NULL, "axera_npu_model", model_dtor,
					     ERL_NIF_RT_CREATE, NULL);
	ocr_type = enif_open_resource_type(env, NULL, "axera_npu_ocr", ocr_dtor, ERL_NIF_RT_CREATE,
					   NULL);
	atom_ok = enif_make_atom(env, "ok");
	atom_error = enif_make_atom(env, "error");
	atom_f = enif_make_atom(env, "f");
	atom_u = enif_make_atom(env, "u");
	atom_s = enif_make_atom(env, "s");
	return model_type && ocr_type ? 0 : -1;
}

static ERL_NIF_TERM make_string(ErlNifEnv *env, const char *s)
{
	ERL_NIF_TERM term;
	size_t len = strlen(s);

	memcpy(enif_make_new_binary(env, len, &term), s, len);
	return term;
}

static ERL_NIF_TERM error_message(ErlNifEnv *env, const char *message)
{
	return enif_make_tuple2(env, atom_error, make_string(env, message));
}

static ERL_NIF_TERM error_errno(ErlNifEnv *env, int err)
{
	const char *name;

	switch (err) {
	case ETIMEDOUT:
		name = "etimedout";
		break;
	case EIO:
		name = "eio";
		break;
	case ENODEV:
		name = "enodev";
		break;
	case ENOMEM:
		name = "enomem";
		break;
	case EINVAL:
		name = "einval";
		break;
	default:
		name = "error";
	}
	return enif_make_tuple2(env, atom_error, enif_make_atom(env, name));
}

/* {kind, bits} as Nx names types. */
static ERL_NIF_TERM type_term(ErlNifEnv *env, int dtype)
{
	switch (dtype) {
	case AXNPU_FLOAT:
		return enif_make_tuple2(env, atom_f, enif_make_int(env, 32));
	case AXNPU_FLOAT16:
		return enif_make_tuple2(env, atom_f, enif_make_int(env, 16));
	case AXNPU_UINT8:
		return enif_make_tuple2(env, atom_u, enif_make_int(env, 8));
	case AXNPU_UINT16:
		return enif_make_tuple2(env, atom_u, enif_make_int(env, 16));
	case AXNPU_INT8:
		return enif_make_tuple2(env, atom_s, enif_make_int(env, 8));
	case AXNPU_INT16:
		return enif_make_tuple2(env, atom_s, enif_make_int(env, 16));
	case AXNPU_INT32:
		return enif_make_tuple2(env, atom_s, enif_make_int(env, 32));
	default:
		return enif_make_tuple2(env, atom_s, enif_make_int(env, 64));
	}
}

/* {name, type, shape, size} */
static ERL_NIF_TERM tensor_term(ErlNifEnv *env, const struct axnpu_tensor *t)
{
	ERL_NIF_TERM dims[AXNPU_MAX_DIMS];
	int i;

	for (i = 0; i < t->ndims; i++)
		dims[i] = enif_make_int64(env, t->dims[i]);
	return enif_make_tuple4(env, make_string(env, t->name), type_term(env, t->dtype),
				enif_make_tuple_from_array(env, dims, (unsigned)t->ndims),
				enif_make_uint64(env, t->size));
}

/* {inputs, outputs} */
static ERL_NIF_TERM tensors_term(ErlNifEnv *env, struct axnpu_model *model)
{
	ERL_NIF_TERM inputs = enif_make_list(env, 0), outputs = enif_make_list(env, 0);
	int i;

	for (i = axnpu_num_inputs(model) - 1; i >= 0; i--)
		inputs = enif_make_list_cell(env, tensor_term(env, axnpu_input(model, i)), inputs);
	for (i = axnpu_num_outputs(model) - 1; i >= 0; i--)
		outputs = enif_make_list_cell(env, tensor_term(env, axnpu_output(model, i)), outputs);
	return enif_make_tuple2(env, inputs, outputs);
}

/* model_load(axmodel) :: {:ok, ref, {inputs, outputs}} | {:error, message} */
static ERL_NIF_TERM model_load(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
	struct model_resource *r;
	struct axnpu_model *model;
	ErlNifBinary axmodel;
	ERL_NIF_TERM ref;
	char err[256];

	(void)argc;
	if (!enif_inspect_binary(env, argv[0], &axmodel))
		return enif_make_badarg(env);
	model = axnpu_load(axmodel.data, axmodel.size, err, sizeof(err));
	if (!model)
		return error_message(env, err);

	r = enif_alloc_resource(model_type, sizeof(*r));
	r->model = model;
	r->lock = enif_mutex_create("axera_npu_model");
	ref = enif_make_resource(env, r);
	enif_release_resource(r);
	if (!r->lock)
		return error_errno(env, ENOMEM);
	return enif_make_tuple3(env, atom_ok, ref, tensors_term(env, model));
}

/* model_describe(axmodel) :: {:ok, {inputs, outputs}} | {:error, message} */
static ERL_NIF_TERM model_describe(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
	struct axnpu_model *model;
	ErlNifBinary axmodel;
	ERL_NIF_TERM tensors;
	char err[256];

	(void)argc;
	if (!enif_inspect_binary(env, argv[0], &axmodel))
		return enif_make_badarg(env);
	model = axnpu_describe(axmodel.data, axmodel.size, err, sizeof(err));
	if (!model)
		return error_message(env, err);
	tensors = tensors_term(env, model);
	axnpu_free(model);
	return enif_make_tuple2(env, atom_ok, tensors);
}

/* model_run(ref, [binary]) :: {:ok, [binary]} | {:error, reason}; sizes already checked */
static ERL_NIF_TERM model_run(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
	struct model_resource *r;
	ERL_NIF_TERM list = argv[1], head, outputs;
	ErlNifBinary input;
	unsigned int count;
	int i, ret;

	(void)argc;
	if (!enif_get_resource(env, argv[0], model_type, (void **)&r) ||
	    !enif_get_list_length(env, list, &count) || (int)count != axnpu_num_inputs(r->model))
		return enif_make_badarg(env);

	enif_mutex_lock(r->lock);
	for (i = 0; enif_get_list_cell(env, list, &head, &list); i++) {
		struct axnpu_tensor *t = axnpu_input(r->model, i);

		if (!enif_inspect_binary(env, head, &input) || input.size != t->size) {
			enif_mutex_unlock(r->lock);
			return enif_make_badarg(env);
		}
		memcpy(t->data, input.data, input.size);
	}

	ret = axnpu_run(r->model, RUN_TIMEOUT_MS);
	if (ret) {
		enif_mutex_unlock(r->lock);
		return error_errno(env, -ret);
	}

	outputs = enif_make_list(env, 0);
	for (i = axnpu_num_outputs(r->model) - 1; i >= 0; i--) {
		struct axnpu_tensor *t = axnpu_output(r->model, i);
		ERL_NIF_TERM output;

		memcpy(enif_make_new_binary(env, t->size, &output), t->data, t->size);
		outputs = enif_make_list_cell(env, output, outputs);
	}
	enif_mutex_unlock(r->lock);
	return enif_make_tuple2(env, atom_ok, outputs);
}

/* ocr_load(det, rec, keys) :: {:ok, ref} | {:error, message} */
static ERL_NIF_TERM ocr_load_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
	ErlNifBinary det, rec, keys;
	struct ocr_resource *r;
	struct ocr *ocr;
	ERL_NIF_TERM ref;
	char err[256];

	(void)argc;
	if (!enif_inspect_binary(env, argv[0], &det) || !enif_inspect_binary(env, argv[1], &rec) ||
	    !enif_inspect_binary(env, argv[2], &keys))
		return enif_make_badarg(env);
	ocr = ocr_load(det.data, det.size, rec.data, rec.size, (const char *)keys.data, keys.size,
		       err, sizeof(err));
	if (!ocr)
		return error_message(env, err);

	r = enif_alloc_resource(ocr_type, sizeof(*r));
	r->ocr = ocr;
	r->lock = enif_mutex_create("axera_npu_ocr");
	ref = enif_make_resource(env, r);
	enif_release_resource(r);
	if (!r->lock)
		return error_errno(env, ENOMEM);
	return enif_make_tuple2(env, atom_ok, ref);
}

/* ocr_read(ref, rgb, width, height) :: {:ok, [{text, {x0, y0, x1, y1}, score, confidence}]} */
static ERL_NIF_TERM ocr_read_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
	struct ocr_resource *r;
	struct ocr_line *lines;
	ERL_NIF_TERM list;
	ErlNifBinary rgb;
	int width, height, n, i;

	(void)argc;
	if (!enif_get_resource(env, argv[0], ocr_type, (void **)&r) ||
	    !enif_inspect_binary(env, argv[1], &rgb) || !enif_get_int(env, argv[2], &width) ||
	    !enif_get_int(env, argv[3], &height) || width < 1 || height < 1 ||
	    rgb.size != (size_t)width * height * 3)
		return enif_make_badarg(env);

	lines = enif_alloc(sizeof(*lines) * MAX_LINES);
	if (!lines)
		return error_errno(env, ENOMEM);
	enif_mutex_lock(r->lock);
	n = ocr_read(r->ocr, rgb.data, width, height, lines, MAX_LINES);
	enif_mutex_unlock(r->lock);
	if (n < 0) {
		enif_free(lines);
		return error_errno(env, -n);
	}

	list = enif_make_list(env, 0);
	for (i = n - 1; i >= 0; i--) {
		const struct ocr_line *l = &lines[i];
		ERL_NIF_TERM box = enif_make_tuple4(env, enif_make_int(env, l->x0),
						    enif_make_int(env, l->y0), enif_make_int(env, l->x1),
						    enif_make_int(env, l->y1));

		list = enif_make_list_cell(env,
					   enif_make_tuple4(env, make_string(env, l->text), box,
							    enif_make_double(env, l->score),
							    enif_make_double(env, l->confidence)),
					   list);
	}
	enif_free(lines);
	return enif_make_tuple2(env, atom_ok, list);
}

static ErlNifFunc functions[] = {
	{ "model_load", 1, model_load, ERL_NIF_DIRTY_JOB_IO_BOUND },
	{ "model_describe", 1, model_describe, ERL_NIF_DIRTY_JOB_CPU_BOUND },
	{ "model_run", 2, model_run, ERL_NIF_DIRTY_JOB_IO_BOUND },
	{ "ocr_load", 3, ocr_load_nif, ERL_NIF_DIRTY_JOB_IO_BOUND },
	{ "ocr_read", 4, ocr_read_nif, ERL_NIF_DIRTY_JOB_IO_BOUND },
};

ERL_NIF_INIT(Elixir.AxeraNPU.NIF, functions, load, NULL, NULL, NULL)
