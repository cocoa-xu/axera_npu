# Builds the NIF into the app's priv directory: the open runtime
# (c_src/axnpu.c), the OCR pipeline (c_src/ocr.c) and their Erlang interface.
# elixir_make sets MIX_APP_PATH and ERTS_INCLUDE_DIR; Nerves sets CC, CFLAGS
# and LDFLAGS for the target. It builds on other hosts too, where models can
# be described but not run.
#
# The library has no priv directory in its source, so priv here is a real
# directory per build (host, target) and never a link into the sources.

PREFIX = $(MIX_APP_PATH)/priv
BUILD = $(MIX_APP_PATH)/obj
NIF = $(PREFIX)/axera_npu_nif.so

CFLAGS ?= -O2
CFLAGS += -Wall -Wextra -std=gnu11 -fPIC -fvisibility=hidden -I$(ERTS_INCLUDE_DIR) -Ic_src
LDFLAGS += -shared

ifeq ($(CROSSCOMPILE)$(shell uname -s),Darwin)
LDFLAGS += -undefined dynamic_lookup -flat_namespace
endif

OBJECTS = $(BUILD)/axera_npu_nif.o $(BUILD)/axnpu.o $(BUILD)/ocr.o

all: $(NIF)

$(NIF): $(OBJECTS)
	@mkdir -p $(PREFIX)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

$(BUILD)/%.o: c_src/%.c c_src/axnpu.h c_src/ocr.h Makefile
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(NIF) $(BUILD)

.PHONY: all clean
