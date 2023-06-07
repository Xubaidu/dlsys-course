buildDir := $(shell pwd)/build

.PHONY: build pybind clean format all tests

all: build

build:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE) -j$(shell nproc)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
	rm -rf $(buildDir)

tests:
	python3 -m pytest -v -k "matmul and cpu"
