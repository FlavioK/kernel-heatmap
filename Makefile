.PHONY: all clean target_clean
HOST ?= tx2fk
EXE=testbench
REMOTE_TARGET = nvidia@$(HOST)
REMOTE_WORKING_DIR = ~/heatmap
REMOTE_OUTPUT_DIR ?=$(REMOTE_WORKING_DIR)/out
CFLAGS := -Wall -Werror -O3

NVCCFLAGS := -Xptxas -O3 --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--generate-code arch=compute_62,code=[compute_62,sm_62] \

#Export CUDA paths
export LIBRARY_PATH:=/usr/local/cuda/lib64:$(LIBRARY_PATH)
export LD_LIBRARY_PATH:=/usr/local/cuda/lib64:$(LD_LIBRARY_PATH)
export PATH:=/usr/local/cuda/bin:$(PATH)

build:
	mkdir -p $@

build/build.ninja: | build
	cd $(@D) && cmake \
	-G "Ninja" ..

$(EXE): build/build.ninja
	ninja -C build

all: $(EXE)

target_build: deploy
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && make clean && make all"

target_run: target_build
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && export LD_BIND_NOW && echo 'nvidia' | sudo -S python3 run.py"
	rsync -avz ${REMOTE_TARGET}:${REMOTE_OUTPUT_DIR} .

deploy:
		rsync -avz --exclude '.*' --exclude 'README.md' --exclude 'tags' . ${REMOTE_TARGET}:${REMOTE_WORKING_DIR}

clean:
	rm -f $(EXE)

target_clean:
	ssh $(REMOTE_TARGET) "rm -rf $(REMOTE_WORKING_DIR)"
