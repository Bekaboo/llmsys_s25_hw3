#!/usr/bin/env sh

for kern in combine softmax_kernel layernorm_kernel; do
	nvcc -o "minitorch/cuda_kernels/$kern.so" \
		--shared "src/$kern.cu" \
		-Xcompiler \
		-fPIC \
		-I $CONDA_PREFIX/targets/x86_64-linux/include
done
