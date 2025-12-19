#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<torch/extension.h>
#include<pybind11/pybind11.h>

namespace py = pybind11;
#define WARP_SIZE 32
// CUDA kernel for softmax computation
__global__ void softmax_naive_kernel(
    float* input, float* output, const int batchsize, const int dim) {
    // One thread per row
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_idx >= batchsize) return;

    float max = input[r_idx*dim + 0];
    for (int i=0 ;i<dim; i++){
        max = max > input[r_idx*dim + i] ? max : input[r_idx*dim + i];
    }

    float sum = 0.0;
    for (int i=0;i<dim; i++){
        sum += exp(input[r_idx*dim + i] - max);
    }

    for (int i=0;i<dim;i++){
        output[r_idx *dim + i] = exp(input[r_idx*dim + i] - max) / sum;
    }
}

// __global__ void softmax_naive_kernel(){
//     int blockIdx = blockIdx.x;
//     int threadIdx = threadIdx.x;
//     printf("blockIdx: %d, threadIdx: %d\n", blockIdx, threadIdx);
// }

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// Wrapper function following Elementwise.cu pattern
void softmax_baseline(torch::Tensor input, torch::Tensor output) {
    // Check input tensor dtype
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kFloat32);
    
    // Check if tensors are on CUDA
    if (!input.is_cuda() || !output.is_cuda()) {
        throw std::runtime_error("Input tensors must be on CUDA device");
    }
    
    // Get tensor dimensions
    const int ndim = input.dim();
    if (ndim != 2) {
        throw std::runtime_error("Input tensor must be 2-dimensional");
    }
    
    const int batchsize = input.size(0);
    const int dim = input.size(1);
    
    // Check output tensor dimensions match
    if (output.size(0) != batchsize || output.size(1) != dim) {
        throw std::runtime_error("Output tensor dimensions must match input tensor dimensions");
    }
    
    // Calculate grid and block dimensions
    dim3 block(3); // 256 threads per block
    dim3 grid((batchsize + block.x - 1) / block.x); // Calculate number of blocks needed
    
    // std::cout<<"#######@@@@ ####### grid: "<<grid.x<<", block: "<<block.x<<std::endl;
    // Launch CUDA kernel
    softmax_naive_kernel<<<grid, block>>>(
        reinterpret_cast<float*>(input.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        batchsize,
        dim
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(softmax_baseline);
}