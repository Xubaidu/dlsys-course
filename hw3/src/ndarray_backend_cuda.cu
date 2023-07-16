#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
    CudaArray(const size_t size) {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
    }
    ~CudaArray() {
        cudaFree(ptr);
    }
    size_t ptr_as_int() {
        return (size_t)ptr;
    }

    scalar_t *ptr;
    size_t size;
};

struct CudaDims {
    dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
    /**
     * Utility function to get cuda dimensions for 1D call
     */
    CudaDims dim;
    size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    dim.block = dim3(BASE_THREAD_NUM, 1, 1);
    dim.grid = dim3(num_blocks, 1, 1);
    return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
    uint32_t size;
    uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t> &x) {
    CudaVec shape;
    if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
    shape.size = x.size();
    for (size_t i = 0; i < x.size(); i++) {
        shape.data[i] = x[i];
    }
    return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val) {
    CudaDims dim = CudaOneDim(out->size);
    FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t transform(size_t tid, CudaVec shape, CudaVec strides, size_t offset) {
	CudaVec indices;
    indices.size = shape.size;
	size_t cur_stride = 1, pre_stride = 1;
	for (int i = shape.size - 1; i >= 0; --i) {
		pre_stride *= shape.data[i];
		indices.data[i] = tid % pre_stride / cur_stride;
		cur_stride = pre_stride;
	}
	size_t idx = offset;
	for (int i = 0; i < shape.size; ++i) {
		idx += strides.data[i] * indices.data[i];
	}
	return idx;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
    /**
     * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
     * non-compact input a, to the corresponding item (at location gid) in the compact array out.
     *
     * Args:
     *   a: CUDA pointer to a array
     *   out: CUDA point to out array
     *   size: size of out array
     *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
     *   strides: vector of strides of out array
     *   offset: offset of out array
     */
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    /// BEGIN YOUR SOLUTION
    if (tid < size) {
        out[tid] = a[transform(tid, shape, strides, offset)];
    }
    /// END YOUR SOLUTION
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
    /**
     * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
     * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
     * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
     * the functions after this, however, you'll need to define these kernels as you see fit to
     * execute the underlying function.
     *
     * Args:
     *   a: non-compact represntation of the array, given as input
     *   out: compact version of the array to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *a* array (not out, which has compact strides)
     *   offset: offset of the *a* array (not out, which has zero offset, being compact)
     */

    // Nothing needs to be added here
    CudaDims dim = CudaOneDim(out->size);
    CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                           VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

    /// BEGIN YOUR SOLUTION
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[transform(tid, shape, strides, offset)] = a[tid];
    }
    /// END YOUR SOLUTION
}

void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
    /**
     * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
     * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
     *
     * Args:
     *   a: _compact_ array whose items will be written to out
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *out* array (not a, which has compact strides)
     *   offset: offset of the *out* array (not a, which has zero offset, being compact)
     */
    /// BEGIN YOUR SOLUTION
	CudaDims dim = CudaOneDim(a.size);
	EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
    /// END YOUR SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t *out, CudaVec shape,
                              CudaVec strides, size_t offset) {

    /// BEGIN YOUR SOLUTION
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[transform(tid, shape, strides, offset)] = val;
    }
    /// END YOUR SOLUTION
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
    /**
     * Set items is a (non-compact) array
     *
     * Args:
     *   size: number of elements to write in out array (note that this will note be the same as
     *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
     *         product of items in shape, but covenient to just pass it here.
     *   val: scalar value to write to
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension of out
     *   strides: strides of the out array
     *   offset: offset of the out array
     */
    /// BEGIN YOUR SOLUTION
	CudaDims dim = CudaOneDim(out->size);
	ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), offset);
    /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out) {
    /**
     * Add together two CUDA array
     */
    CudaDims dim = CudaOneDim(out->size);
    EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out) {
    /**
     * Add together a CUDA array and a scalar value.
     */
    CudaDims dim = CudaOneDim(out->size);
    ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION

#define Mul(a, b) ((a) * (b))
#define Div(a, b) ((a) / (b))
#define Power(a, b) (pow(a, b))
#define Maximum(a, b) ((a) > (b) ? (a) : (b)) // cuda does not support std::max
#define Eq(a, b) (static_cast<float>(a == b))
#define Ge(a, b) (static_cast<float>(a >= b))
#define Log(a) (log(a))
#define Exp(a) (exp(a))
#define Tanh(a) (tanh(a))

#define EwiseOpKernel(op)                                                                                 \
    __global__ void Ewise##op##Kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size) { \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
        if (gid < size) out[gid] = op(a[gid], b[gid]);                                                    \
    }

#define EwiseOp(op)                                                                    \
    void Ewise##op(const CudaArray &a, const CudaArray &b, CudaArray *out) {           \
        CudaDims dim = CudaOneDim(out->size);                                          \
        Ewise##op##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
    }

#define ScalarOpKernel(op)                                                                            \
    __global__ void Scalar##op##Kernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size) { \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                           \
        if (gid < size) out[gid] = op(a[gid], val);                                                   \
    }

#define ScalarOp(op)                                                                  \
    void Scalar##op(const CudaArray &a, scalar_t val, CudaArray *out) {               \
        CudaDims dim = CudaOneDim(out->size);                                         \
        Scalar##op##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
    }

#define SingleEwiseOpKernel(op)                                                        \
    __global__ void Ewise##op##Kernel(const scalar_t *a, scalar_t *out, size_t size) { \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                            \
        if (gid < size) out[gid] = op(a[gid]);                                         \
    }

#define SingleEwiseOp(op)                                                       \
    void Ewise##op(const CudaArray &a, CudaArray *out) {                        \
        CudaDims dim = CudaOneDim(out->size);                                   \
        Ewise##op##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
    }

#define Ewise(op)     \
    EwiseOpKernel(op) \
        EwiseOp(op)

#define Scalar(op)     \
    ScalarOpKernel(op) \
        ScalarOp(op)

#define SingleEwise(op)     \
    SingleEwiseOpKernel(op) \
        SingleEwiseOp(op)

Ewise(Mul)
Ewise(Div)
Ewise(Maximum)
Ewise(Eq)
Ewise(Ge)

Scalar(Mul)
Scalar(Div)
Scalar(Maximum)
Scalar(Eq)
Scalar(Ge)
Scalar(Power)

SingleEwise(Log)
SingleEwise(Exp)
SingleEwise(Tanh)

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void NaiveGemm(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t K, uint32_t N) {
    size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < M && tid_y < N) {
        scalar_t temp = 0;
        for (size_t i = 0; i < K; ++i) {
            // out[tid_x][tid_y] += a[tid_x][i] * b[i][tid_y]
            temp += a[tid_x * K + i] * b[i * N + tid_y];
        }
        out[tid_x * N + tid_y] = temp;
    }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t K,
            uint32_t N) {
    CudaDims dim;
    dim.block = dim3(16, 16, 1);
    dim.grid = dim3((M + 15) / 16, (N + 15) / 15, 1);
    NaiveGemm<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, K, N);
}

#define OFFSET(i, j, stride) ((i) * (stride) + (j))

template<const int bm, const int bn, const int bk, const int rm, const int rn>
__global__ void TiledGemm(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t K)
{
    __shared__ scalar_t smem_a[bm][bk];
    __shared__ scalar_t smem_b[bn][bk];
    scalar_t ldg_b[bn * bk];
    scalar_t reg_a[rm], reg_b[rn];
    scalar_t reg_c[rm][rn];

    // The logical dim and strides of the above several buffers:
    // - a[M x K]
    // - b[K x N]
    // - smem_a[bm][bk]
    // - smem_b[bn][bk]
    // - reg_a[rm]
    // - reg_b[rn]
    // - reg_c[rm][rn]

    size_t bx = blockIdx.x, by = blockIdx.y;
    size_t tx = threadIdx.x, ty = threadIdx.y;
    for (size_t k_outer = 0; k_outer < K; k_outer += bk) {

        // load data from gmem to smem: a -> smem_a and b -> smem_b
        // smem_a = a[bx : bx + bm][k_outer : k_outer + bk]
        // smem_b = b[by : by + bn][k_outer : k_outer + bk], need to be transposed

        for (size_t x_outer = 0; x_outer < bm; ++x_outer) {
            for (size_t y_outer = 0; y_outer < bk; ++y_outer) {
                smem_a[x_outer][y_outer] = a[OFFSET(bx + x_outer, k_outer + y_outer, bk)];
            }
        }
        if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
            for (int i = 0; i < bm; ++i) {
                for (int j = 0; j < bk; ++j) {
                    printf("%f%c", smem_a[i][j], " \n"[j == bk - 1]);
                }
            }
            printf("\n");
        }

        for (size_t x_outer = 0; x_outer < bk; ++x_outer) {
            for (size_t y_outer = 0; y_outer < bn; ++y_outer) {
                ldg_b[OFFSET(x_outer, y_outer, bn)] = b[OFFSET(by + x_outer, k_outer + y_outer, bn)];
            }
            for (size_t y_outer = 0; y_outer < bn; ++y_outer) {
                smem_b[y_outer][x_outer] = ldg_b[OFFSET(x_outer, y_outer, bn)];
            }
        }
        __syncthreads();

        for (size_t k_inner = 0; k_inner < bk; ++k_inner) {
            // load data from smem to register: smem_a -> reg_a and smem_b -> reg_b
            // reg_a = smem_a[tx : tx + rm][k_inner : k_inner + 1]
            // reg_b = smem_b[ty : ty + rn][k_inner : k_inner + 1]
            for (size_t x_inner = 0; x_inner < rm; ++x_inner) {
                reg_a[x_inner] = smem_a[tx + x_inner][k_inner];
            }

            if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
                for (int i = 0; i < rm; ++i) {
                    printf("%f%c", reg_a[i], " \n"[i == rm - 1]);
                }
                printf("\n");
            }
            
            for (size_t x_inner = 0; x_inner < rn; ++x_inner) {
                reg_b[x_inner] = smem_b[ty + x_inner][k_inner];
            }

            // compute in register-level
            for (size_t i = 0; i < rm; ++i) {
                for (size_t j = 0; j < rn; ++j) {
                    reg_c[i][j] = reg_a[i] * reg_b[j];
                }
            }
        }
    }

    // out[xbase : xbase + rm, ybase : ybase + rn] = c[:]
    size_t xbase = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ybase = blockIdx.y * blockDim.y + threadIdx.y;
    for (size_t i = 0; i < rm; ++i) {
        for (size_t j = 0; j < rn; ++j) {
            out[OFFSET(xbase + i, ybase + j, rn)] += reg_c[i][j];
        }
    }
}

void Tiled_Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t K,
            uint32_t N) {
    /**
     * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
     * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
     * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
     * over (i,j) entries in the output array.  However, to really get the full benefit of this
     * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
     * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
     * the CPU backend, here you should implement a single function that works across all size
     * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
     * implementations, this function here will largely just set up the kernel call, and you should
     * implement the logic in a separate MatmulKernel() call.
     *
     *
     * Args:
     *   a: compact 2D array of size m x k
     *   b: comapct 2D array of size k x n
     *   out: compact 2D array of size m x n to write the output to
     *   M: rows of a / out
     *   K: columns of a / rows of b
     *   N: columns of b / out
     */

    /// BEGIN YOUR SOLUTION
    
    constexpr int bm = 64;
    constexpr int bn = 64;
    constexpr int bk = 16;
    constexpr int rm = 4;
    constexpr int rn = 4;

    CudaDims dim;
    dim.block = dim3(bm / rm, bn / rn, 1);
    dim.grid = dim3((M + bm - 1) / bm, (N + bn - 1) / bn, 1);
    TiledGemm<bm, bn, bk, rm, rn><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, K, N);
    /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t reduce_num, size_t reduce_size) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < reduce_num) {
        size_t offset = reduce_size * tid;
        scalar_t val = a[offset];
        for (int i = 1; i < reduce_size; ++i) {
            val = max(val, a[offset + i]);
        }
        out[tid] = val;
    }
}

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
    /**
     * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
     * for simplicity you can perform each reduction in a single CUDA thread.
     *
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   redice_size: size of the dimension to reduce over
     */
    /// BEGIN YOUR SOLUTION
    size_t reduce_num = out->size;
    CudaDims dim = CudaOneDim(reduce_num);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_num, reduce_size);
    /// END YOUR SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t reduce_num, size_t reduce_size) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < reduce_num) {
        size_t offset = reduce_size * tid;
        scalar_t sum = a[offset];
        for (int i = 1; i < reduce_size; ++i) {
            sum += a[offset + i];
        }
        out[tid] = sum;
    }
}
void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
    /**
     * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
     * can perform each reduction in a single CUDA thread.
     *
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   redice_size: size of the dimension to reduce over
     */
    /// BEGIN YOUR SOLUTION
    size_t reduce_num = out->size;
    CudaDims dim = CudaOneDim(reduce_num);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_num, reduce_size);
    /// END YOUR SOLUTION
}

}
} // namespace needle::cuda

PYBIND11_MODULE(ndarray_backend_cuda, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace cuda;

    m.attr("__device_name__") = "cuda";
    m.attr("__tile_size__") = TILE;

    py::class_<CudaArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &CudaArray::size)
        .def("ptr", &CudaArray::ptr_as_int);

    // return numpy array, copying from CPU
    m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                         size_t offset) {
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                       [](size_t &c) { return c * ELEM_SIZE; });

        // copy memory to host
        scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
        if (host_ptr == 0) throw std::bad_alloc();
        cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

        // return numpy array
        py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
        return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
    });

    // copy numpy array to GPU
    m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
        cudaError_t err =
            cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    });

    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);

    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);

    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);

    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    m.def("matmul", Matmul);
    m.def("matmul_tiled", Tiled_Matmul);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
