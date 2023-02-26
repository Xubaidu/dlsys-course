#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

template<typename T>
class matrix_op {
    using Vector = std::vector<T>;
    using Matrix = std::vector< std::vector<T> >;

public:
    Matrix gemm(const Matrix& A, const Matrix& B)
    {
        int M = A.size();
        int N = B[0].size();
        int K = B.size();

        Matrix C(M, std::vector<T>(N));
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    Matrix exp(const Matrix& A)
    {
        Matrix C(A);
        for (int i = 0; i < C.size(); ++i) {
            for (int j = 0; j < C[0].size(); ++j) {
                C[i][j] = std::exp(C[i][j]);
            }
        }
        return C;
    }

    Vector sum_row(const Matrix& A)
    {
        Vector res(A.size());
        for (int i = 0; i < A.size(); ++i) {
            T sum = 0;
            for (int j = 0; j < A[0].size(); ++j) {
                sum += A[i][j];
            }
            res[i] = sum;
        }
        return res;
    }

    Matrix normalize(const Matrix& A)
    {
        Matrix C(A);
        Vector res = this->sum_row(A);
        for (int i = 0; i < C.size(); ++i) {
            for (int j = 0; j < C[0].size(); ++j) {
                C[i][j] /= res[i];
            }
        }
        return C;
    }

    Matrix transpose(const Matrix& A)
    {
        Matrix C(A[0].size(), std::vector<T>(A.size()));
        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < A[0].size(); ++j) {
                C[j][i] = A[i][j];
            }
        }
        return C;
    }

    Matrix elementwise(const Matrix& A, T v)
    {
        Matrix C(A);
        for (int i = 0; i < C.size(); ++i) {
            for (int j = 0; j < C[0].size(); ++j) {
                C[i][j] *= v;
            }
        }
        return C;
    }

    Matrix sub(const Matrix& A, const Matrix& B)
    {
        Matrix C(A);
        for (int i = 0; i < C.size(); ++i) {
            for (int j = 0; j < C[0].size(); ++j) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }
};

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    using uint8 = unsigned char;
    using fp32 = float;

    std::vector< std::vector<fp32> > t(n, std::vector<fp32>(k)); // n x k
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            t[i][j] = theta[i * k + j];
        }
    }

    const int iterations = (m + batch - 1) / batch;
    for (int i = 0; i < iterations; ++i) {
        
        std::vector< std::vector<fp32> > xx(batch, std::vector<fp32>(n)); // batch x n
        std::vector<uint8> yy(batch); // batch x 1

        for (int j = 0; j < batch; ++j) {
            int row = i * batch + j;
            for (int k = 0; k < n; ++k) {
                xx[j][k] = X[row * n + k];
            }
        }

        for (int j = 0; j < batch; ++j) {
            int row = i * batch + j;
            yy[j] = y[row];
        }

        matrix_op<fp32> np;
        auto Z = np.normalize(np.exp(np.gemm(xx, t))); // batch x k

        std::vector< std::vector<fp32> > Iy(batch, std::vector<fp32>(k));
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < k; ++j) {
                if (j == yy[i]) {
                    Iy[i][j] = 1.0;
                } else {
                    Iy[i][j] = 0.0;
                }
            }
        }

        auto grad = np.gemm(np.transpose(xx), np.sub(Z, Iy)); // n x k
        auto new_theta = np.elementwise(grad, (lr / batch));
        t = np.sub(t, new_theta);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            theta[i * k + j] = t[i][j];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
