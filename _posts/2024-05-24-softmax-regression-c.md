# Softmax Regression in C, from scratch

This doc shows C code to train a softmax regression model on MNIST data.  
<br>
The softmax regression model is a linear regression + softmax loss (cross-entropy loss).

## 1. utilities
```c
#include <cmath>
#include <iostream>

void transpose(const float *A, float *A_T, size_t m, size_t n) {
    /**
     * Transpose matrix A of size m*n to matrix A_T of size n*m;
     *
     * Return:
     *      None.
     */
     for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            A_T[j*m + i] = A[i*n + j];
        }
     }
}

void softmax(float *H, size_t m, size_t k) {
    /**
     * Compute softmax on the matrix H of size m*k, in place.
     *
     * Return:
     *      None.
     */
     for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;

        // exp
        for (size_t j = 0; j < k; j++) {
            H[i*k + j] = exp(H[i*k + j]);
            sum += H[i*k + j];
        }

        // normalize
        for (size_t j = 0; j < k; j++) {
            H[i*k + j] = H[i*k + j] / sum;
        }
     }
}

void matmul(const float *A, const float *B, float* C, size_t m, size_t n, size_t k) {
    /**
     * compute the multiplication of matrix A and matrix B,
     * and output the result into C.
     *
     * Args:
     *      A of size m*n
     *      B of size n*k
     *      C of size m*k
     * Return:
     *      None.
     */
     for (size_t i = 0; i < m; i +=1) {
        for (size_t j = 0; j < k; j++) {
            // C[i,j] is the production of row vector A[i:] and col vector B[:j]
            C[i*k + j] = 0;
            for (size_t q = 0; q < n; q++) {
                C[i*k + j] += A[i*n + q] * B[j + q*k];
            }
        }
     }
}
```

## 2. train in 1 epoch
```c
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

    for (int i=0; i < m; i+=batch) {
        const float *X_minibatch = &(X[i*n]);
        const unsigned char *y_minibatch = &(y[i]);

        // compute H = X_minibatch @ theta, of size batch*k
        float* H = (float*)malloc(batch * k * sizeof(float));
        matmul(X_minibatch, theta, H, batch, n, k);

        // compute softmax on H
        softmax(H, batch, k); // softmax on H

        // compute I_y
        float* I_y = (float*)malloc(batch * k * sizeof(float));
        for (size_t p = 0; p < batch; p++) {
            for (size_t q = 0; q < k; q++) {
                if (q == (size_t)y_minibatch[p]) {
                    I_y[p*k + q] = 1.0f;
                } else {
                    I_y[p*k + q] = 0.0f;
                }

                // compute (H - I_y)
                H[p*k + q] -= I_y[p*k + q];
            }
        }

        // compute gradient = X_minibatch.T @ H, of size n*k
        float* X_minibatch_T = (float*)malloc(batch * n * sizeof(float));
        transpose(X_minibatch, X_minibatch_T, batch, n);

        float* gradient = (float*)malloc(n * k * sizeof(float));
        matmul(X_minibatch_T, H, gradient, n, batch, k);

        // SGD: update theta
        for (size_t p = 0; p < n; p++) {
            for (size_t q = 0; q < k; q++) {
                theta[p*k + q] -= lr * gradient[p*k + q] / batch;
            }
        }

        // de-malloc H, I_y, gradient, X_minibatch_T
        free(H);
        free(I_y);
        free(gradient);
        free(X_minibatch_T);
    }
}
```

## 3. use `pybind11` to expose C API to Python API
```c
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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
```

## 4. feel free to call `softmax_regression_epoch_cpp` API in your python code.
