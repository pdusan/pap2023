#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>

const int matrixSize = 4096;

// OpenCL kernel to update the matrix
const char *kernelSource = R"(
    __kernel void updateMatrix(__global double* input, __global double* output, const int n, const int iteration) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
            double newValue = 0.0;
            if (iteration % 2 == 1) {
                newValue = (input[(i-1)*n + (j-1)] + input[(i-1)*n + j] + input[i*n + (j-1)]) / 3.0;
            } else {
                newValue = (input[(i+1)*n + (j+1)] + input[(i+1)*n + j] + input[i*n + (j+1)]) / 3.0;
            }
            output[i*n + j] = newValue;
        } else if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
            output[i*n + j] = input[i*n + j];
        }
    }
)";

void initializeMatrix(double** matrix) {
    for (int i = 0; i < matrixSize; ++i) {
        matrix[0][i] = (i % 2 == 0) ? 0.1 : 0.2;
        matrix[i][0] = (i % 2 == 0) ? 0.1 : 0.2;
        matrix[matrixSize - 1][i] = (i % 2 == 0) ? 0.2 : 0.1;
        matrix[i][matrixSize - 1] = (i % 2 == 0) ? 0.2 : 0.1;
    }
    for (int i = 1; i < matrixSize - 1; ++i) {
        for (int j = 1; j < matrixSize - 1; ++j) {
            matrix[i][j] = (i % 2 == 0) ? 1.0 : 0.0;
        }
    }
}

// Function to print the matrix
void printMatrix(double* flatMatrix) {
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            std::cout << flatMatrix[i * matrixSize + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Initialize the matrix
    double** matrix = new double*[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        matrix[i] = new double[matrixSize];
    }
    initializeMatrix(matrix);

    // Flatten the matrix for OpenCL
    double* flatMatrix = new double[matrixSize * matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            flatMatrix[i * matrixSize + j] = matrix[i][j];
        }
    }

    // OpenCL setup
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;
    sources.push_back({kernelSource, strlen(kernelSource)});
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(1);
    }

    // Allocate memory for matrices in OpenCL
    cl::Buffer bufIn(context, CL_MEM_READ_WRITE, matrixSize * matrixSize * sizeof(double));
    cl::Buffer bufOut(context, CL_MEM_READ_WRITE, matrixSize * matrixSize * sizeof(double));
    queue.enqueueWriteBuffer(bufIn, CL_TRUE, 0, matrixSize * matrixSize * sizeof(double), flatMatrix);

    // Execute the kernel
    cl::Kernel kernel(program, "updateMatrix");

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= 96; ++iter) {
        kernel.setArg(0, bufIn);
        kernel.setArg(1, bufOut);
        kernel.setArg(2, matrixSize);
        kernel.setArg(3, iter);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(matrixSize, matrixSize), cl::NullRange);
        queue.finish();

        // Swap buffers
        std::swap(bufIn, bufOut);
    }
    queue.enqueueReadBuffer(bufIn, CL_TRUE, 0, matrixSize * matrixSize * sizeof(double), flatMatrix);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Update time: " << duration.count() << " milliseconds" << std::endl;

    printMatrix(flatMatrix);

    // Cleanup
    for (int i = 0; i < matrixSize; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] flatMatrix;

    return 0;
}
