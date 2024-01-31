#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>

const int matrixSize = 4096;
const int TILE_SIZE = 32;
const int COLS_PER_THREAD = 8;
const int WG_SIZE = 8;

//=================================================================================================================================================
//==================================== << BASIC UNOPTIMIZED (NATIVE FUNCTIONS) KERNEL >> ==========================================================
//=================================================================================================================================================

const char *basic = R"(
    __kernel void updateMatrix(__global double* input, __global double* output, const int n, const int iteration) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
            double newValue = 0.0;
            if (iteration % 2 == 1) {
                newValue = native_divide((input[(i-1)*n + (j-1)] + input[(i-1)*n + j] + input[i*n + (j-1)]), 3.0);
            } else {
                newValue = native_divide((input[(i+1)*n + (j+1)] + input[(i+1)*n + j] + input[i*n + (j+1)]), 3.0);
            }
            output[i*n + j] = newValue;
        } else if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
            output[i*n + j] = input[i*n + j];
        }
    }
)";

//=================================================================================================================================================
//==================================== << THREAD GRANULARIT OPTIMIZATION KERNEL >> ================================================================
//=================================================================================================================================================

const char *op_thread_row = R"(
    #define COLS_PER_THREAD 8
    __kernel void updateMatrix(__global double* input, __global double* output, const int n, const int iteration) {
        int row = get_global_id(0);
        int startCol = get_global_id(1) * COLS_PER_THREAD;

        for (int j = startCol; j < min(startCol + COLS_PER_THREAD, n - 1); ++j) {
            if (row > 0 && row < n - 1) {
                double newValue = 0.0;
                if (iteration % 2 == 1) {
                    newValue = (input[(row-1)*n + j-1] + input[(row-1)*n + j] + input[row*n + j-1]) / 3.0;
                } else {
                    newValue = (input[(row+1)*n + j+1] + input[(row+1)*n + j] + input[row*n + j+1]) / 3.0;
                }
                output[row*n + j] = newValue;
            } else if (row == 0 || row == n - 1) {
                output[row*n + j] = input[row*n + j];
            }
        }
    }
)";

//=================================================================================================================================================
//================================================ << LOCAL MEMORY KERNEL >> ======================================================================
//=================================================================================================================================================

const char *opt_local = R"(
    #define TILE_SIZE 32
    __kernel void updateMatrix(__global double* input, __global double* output, const int n, const int iteration) {
        __local double localMem[TILE_SIZE * TILE_SIZE];

        int globalRow = get_global_id(0);
        int globalCol = get_global_id(1);
        int localRow = get_local_id(0);
        int localCol = get_local_id(1);

        // Load data into local memory
        int localIndex = localRow * TILE_SIZE + localCol;
        int globalIndex = globalRow * n + globalCol;
        if (globalRow < n && globalCol < n) {
            localMem[localRow * TILE_SIZE + localCol] = input[globalIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform computation using local memory
        double newValue = 0.0;
        if (globalRow > 0 && globalRow < n - 1 && globalCol > 0 && globalCol < n - 1) {
            if (iteration % 2 == 1) {
                newValue = (localMem[(localRow-1) * TILE_SIZE + localCol-1] + localMem[(localRow-1) * TILE_SIZE + localCol] + localMem[localRow * TILE_SIZE + localCol-1]) / 3.0;
            } else {
                newValue = (localMem[(localRow+1) * TILE_SIZE + localCol+1] + localMem[(localRow+1) * TILE_SIZE + localCol] + localMem[localRow * TILE_SIZE + localCol+1]) / 3.0;
            }
            output[globalIndex] = newValue;
        } else {
            output[globalIndex] = input[localIndex];
        }

    }
)";


//================================================================================================================
//================================= << HELPERS >> ================================================================
//================================================================================================================

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

// Function to print the flat matrix
void printFlatMatrix(double* flatMatrix) {
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            std::cout << flatMatrix[i * matrixSize + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to print the matrix
void printMatrix(double** matrix) {
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to round up the global work size
size_t roundUp(int groupSize, int globalSize) {
    int r = globalSize % groupSize;
    return r == 0 ? globalSize : globalSize + groupSize - r;
}

// Sequential update function taken from provided file
void updateMatrix(double** matrix) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= 96; ++iter) {
        double** tempMatrix = new double*[matrixSize];
        for (int i = 0; i < matrixSize; ++i) {
            tempMatrix[i] = new double[matrixSize];
        }

        for (int i = 1; i < matrixSize - 1; ++i) {
            for (int j = 1; j < matrixSize - 1; ++j) {
                if (iter % 2 == 0) {
                    // Even iteration
                    tempMatrix[i][j] = ((matrix[i + 1][j + 1] + matrix[i][j + 1] + matrix[i + 1][j]) / 3.0);
                } else {
                    // Odd iteration
                    tempMatrix[i][j] = ((matrix[i - 1][j - 1] + matrix[i][j - 1] + matrix[i - 1][j]) / 3.0);
                }
            }
        }

        for (int i = 1; i < matrixSize - 1; ++i) {
            for (int j = 1; j < matrixSize - 1; ++j) {
                matrix[i][j] = tempMatrix[i][j];
            }
        }

        for (int i = 0; i < matrixSize; ++i) {
            delete[] tempMatrix[i];
        }
        delete[] tempMatrix;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Update time: " << duration.count() << " milliseconds" << std::endl;
}

// print the help command
void displayHelp(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --mode=<number>  Set the mode of the program (valid modes are 0 to 4). Default is 4.\n"
              << "  --help           Display this help message.\n"
              << "  --print          Set the program to print result matrices. Sending output to file recommended.\n";
}

//================================================================================================================
//================================= << END OF HELPERS >> =========================================================
//================================================================================================================

int main(int argc, char *argv[]) {
    int mode = 4;
    const char *kernelSource;
    cl::NDRange global, local;
    bool printMat = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help") {
            displayHelp(argv[0]);
            return 0;
        } else if (arg.find("--mode=") == 0) {
            mode = std::stoi(arg.substr(7));
            if (mode < 0 || mode > 4) {
                std::cerr << "Error: Invalid mode number. Valid modes are 0, 1, 2, 3, and 4.\n";
                std::exit(1);
            }
        } else if (arg == "--print") {
            printMat = true;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'. Use --help for usage information.\n";
            std::exit(1);
        }
    }                                                                                                            

    switch (mode)
    {
    case 1:
        {
            kernelSource = basic;
            global = cl::NDRange(matrixSize, matrixSize);
            local = cl::NullRange;
            break;
        }
    case 2:
        {
            kernelSource = op_thread_row;
            size_t globalWorkSize[2] = { matrixSize, (matrixSize + COLS_PER_THREAD - 1) / COLS_PER_THREAD };
            size_t localWorkSize[2] = { 1, 1 };  
            global = cl::NDRange(globalWorkSize[0], globalWorkSize[1]);
            local = cl::NDRange(localWorkSize[0], localWorkSize[1]);
            break;
        }
    case 3:
        {
            kernelSource = opt_local;
            size_t globalWorkSize[2] = { roundUp(TILE_SIZE, matrixSize), roundUp(TILE_SIZE, matrixSize) };
            size_t localWorkSize[2] = { TILE_SIZE, TILE_SIZE };
            global = cl::NDRange(globalWorkSize[0], globalWorkSize[1]);
            local = cl::NDRange(localWorkSize[0], localWorkSize[1]);
            break;
        }
    case 4:
        {
            kernelSource = basic;
            size_t localWorkSize[2] = { WG_SIZE, WG_SIZE }; 
            size_t globalWorkSize[2] = { roundUp(localWorkSize[0], matrixSize), roundUp(localWorkSize[1], matrixSize) };
            global = cl::NDRange(globalWorkSize[0], globalWorkSize[1]);
            local = cl::NDRange(localWorkSize[0], localWorkSize[1]);
            break;
        }
    default:
        {
            kernelSource = basic;
            size_t localWorkSize[2] = { WG_SIZE, WG_SIZE }; 
            size_t globalWorkSize[2] = { roundUp(localWorkSize[0], matrixSize), roundUp(localWorkSize[1], matrixSize) };
            global = cl::NDRange(globalWorkSize[0], globalWorkSize[1]);
            local = cl::NDRange(localWorkSize[0], localWorkSize[1]);
            break;
        }
    }                                                                                                                   
    
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
    
    if (mode == 0) {                    // sequential
        updateMatrix(matrix);

        if (printMat)
            printMatrix(matrix);
    } else {                            // parallel
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 1; iter <= 96; ++iter) {
            kernel.setArg(0, bufIn);
            kernel.setArg(1, bufOut);
            kernel.setArg(2, matrixSize);
            kernel.setArg(3, iter);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

            queue.finish();

            // Swap buffers
            std::swap(bufIn, bufOut);
        }
        queue.enqueueReadBuffer(bufIn, CL_TRUE, 0, matrixSize * matrixSize * sizeof(double), flatMatrix);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Update time: " << duration.count() << " milliseconds" << std::endl;

        if(printMat)
            printFlatMatrix(flatMatrix);
    }


    // Cleanup
    for (int i = 0; i < matrixSize; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] flatMatrix;

    return 0;
}
