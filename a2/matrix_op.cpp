#include <iostream>
#include <chrono>
#include <limits>

const int matrixSize = 4096;

// Function to initialize the matrix based on the specified pattern
void initializeMatrix(double** matrix) {
    // Fill the matrix as per the initial requirements

    // Fill the first row and first column
    for (int i = 0; i < matrixSize; ++i) {
        matrix[0][i] = (i % 2 == 0) ? 0.1 : 0.2;
        matrix[i][0] = (i % 2 == 0) ? 0.1 : 0.2;
        matrix[matrixSize - 1][i] = (i % 2 == 0) ? 0.2 : 0.1;
        matrix[i][matrixSize - 1] = (i % 2 == 0) ? 0.2 : 0.1;
    }

    // Fill the remaining cells
    for (int i = 1; i < matrixSize - 1; ++i) {
        for (int j = 1; j < matrixSize - 1; ++j) {
            matrix[i][j] = (i % 2 == 0) ? 1.0 : 0.0;
        }
    }
}

// Function to update the matrix according to the specified rules
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

// Function to print the matrix
void printMatrix(double** matrix) {
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Initialize a 4192x4192 matrix on the heap
    double** matrix = new double*[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        matrix[i] = new double[matrixSize];
    }

    // Initialize the matrix
    initializeMatrix(matrix);

    // Call the updateMatrix function
    updateMatrix(matrix);

    // Print the updated matrix (optional)
    printMatrix(matrix);

    // Deallocate the memory
    for (int i = 0; i < matrixSize; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}
