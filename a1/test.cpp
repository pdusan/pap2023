#include <iostream>

const int SIZE = 12;
const int BLOCK_SIZE = 4;

int main() {
    int matrix[SIZE][SIZE];

    // Initialize the matrix with some values for demonstration
    for (int i = 0; i < SIZE; ++i) {
        std::cout << std::endl;
        for (int j = 0; j < SIZE; ++j)
        {
            matrix[i][j] = i * SIZE + j + 10;
            std::cout << matrix[i][j] << ' ';
        }
    }

    std::cout << std::endl
              << "Matrix in antidiagonal pattern:" << std::endl;

    // Traverse the matrix in antidiagonal pattern
    for (int sum = 0; sum <= 2 * (SIZE - 1); sum+=BLOCK_SIZE) {
        int steps = (int)sum / BLOCK_SIZE;

        for (int d = steps; d < 2 * BLOCK_SIZE - 1; d++) {
            for (int i = 0; i <= d; ++i)
            {
                int j = d - i;
                if (i < SIZE && j < SIZE)
                {
                    std::cout << matrix[i][j] << " ";
                }
            }
        }
    }

    return 0;
}
