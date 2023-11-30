#include <iostream>

const int SIZE = 7;
const int BLOCK_SIZE = 1;

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

    for (int D = 0; D < 2 * SIZE - 1 - BLOCK_SIZE; D+=BLOCK_SIZE) {

        int steps = (int)D / BLOCK_SIZE;

        // std::cout<<"steps: " << steps << std::endl;

        for (int p = std::max((steps - BLOCK_SIZE) * BLOCK_SIZE, 0); p <= BLOCK_SIZE * steps; p+=BLOCK_SIZE) {
            if (p < SIZE-1){
                int q = std::abs(p - BLOCK_SIZE * steps);
                if (q < SIZE-1) {

                    // std::cout << "p: " << p << " q: " << q << std::endl;
    
                    for (int d = 0; d < 2 * BLOCK_SIZE - 1;d++) {
                        // std::cout<<std::endl << "d: " << d << std::endl;
                        for (int i = std::max(d - BLOCK_SIZE, 0); i <= d; i++)
                        {
                            int j = d - i;

                            // std::cout << "i: " << i << " j: " << j << std::endl;
    
                            if (i < BLOCK_SIZE && j < BLOCK_SIZE && i >= 0 && j >= 0) {
                                int m = i + p +1;
                                int n = j + q +1;
    
                                // std::cout << "m: " << m << " n: " << n << std::endl;
                                std::cout<< ' ' << matrix[n][m];
                            }
                        }
                    }
                }
            }
        }
    }

        return 0;
}
