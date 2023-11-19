#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float **S, float **SUB, std::unordered_map<char, int> &cmap)
{
	unsigned long visited = 0;
	gap_penalty = SUB[0][cmap['*']]; // min score

	// Boundary
	for (unsigned int i = 1; i < rows; i++)
	{
		S[i][0] = i * gap_penalty;
		visited++;
	}

	for (unsigned int j = 0; j < cols; j++)
	{
		S[0][j] = j * gap_penalty;
		visited++;
	}

	// Main part
	for (unsigned int i = 1; i < rows; i++)
	{
		for (unsigned int j = 1; j < cols; j++)
		{
			float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});

			visited++;
		}
	}

	return visited;
}

unsigned long SequenceInfo::gpsa_taskloop(float **S, float **SUB, std::unordered_map<char, int> cmap, int grain_size = 1)
{
	// Your work goes here, paralellize the code below using OpenMP taskloop
	unsigned long visited = 0;
	gap_penalty = SUB[0][cmap['*']]; // min score

	// Boundary
	#pragma omp parallel
	{
		#pragma omp single
		{
			#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) shared(S, gap_penalty) 
			for (unsigned int i = 1; i < rows; i++)
			{
				S[i][0] = i * gap_penalty;
				visited++;
			}

			#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) shared(S, gap_penalty)
			for (unsigned int j = 0; j < cols; j++)
			{
				S[0][j] = j * gap_penalty;
				visited++;
			}
		}
	}
	
	// Main part
	// #pragma omp parallel
	// {
	// 	#pragma omp single
	// 	#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) shared(S, SUB, cmap)
	// 	for (unsigned int i = 1; i < rows; i++)
	// 	{
	// 		for (unsigned int j = 1; j < cols; j++)
	// 		{
	// 			float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
	// 			float del = S[i - 1][j] + gap_penalty;
	// 			float insert = S[i][j - 1] + gap_penalty;
	// 			#pragma omp critical
	// 			S[i][j] = std::max({match, del, insert});
		
	// 			visited++;
	// 		}
	// 	}
	// }



	// INTERESTING TEST HERE

	#pragma omp parallel 
	{
		float **localS;

		#pragma omp single
		{
			localS = new float*[omp_get_num_threads()];
			for (unsigned int i = 0; i < omp_get_num_threads(); ++i)
				localS[i] = new float[cols];
		}

		int threadId = omp_get_thread_num();
		#pragma omp tasklopp reduction(+ : visited) shared(S, localS, gap_penalty) firstprvate(rows, cols, SUB, cmap, X, Y)
		for (unsigned int i = 1; i < rows; i++) {
            for (unsigned int j = 1; j < cols; j++) {
                float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
                float del = S[i - 1][j] + gap_penalty;
                float insert = S[i][j - 1] + gap_penalty;
                localS[threadId][j] = std::max({match, del, insert});
                visited++;
            }
        }

		#pragma omp single
		{
			 // Combine the local copies into the global matrix
                for (unsigned int i = 1; i < rows; i++) {
                    for (unsigned int j = 1; j < cols; j++) {
                        S[i][j] = localS[0][j];
                    }
                }
                delete[] localS;
		}

	}

	return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float **S, float **SUB, std::unordered_map<char, int> cmap, int grain_size = 1)
{
	// Your work goes here, paralellize the code below using OpenMP tasks
	unsigned long visited = 0;
	gap_penalty = SUB[0][cmap['*']]; // min score

	// Boundary
	for (unsigned int i = 1; i < rows; i++)
	{
		S[i][0] = i * gap_penalty;
		visited++;
	}

	for (unsigned int j = 0; j < cols; j++)
	{
		S[0][j] = j * gap_penalty;
		visited++;
	}

	// Main part
	for (unsigned int i = 1; i < rows; i++)
	{
		for (unsigned int j = 1; j < cols; j++)
		{
			float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});

			visited++;
		}
	}
	return visited;
}
