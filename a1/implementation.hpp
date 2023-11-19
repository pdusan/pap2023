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
	// 	#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) firstprivate(SUB, cmap, X, Y) shared(S, gap_penalty)
	// 	for (unsigned int i = 1; i < rows; i++)
	// 	{
	// 		for (unsigned int j = 1; j < cols; j++)
	// 		{
	// 			float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
	// 			float del = S[i - 1][j] + gap_penalty;
	// 			float insert = S[i][j - 1] + gap_penalty;
	// 			S[i][j] = std::max({match, del, insert});
		
	// 			visited++;
	// 		}
	// 	}

	 #pragma omp parallel
        {
            #pragma omp single
            {
                for (int d = 1; d < rows + cols - 1; d++) {
                    int start_row = std::max(1, d - (int)cols + 1);
                    int end_row = std::min(d, (int)rows - 1);

                    #pragma omp taskloop firstprivate(start_row, end_row, SUB, cmap, X, Y) reduction(+:visited) shared(S, gap_penalty)
                    for (int i = start_row; i <= end_row; i++) {
                        int j = d - i;
                        float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
                        float del = S[i - 1][j] + gap_penalty;
                        float insert = S[i][j - 1] + gap_penalty;

                        S[i][j] = std::max({match, del, insert});

                        visited++;
                    }
                }
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
