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
			#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) shared(S, rows) firstprivate(gap_penalty)
			for (unsigned int i = 1; i < rows; i++)
			{
				S[i][0] = i * gap_penalty;
				visited++;
			}

			#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) shared(S, cols) firstprivate(gap_penalty)
			for (unsigned int j = 0; j < cols; j++)
			{
				S[0][j] = j * gap_penalty;
				visited++;
			}
		}
	}
	
	#pragma omp parallel
	{
		#pragma omp single
		for (int d = 1; d < rows+cols-1; d+=grain_size)
		{
			#pragma omp taskloop grainsize(grain_size) reduction(+ : visited) shared(S, cmap) firstprivate(gap_penalty,SUB, d)
			for (unsigned int i = std::max(1, d-cols+1); i < std::min(d,rows-1)+1; i++)
			{
				unsigned int j = d - i;
				if (j>0) {
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
	for (int d = 1; d < rows + cols - 1; d++)
	{
		for (unsigned int i = std::max(1, d - cols + 1); i < std::min(d, rows - 1) + 1; i++)
		{
			unsigned int j = d - i;
			if (j>0) {
				float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j - 1])];
				float del = S[i - 1][j] + gap_penalty;
				float insert = S[i][j - 1] + gap_penalty;
				S[i][j] = std::max({match, del, insert});
		
				visited++;
			}
		}
	}

	return visited;
}
