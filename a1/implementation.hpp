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

	int tile_size = 129;
// Boundary
#pragma omp parallel
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

#pragma omp parallel
#pragma omp single
	{
		for (int D = 0; D < rows + cols - 1 - tile_size; D += tile_size)
		{
			int steps = (int)D / tile_size;
#pragma omp taskloop reduction(+ : visited) grainsize(grain_size)
			for (int p = std::max((steps - tile_size) * tile_size, 0); p <= tile_size * steps; p += tile_size)
			{
				if (p < rows - 1)
				{
					int q = std::abs(p - tile_size * steps);
					if (q < rows - 1)
					{

						for (int d = 0; d < 2 * tile_size - 1; d++)
						{
#pragma omp taskloop reduction(+ : visited) grainsize(grain_size)
							for (int i = std::max(d - tile_size, 0); i <= d; i++)
							{
								int j = d - i;

								if (i < tile_size && j < tile_size && i >= 0 && j >= 0)
								{
									int m = i + p + 1;
									int n = j + q + 1;

									float match = S[n - 1][m - 1] + SUB[cmap.at(X[n - 1])][cmap.at(Y[m - 1])];
									float del = S[n - 1][m] + gap_penalty;
									float insert = S[n][m - 1] + gap_penalty;
									S[n][m] = std::max({match, del, insert});

									visited++;
								}
							}
						}
					}
				}
			}
		}
	}

	return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float **S, float **SUB, std::unordered_map<char, int> cmap, int tile_size = 1)
{
	// Your work goes here, paralellize the code below using OpenMP tasks
	unsigned long visited = 0;
	gap_penalty = SUB[0][cmap['*']]; // min score

// Boundary
#pragma omp parallel
#pragma omp single
	{
#pragma omp taskgroup task_reduction(+ : visited)
		for (unsigned int i = 1; i < rows; i++)
		{
#pragma omp task in_reduction(+ : visited)
			{
				S[i][0] = i * gap_penalty;
				visited++;
			}
		}

#pragma omp taskgroup task_reduction(+ : visited)
		for (unsigned int j = 0; j < cols; j++)
		{
#pragma omp task in_reduction(+ : visited)
			{
				S[0][j] = j * gap_penalty;
				visited++;
			}
		}
	}

#pragma omp parallel
#pragma omp single
	{
		for (int D = 0; D < rows + cols - 1 - tile_size; D += tile_size)
		{

			int steps = (int)D / tile_size;
#pragma omp taskgroup task_reduction(+ : visited)
			for (int p = std::max((steps - tile_size) * tile_size, 0); p <= tile_size * steps; p += tile_size)
			{
				if (p < rows - 1)
				{
					int q = std::abs(p - tile_size * steps);
					if (q < rows - 1)
					{

#pragma omp taskgroup task_reduction(+ : visited)
						for (int d = 0; d < 2 * tile_size - 1; d++)
						{
							for (int i = std::max(d - tile_size, 0); i <= d; i++)
							{
								int j = d - i;

								if (i < tile_size && j < tile_size && i >= 0 && j >= 0)
								{
									int m = i + p + 1;
									int n = j + q + 1;

#pragma omp task in_reduction(+ : visited)
									{
										float match = S[n - 1][m - 1] + SUB[cmap.at(X[n - 1])][cmap.at(Y[m - 1])];
										float del = S[n - 1][m] + gap_penalty;
										float insert = S[n][m - 1] + gap_penalty;
										S[n][m] = std::max({match, del, insert});

										visited++;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return visited;
}
