#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>

int main() {
	auto f = [](double x) {return cos(x); };
	constexpr double accuracy = 0.000001;
	double a, b;
	a = -M_PI / 2;
	b = M_PI / 2;
	for (int num_threads : {1, 2, 3, 4}) {
		auto start = omp_get_wtime();
		double dx = accuracy;
		double S = 0;
		std::vector<double> s(num_threads, 0);
#pragma omp parallel num_threads(num_threads)
		{
			size_t thread_num = omp_get_thread_num();
			double x_range = (b - a) / (accuracy * num_threads);
			for (size_t i = x_range * thread_num; i < x_range * (thread_num + 1); ++i)
				s[thread_num] += (f(a + dx * i) + f(a + dx * (i + 1))) / 2 * dx;
		}
		for (auto e : s)
			S += e;
		auto end = omp_get_wtime();
		std::cout << "Number of threads: " << num_threads;
		std::cout << "\nTime:\n";
		std::cout << end - start << '\n';
		std::cout << S << std::endl;
	}
	system("pause");
	return 0;
}