#include <vector>

#include "quickmp.hpp"

int main()
{
    size_t n = 7200, m = 10;

    std::vector<double> T(n), P(n - m + 1);

    quickmp::initialize();

    for (int i = 0; i < 50; i++) {
        quickmp::selfjoin(T.data(), P.data(), n, m);
    }

    quickmp::finalize();
}
