#include "matrix/matrix.hpp"
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace matrix {

bool Matrix::is_approx_equal(const Matrix& other, float rtol, float atol,
                             float* max_abs_diff_out, float* max_rel_diff_out) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    bool match = true;

    for (size_t i = 0; i < size(); ++i) {
        float a = data_[i];
        float b = other.data_[i];
        float abs_diff = std::fabs(a - b);
        float denom = std::max(std::fabs(a), std::fabs(b));
        float rel_diff = (denom > 0.0f) ? (abs_diff / denom) : 0.0f;

        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        float allowed_diff = atol + rtol * denom;
        if (abs_diff > allowed_diff) {
            match = false;
        }
    }

    if (max_abs_diff_out) *max_abs_diff_out = max_abs_diff;
    if (max_rel_diff_out) *max_rel_diff_out = max_rel_diff;
    return match;
}

void Matrix::print(std::ostream& os, size_t max_rows, size_t max_cols) const {
    os << "Matrix (" << rows_ << "x" << cols_ << "):\n";
    size_t r_limit = std::min(rows_, max_rows);
    size_t c_limit = std::min(cols_, max_cols);

    for (size_t i = 0; i < r_limit; ++i) {
        os << "  [ ";
        for (size_t j = 0; j < c_limit; ++j) {
            os << std::setw(8) << std::fixed << std::setprecision(4) << (*this)(i, j) << " ";
        }
        if (c_limit < cols_) os << "... ";
        os << "]\n";
    }
    if (r_limit < rows_) {
        os << "  ...\n";
    }
}

} // namespace matrix
