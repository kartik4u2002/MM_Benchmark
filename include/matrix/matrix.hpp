#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace matrix {

// Custom deleter for 64-byte aligned memory
struct AlignedDeleter {
    void operator()(float* ptr) const noexcept {
        if (!ptr) return;
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

/**
 * @brief Dense 2D row-major matrix class with 64-byte aligned contiguous memory.
 * 
 * Supports cache-line aligned allocations suitable for AVX2 and AVX-512 vectorization.
 * Provides safe bounds checking, factory initializers, copy/move semantics, and
 * numerical approximation checks.
 */
class Matrix {
public:
    static constexpr size_t ALIGNMENT = 64; // 64-byte alignment for cache line and AVX-512

    Matrix() : rows_(0), cols_(0), data_(nullptr) {}

    Matrix(size_t rows, size_t cols, float init_val = 0.0f)
        : rows_(rows), cols_(cols), data_(allocate(rows * cols)) {
        if (rows_ > 0 && cols_ > 0) {
            std::fill(data_.get(), data_.get() + (rows_ * cols_), init_val);
        }
    }

    // Copy constructor (deep copy)
    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), data_(allocate(other.rows_ * other.cols_)) {
        if (rows_ > 0 && cols_ > 0) {
            std::memcpy(data_.get(), other.data_.get(), rows_ * cols_ * sizeof(float));
        }
    }

    // Copy assignment
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Matrix tmp(other);
            swap(tmp);
        }
        return *this;
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    ~Matrix() = default;

    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(data_, other.data_);
    }

    // Dimensions and size
    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return rows_ * cols_; }
    size_t stride() const noexcept { return cols_; }
    bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    // Direct data access (64-byte aligned)
    float* data() noexcept { return data_.get(); }
    const float* data() const noexcept { return data_.get(); }

    // Fast unchecked element access (row-major)
    inline float& operator()(size_t r, size_t c) noexcept {
        return data_[r * cols_ + c];
    }

    inline const float& operator()(size_t r, size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    // Checked element access
    float& at(size_t r, size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of range for (" +
                                    std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return (*this)(r, c);
    }

    const float& at(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of range for (" +
                                    std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return (*this)(r, c);
    }

    // Zero-fill buffer
    void set_zero() {
        if (data_) {
            std::memset(data_.get(), 0, rows_ * cols_ * sizeof(float));
        }
    }

    // Factory methods
    static Matrix zeros(size_t rows, size_t cols) {
        return Matrix(rows, cols, 0.0f);
    }

    static Matrix ones(size_t rows, size_t cols) {
        return Matrix(rows, cols, 1.0f);
    }

    static Matrix identity(size_t n) {
        Matrix mat(n, n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            mat(i, i) = 1.0f;
        }
        return mat;
    }

    static Matrix random(size_t rows, size_t cols, float min_val = -1.0f, float max_val = 1.0f, uint32_t seed = 42) {
        Matrix mat(rows, cols);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < mat.size(); ++i) {
            mat.data()[i] = dist(gen);
        }
        return mat;
    }

    /**
     * @brief Numerical equivalence test with both relative and absolute tolerance.
     * 
     * Condition: |A(i, j) - B(i, j)| <= atol + rtol * max(|A(i, j)|, |B(i, j)|)
     */
    bool is_approx_equal(const Matrix& other, float rtol = 1e-4f, float atol = 1e-4f,
                         float* max_abs_diff_out = nullptr, float* max_rel_diff_out = nullptr) const;

    void print(std::ostream& os, size_t max_rows = 6, size_t max_cols = 6) const;

private:
    size_t rows_{0};
    size_t cols_{0};
    std::unique_ptr<float[], AlignedDeleter> data_{nullptr};

    static std::unique_ptr<float[], AlignedDeleter> allocate(size_t count) {
        if (count == 0) return nullptr;
        float* raw_ptr = nullptr;
#if defined(_MSC_VER)
        raw_ptr = static_cast<float*>(_aligned_malloc(count * sizeof(float), ALIGNMENT));
        if (!raw_ptr) throw std::bad_alloc();
#else
        void* ptr = nullptr;
        // posix_memalign requires alignment to be power of 2 and multiple of sizeof(void*)
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(float)) != 0) {
            throw std::bad_alloc();
        }
        raw_ptr = static_cast<float*>(ptr);
#endif
        return std::unique_ptr<float[], AlignedDeleter>(raw_ptr);
    }
};

} // namespace matrix
