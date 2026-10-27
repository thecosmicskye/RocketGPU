#pragma once

#include <algorithm>
#include <new>
#include <stdexcept>
#include <vector>

#include "rocketgpu/detail/common.hpp"

namespace rocketgpu {
namespace detail {
template <typename T>
class Buffer {
public:
    Buffer() = default;
    ~Buffer() { reset(); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept {
        move_from(std::move(other));
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    void resize(size_t n) {
        reset();
        size_ = n;
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc buffer");

        CheckCuda(cudaMemset(ptr_, 0, n * sizeof(T)), "cudaMemset buffer");
#else
        host_.resize(n);
        ptr_ = host_.data();
#endif
    }

    void reset() {
#if defined(ROCKETGPU_USE_CUDA)
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
#else
        host_.clear();
        host_.shrink_to_fit();
#endif
        ptr_ = nullptr;
        size_ = 0;
    }

    size_t size() const noexcept { return size_; }
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    void memset_zero(cudaStream_t stream = nullptr) {
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream), "cudaMemsetAsync buffer");
#else
        std::fill(host_.begin(), host_.end(), T{});
        (void)stream;
#endif
    }

    void copy_from_host(const T* host_src, size_t count, cudaStream_t stream = nullptr) {
        if (count > size_) {
            throw std::out_of_range("Buffer::copy_from_host count exceeds buffer size");
        }
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaMemcpyAsync(ptr_, host_src, count * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
#else
        std::copy(host_src, host_src + count, host_.begin());
        (void)stream;
#endif
    }

    void copy_range_to_host(T* host_dst, size_t first, size_t count, cudaStream_t stream = nullptr) const {
        if (first > size_ || count > size_ - first) {
            throw std::out_of_range("Buffer::copy_range_to_host range exceeds buffer size");
        }
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaMemcpyAsync(host_dst, ptr_ + first, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H range");
#else
        std::copy(host_.begin() + static_cast<std::ptrdiff_t>(first),
                  host_.begin() + static_cast<std::ptrdiff_t>(first + count), host_dst);
        (void)stream;
#endif
    }

    void copy_to_host(T* host_dst, size_t count, cudaStream_t stream = nullptr) const {
        if (count > size_) {
            throw std::out_of_range("Buffer::copy_to_host count exceeds buffer size");
        }
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaMemcpyAsync(host_dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
#else
        std::copy(host_.begin(), host_.begin() + static_cast<std::ptrdiff_t>(count), host_dst);
        (void)stream;
#endif
    }

    void copy_from_buffer(const Buffer& source, size_t count, cudaStream_t stream = nullptr) {
        if (count > size_ || count > source.size_) {
            throw std::out_of_range("Buffer::copy_from_buffer count exceeds buffer size");
        }
        if (count == 0 || this == &source) return;
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaMemcpyAsync(ptr_, source.ptr_, count * sizeof(T), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync D2D buffer");
#else
        std::copy_n(source.host_.begin(), count, host_.begin());
        (void)stream;
#endif
    }

private:
    void move_from(Buffer&& other) noexcept {
        ptr_ = other.ptr_;
        size_ = other.size_;
#if !defined(ROCKETGPU_USE_CUDA)
        host_ = std::move(other.host_);
        ptr_ = host_.data();
#endif
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    T* ptr_ = nullptr;
    size_t size_ = 0;
#if !defined(ROCKETGPU_USE_CUDA)
    std::vector<T> host_;
#endif
};
}
}
