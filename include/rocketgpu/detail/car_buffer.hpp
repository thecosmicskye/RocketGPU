#pragma once

#include "rocketgpu/detail/buffer.hpp"
#include "rocketgpu/detail/car_ref.hpp"

#if defined(ROCKETGPU_CAR_TILES)
namespace rocketgpu {
namespace detail {

void LaunchCarsToTiles(CarSpan dst, const CarState* src, size_t count, cudaStream_t stream);
void LaunchCarsFromTiles(CarState* dst, CarCSpan src, size_t count, cudaStream_t stream);
void LaunchCopyCars(CarSpan dst, CarCSpan src, size_t count, cudaStream_t stream);

class CarTileBuffer {
public:
    void resize(size_t n) {
        stage_.reset();
        bytes_.resize(CarStoreBytes(n));
        size_ = n;
    }
    size_t size() const noexcept { return size_; }
    CarSpan data() noexcept { return CarSpanFromBytes(bytes_.data()); }
    CarCSpan data() const noexcept { return CarSpanFromBytes(const_cast<unsigned char*>(bytes_.data())); }

    void copy_from_host(const CarState* host_src, size_t count, cudaStream_t stream = nullptr) {
        if (count > size_) {
            throw std::out_of_range("Buffer::copy_from_host count exceeds buffer size");
        }
        if (count == 0) return;
        stage().copy_from_host(host_src, count, stream);
        LaunchCarsToTiles(data(), stage_.data(), count, stream);
    }

    void copy_to_host(CarState* host_dst, size_t count, cudaStream_t stream = nullptr) const {
        if (count > size_) {
            throw std::out_of_range("Buffer::copy_to_host count exceeds buffer size");
        }
        if (count == 0) return;
        LaunchCarsFromTiles(stage().data(), data(), count, stream);
        stage_.copy_to_host(host_dst, count, stream);
    }

    void copy_from_buffer(const CarTileBuffer& source, size_t count, cudaStream_t stream = nullptr) {
        if (count > size_ || count > source.size_) {
            throw std::out_of_range("Buffer::copy_from_buffer count exceeds buffer size");
        }
        if (count == 0 || this == &source) return;
        LaunchCopyCars(data(), source.data(), count, stream);
    }

private:
    Buffer<CarState>& stage() const {
        if (stage_.size() != size_) stage_.resize(size_);
        return stage_;
    }

    Buffer<unsigned char> bytes_;
    mutable Buffer<CarState> stage_;
    size_t size_ = 0;
};

}
}
#endif
