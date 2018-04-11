// Copyright 2016 The Sonnet Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/render_sprites/kernels/render_sprites_ops.h"

#include <stdio.h>
#include <cmath>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

#define GET_SPRITE_POINT(x, y)                   \
  sprites[batch_id * sprites_batch_stride +         \
          sprite_id * sprites_sprite_stride +       \
          y * sprites_row_stride +                  \
          x * (n_channels + 1) + chan]

#define GET_ALPHA_POINT(x, y)                    \
  sprites[batch_id * sprites_batch_stride +         \
          sprite_id * sprites_sprite_stride +       \
          y * sprites_row_stride +                  \
          x * (n_channels + 1) + n_channels]


template <typename T>
__global__ void RenderSprites2DKernel(const T* __restrict__ sprites,
                                      const int* __restrict__ n_sprites,
                                      const T* __restrict__ scales,
                                      const T* __restrict__ offsets,
                                      const T* __restrict__ backgrounds,

                                      T* __restrict__ output,

                                      const int batch_size,

                                      const int max_sprites,
                                      const int sprite_height,
                                      const int sprite_width,

                                      const int img_height,
                                      const int img_width,

                                      const int n_channels){

  const int output_size = batch_size * img_height * img_width * n_channels;

  CUDA_1D_KERNEL_LOOP(index, output_size) {
    const int sprites_batch_stride = max_sprites * sprite_height * sprite_width * (n_channels + 1);
    const int sprites_sprite_stride = sprite_height * sprite_width * (n_channels + 1);
    const int sprites_row_stride = sprite_width * (n_channels + 1);

    const int scales_batch_stride = 2 * max_sprites;
    const int offsets_batch_stride = 2 * max_sprites;

    const int img_batch_stride = img_height * img_width * n_channels;
    const int img_row_stride = img_width * n_channels;

    const T sprite_height_T = static_cast<T>(sprite_height);
    const T sprite_width_T = static_cast<T>(sprite_width);

    const T img_height_T = static_cast<T>(img_height);
    const T img_width_T = static_cast<T>(img_width);

    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);

    const int batch_id = index / img_batch_stride;
    const int index_in_batch = index % img_batch_stride;
    const int img_y = index_in_batch / img_row_stride;
    const int index_in_row = index_in_batch % img_row_stride;
    const int img_x = index_in_row / n_channels;
    const int chan = index_in_row % n_channels;

    const T img_y_T = static_cast<T>(img_y);
    const T img_x_T = static_cast<T>(img_x);

    T pixel_value = backgrounds[index];

    for (int sprite_id = 0; sprite_id < n_sprites[batch_id]; ++sprite_id) {
      const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
      const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];

      const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
      const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];

      // The pixel location represented in the sprites's co-ordinate frame
      const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
      const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

      const bool within_bounds = x > static_cast<T>(-1.0) &&
                                 y > static_cast<T>(-1.0) &&
                                 x < sprite_width_T &&
                                 y < sprite_height_T;
      if (within_bounds){
        const int fx = std::floor(static_cast<float>(x));
        const int fy = std::floor(static_cast<float>(y));

        const int cx = fx + 1;
        const int cy = fy + 1;

        const T dx = static_cast<T>(cx) - x;
        const T dy = static_cast<T>(cy) - y;

        const T alpha_fxfy = (fx >= 0 && fy >= 0)
                           ? GET_ALPHA_POINT(fx, fy)
                           : zero;
        const T alpha_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
                           ? GET_ALPHA_POINT(cx, cy)
                           : zero;
        const T alpha_fxcy = (fx >= 0 && cy <= sprite_height - 1)
                           ? GET_ALPHA_POINT(fx, cy)
                           : zero;
        const T alpha_cxfy = (cx <= sprite_width - 1 && fy >= 0)
                           ? GET_ALPHA_POINT(cx, fy)
                           : zero;
        const T alpha = dx * dy * alpha_fxfy +
                        (one - dx) * (one - dy) * alpha_cxcy +
                        dx * (one - dy) * alpha_fxcy +
                        (one - dx) * dy * alpha_cxfy;

        const T img_fxfy = (fx >= 0 && fy >= 0)
                           ? GET_SPRITE_POINT(fx, fy)
                           : zero;
        const T img_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
                           ? GET_SPRITE_POINT(cx, cy)
                           : zero;
        const T img_fxcy = (fx >= 0 && cy <= sprite_height - 1)
                           ? GET_SPRITE_POINT(fx, cy)
                           : zero;
        const T img_cxfy = (cx <= sprite_width - 1 && fy >= 0)
                           ? GET_SPRITE_POINT(cx, fy)
                           : zero;
        const T interp = dx * dy * img_fxfy +
                         (one - dx) * (one - dy) * img_cxcy +
                         dx * (one - dy) * img_fxcy +
                         (one - dx) * dy * img_cxfy;

        pixel_value = alpha * interp + (1 - alpha) * pixel_value;
      }
    } // sprite_id
    output[index] = pixel_value;
  }
}

}  // namespace

namespace functor {

template <typename T>
struct RenderSprites2DFunctor<GPUDevice, T>{
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const GPUDevice& d,

                   const T* __restrict__ sprites,
                   const int* __restrict__ n_sprites,
                   const T* __restrict__ scales,
                   const T* __restrict__ offsets,
                   const T* __restrict__ backgrounds,

                   T* __restrict__ output,

                   const int batch_size,

                   const int max_sprites,
                   const int sprite_height,
                   const int sprite_width,

                   const int img_height,
                   const int img_width,

                   const int n_channels){

    const int output_size = batch_size * img_height * img_width * n_channels;

    ::tensorflow::CudaLaunchConfig config = ::tensorflow::GetCudaLaunchConfig(output_size, d);

    RenderSprites2DKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            sprites, n_sprites, scales, offsets, backgrounds, output,
            batch_size, max_sprites, sprite_height, sprite_width,
            img_height, img_width, n_channels);
  }
};

// TODO(fviola): gcudacc fails at compile time with Eigen::half.
// template struct RenderSprites2DFunctor<GPUDevice, Eigen::half>;
template struct RenderSprites2DFunctor<GPUDevice, float>;
// template struct RenderSprites2DFunctor<GPUDevice, double>;

}  // namespace functor




namespace {

#define UPDATE_GRAD_SPRITES(x, y, v) \
  atomicAdd(grad_sprites + \
            batch_id * sprites_batch_stride + \
            sprite_id * sprites_sprite_stride + \
            y * sprites_row_stride + \
            x * (n_channels + 1) + \
            chan, \
            v)

#define UPDATE_GRAD_ALPHAS(x, y, v) \
  atomicAdd(grad_sprites + \
            batch_id * sprites_batch_stride + \
            sprite_id * sprites_sprite_stride + \
            y * sprites_row_stride + \
            x * (n_channels + 1) + \
            n_channels, \
            v)

#define UPDATE_GRAD_SCALES_Y(v) \
  atomicAdd(grad_scales + \
            batch_id * scales_batch_stride + \
            sprite_id * 2, \
            v)

#define UPDATE_GRAD_SCALES_X(v) \
  atomicAdd(grad_scales + \
            batch_id * scales_batch_stride + \
            sprite_id * 2 + 1, \
            v)

#define UPDATE_GRAD_OFFSETS_Y(v) \
  atomicAdd(grad_offsets + \
            batch_id * offsets_batch_stride + \
            sprite_id * 2, \
            v)

#define UPDATE_GRAD_OFFSETS_X(v) \
  atomicAdd(grad_offsets + \
            batch_id * offsets_batch_stride + \
            sprite_id * 2 + 1, \
            v)

template <typename T>
__global__ void RenderSpritesGrad2DKernel(const T* __restrict__ sprites,
                                          const int* __restrict__ n_sprites,
                                          const T* __restrict__ scales,
                                          const T* __restrict__ offsets,
                                          const T* __restrict__ backgrounds,
                                          const T* __restrict__ grad_output,

                                          T* __restrict__ grad_sprites,
                                          T* __restrict__ grad_n_sprites,
                                          T* __restrict__ grad_scales,
                                          T* __restrict__ grad_offsets,
                                          T* __restrict__ grad_backgrounds,

                                          const int batch_size,

                                          const int max_sprites,
                                          const int sprite_height,
                                          const int sprite_width,

                                          const int img_height,
                                          const int img_width,

                                          const int n_channels){

  extern __shared__ T shared_scratch[];
  T* scratch = &shared_scratch[(max_sprites+1) * threadIdx.x];

  const int output_size = batch_size * img_height * img_width * n_channels;

  CUDA_1D_KERNEL_LOOP(index, output_size) {
    const int sprites_batch_stride = max_sprites * sprite_height * sprite_width * (n_channels + 1);
    const int sprites_sprite_stride = sprite_height * sprite_width * (n_channels + 1);
    const int sprites_row_stride = sprite_width * (n_channels + 1);

    const int scales_batch_stride = 2 * max_sprites;
    const int offsets_batch_stride = 2 * max_sprites;

    const int img_batch_stride = img_height * img_width * n_channels;
    const int img_row_stride = img_width * n_channels;

    const T sprite_height_T = static_cast<T>(sprite_height);
    const T sprite_width_T = static_cast<T>(sprite_width);

    const T img_height_T = static_cast<T>(img_height);
    const T img_width_T = static_cast<T>(img_width);

    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);

    const int batch_id = index / img_batch_stride;
    const int index_in_batch = index % img_batch_stride;
    const int img_y = index_in_batch / img_row_stride;
    const int index_in_row = index_in_batch % img_row_stride;
    const int img_x = index_in_row / n_channels;
    const int chan = index_in_row % n_channels;

    const T img_y_T = static_cast<T>(img_y);
    const T img_x_T = static_cast<T>(img_x);

    scratch[0] = backgrounds[batch_id * img_batch_stride +
                             img_y * img_row_stride +
                             img_x * n_channels + chan];

    for (int sprite_id = 0; sprite_id < n_sprites[batch_id]; ++sprite_id) {
      const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
      const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];

      const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
      const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];

      // The pixel location represented in the sprites's co-ordinate frame
      const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
      const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

      const bool within_bounds = x > static_cast<T>(-1.0) &&
                                 y > static_cast<T>(-1.0) &&
                                 x < sprite_width_T &&
                                 y < sprite_height_T;
      if (within_bounds){
        const int fx = std::floor(static_cast<float>(x));
        const int fy = std::floor(static_cast<float>(y));

        const int cx = fx + 1;
        const int cy = fy + 1;

        const T dx = static_cast<T>(cx) - x;
        const T dy = static_cast<T>(cy) - y;

        const T alpha_fxfy = (fx >= 0 && fy >= 0)
                             ? GET_ALPHA_POINT(fx, fy)
                             : zero;
        const T alpha_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
                             ? GET_ALPHA_POINT(cx, cy)
                             : zero;
        const T alpha_fxcy = (fx >= 0 && cy <= sprite_height - 1)
                             ? GET_ALPHA_POINT(fx, cy)
                             : zero;
        const T alpha_cxfy = (cx <= sprite_width - 1 && fy >= 0)
                             ? GET_ALPHA_POINT(cx, fy)
                             : zero;
        const T alpha = dx * dy * alpha_fxfy +
                        (one - dx) * (one - dy) * alpha_cxcy +
                        dx * (one - dy) * alpha_fxcy +
                        (one - dx) * dy * alpha_cxfy;

        const T img_fxfy = (fx >= 0 && fy >= 0)
                           ? GET_SPRITE_POINT(fx, fy)
                           : zero;
        const T img_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
                           ? GET_SPRITE_POINT(cx, cy)
                           : zero;
        const T img_fxcy = (fx >= 0 && cy <= sprite_height - 1)
                           ? GET_SPRITE_POINT(fx, cy)
                           : zero;
        const T img_cxfy = (cx <= sprite_width - 1 && fy >= 0)
                           ? GET_SPRITE_POINT(cx, fy)
                           : zero;
        const T interp = dx * dy * img_fxfy +
                         (one - dx) * (one - dy) * img_cxcy +
                         dx * (one - dy) * img_fxcy +
                         (one - dx) * dy * img_cxfy;

        scratch[sprite_id + 1] = alpha * interp + (1 - alpha) * scratch[sprite_id];
      }
    } // sprite_id - forward pass

    T premult = grad_output[batch_id * img_batch_stride +
                            img_y * img_row_stride +
                            img_x * n_channels + chan];

    for (int sprite_id = n_sprites[batch_id]-1; sprite_id >= 0 ; --sprite_id) {
      const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
      const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];

      const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
      const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];

      // The pixel location represented in the sprites's co-ordinate frame
      const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
      const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

      const bool within_bounds = x > static_cast<T>(-1.0) &&
                                 y > static_cast<T>(-1.0) &&
                                 x < sprite_width_T &&
                                 y < sprite_height_T;
      if (within_bounds){
        const int fx = std::floor(static_cast<float>(x));
        const int fy = std::floor(static_cast<float>(y));

        const int cx = fx + 1;
        const int cy = fy + 1;

        const T dx = static_cast<T>(cx) - x;
        const T dy = static_cast<T>(cy) - y;

        const T alpha_fxfy = (fx >= 0 && fy >= 0)
                           ? GET_ALPHA_POINT(fx, fy)
                           : zero;
        const T alpha_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
                           ? GET_ALPHA_POINT(cx, cy)
                           : zero;
        const T alpha_fxcy = (fx >= 0 && cy <= sprite_height - 1)
                           ? GET_ALPHA_POINT(fx, cy)
                           : zero;
        const T alpha_cxfy = (cx <= sprite_width - 1 && fy >= 0)
                           ? GET_ALPHA_POINT(cx, fy)
                           : zero;
        const T alpha = dx * dy * alpha_fxfy +
                        (one - dx) * (one - dy) * alpha_cxcy +
                        dx * (one - dy) * alpha_fxcy +
                        (one - dx) * dy * alpha_cxfy;

        const T img_fxfy = (fx >= 0 && fy >= 0)
                           ? GET_SPRITE_POINT(fx, fy)
                           : zero;
        const T img_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
                           ? GET_SPRITE_POINT(cx, cy)
                           : zero;
        const T img_fxcy = (fx >= 0 && cy <= sprite_height - 1)
                           ? GET_SPRITE_POINT(fx, cy)
                           : zero;
        const T img_cxfy = (cx <= sprite_width - 1 && fy >= 0)
                           ? GET_SPRITE_POINT(cx, fy)
                           : zero;
        const T interp = dx * dy * img_fxfy +
                         (one - dx) * (one - dy) * img_cxcy +
                         dx * (one - dy) * img_fxcy +
                         (one - dx) * dy * img_cxfy;

        const T grad_y_wrt_scale_y = -sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / (scale_y * scale_y);
        const T grad_x_wrt_scale_x = -sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / (scale_x * scale_x);

        const T grad_y_wrt_offset_y = -sprite_height_T / scale_y;
        const T grad_x_wrt_offset_x = -sprite_width_T / scale_x;

        // ------ update gradient through alpha ------

        const T alpha_y_factor = dx * (alpha_fxcy - alpha_fxfy) + (1 - dx) * (alpha_cxcy - alpha_cxfy);
        const T alpha_x_factor = dy * (alpha_cxfy - alpha_fxfy) + (1 - dy) * (alpha_cxcy - alpha_fxcy);

        const T bg = scratch[sprite_id * n_channels + chan];

        const T alpha_premult = premult * (interp - bg);

        UPDATE_GRAD_SCALES_Y(alpha_premult * alpha_y_factor * grad_y_wrt_scale_y);
        UPDATE_GRAD_SCALES_X(alpha_premult * alpha_x_factor * grad_x_wrt_scale_x);
        UPDATE_GRAD_OFFSETS_Y(alpha_premult * alpha_y_factor * grad_y_wrt_offset_y);
        UPDATE_GRAD_OFFSETS_X(alpha_premult * alpha_x_factor * grad_x_wrt_offset_x);

        if (fx >= 0 && fy >= 0) {
          UPDATE_GRAD_ALPHAS(fx, fy, alpha_premult * dx * dy);
        }
        if (cx <= sprite_width - 1 && cy <= sprite_height - 1) {
          UPDATE_GRAD_ALPHAS(cx, cy, alpha_premult * (1-dx) * (1-dy));
        }
        if (fx >= 0 && cy <= sprite_height - 1) {
          UPDATE_GRAD_ALPHAS(fx, cy, alpha_premult * dx * (1-dy));
        }
        if (cx <= sprite_width - 1 && fy >= 0) {
          UPDATE_GRAD_ALPHAS(cx, fy, alpha_premult * (1-dx) * dy);
        }

        // ------ update gradient through sprites ------

        const T sprite_premult = premult * alpha;

        const T sprite_y_factor = dx * (img_fxcy - img_fxfy) + (1 - dx) * (img_cxcy - img_cxfy);
        const T sprite_x_factor = dy * (img_cxfy - img_fxfy) + (1 - dy) * (img_cxcy - img_fxcy);

        UPDATE_GRAD_SCALES_Y(sprite_premult * sprite_y_factor * grad_y_wrt_scale_y);
        UPDATE_GRAD_SCALES_X(sprite_premult * sprite_x_factor * grad_x_wrt_scale_x);
        UPDATE_GRAD_OFFSETS_Y(sprite_premult * sprite_y_factor * grad_y_wrt_offset_y);
        UPDATE_GRAD_OFFSETS_X(sprite_premult * sprite_x_factor * grad_x_wrt_offset_x);

        if (fx >= 0 && fy >= 0) {
          UPDATE_GRAD_SPRITES(fx, fy, sprite_premult * dx * dy);
        }
        if (cx <= sprite_width - 1 && cy <= sprite_height - 1) {
          UPDATE_GRAD_SPRITES(cx, cy, sprite_premult * (1-dx) * (1-dy));
        }
        if (fx >= 0 && cy <= sprite_height - 1) {
          UPDATE_GRAD_SPRITES(fx, cy, sprite_premult * dx * (1-dy));
        }
        if (cx <= sprite_width - 1 && fy >= 0) {
          UPDATE_GRAD_SPRITES(cx, fy, sprite_premult * (1-dx) * dy);
        }

        premult *= (1 - alpha);
      }
    } // sprite_id - backward pass

    grad_backgrounds[batch_id * img_batch_stride +
                     img_y * img_row_stride +
                     img_x * n_channels + chan] = premult;
  }
}

#undef GET_ALPHA_POINT
#undef GET_SPRITE_POINT
#undef UPDATE_GRAD_SPRITES
#undef UPDATE_GRAD_ALPHAS
#undef UPDATE_GRAD_SCALES_Y
#undef UPDATE_GRAD_SCALES_X
#undef UPDATE_GRAD_OFFSETS_Y
#undef UPDATE_GRAD_OFFSETS_X

}  // namespace

namespace functor {

template <typename T>
struct RenderSpritesGrad2DFunctor<GPUDevice, T>{

  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const GPUDevice& d,

                   const T* __restrict__ sprites,
                   const int* __restrict__ n_sprites,
                   const T* __restrict__ scales,
                   const T* __restrict__ offsets,
                   const T* __restrict__ backgrounds,
                   const T* __restrict__ grad_output,

                   T* __restrict__ scratch,

                   T* __restrict__ grad_sprites,
                   T* __restrict__ grad_n_sprites,
                   T* __restrict__ grad_scales,
                   T* __restrict__ grad_offsets,
                   T* __restrict__ grad_backgrounds,

                   const int batch_size,

                   const int max_sprites,
                   const int sprite_height,
                   const int sprite_width,

                   const int img_height,
                   const int img_width,

                   const int n_channels){

    // Set gradients to 0, because the kernel incrementally updates the
    // tensor entries by adding partial contributions (except for grad_backgrounds, which is only set one).
    const int grad_sprites_size = batch_size * max_sprites * sprite_height * sprite_width * (n_channels + 1);
    const int grad_n_sprites_size = batch_size;
    const int grad_scales_size = batch_size * max_sprites * 2;
    const int grad_offsets_size = batch_size * max_sprites * 2;
    const int grad_backgrounds_size = batch_size * img_height * img_width * n_channels;

    ::tensorflow::CudaLaunchConfig config = ::tensorflow::GetCudaLaunchConfig(grad_sprites_size, d);
    ::tensorflow::SetZero
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_sprites_size, grad_sprites);

    config = ::tensorflow::GetCudaLaunchConfig(grad_n_sprites_size, d);
    ::tensorflow::SetZero
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_n_sprites_size, grad_n_sprites);

    config = ::tensorflow::GetCudaLaunchConfig(grad_scales_size, d);
    ::tensorflow::SetZero
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_scales_size, grad_scales);

    config = ::tensorflow::GetCudaLaunchConfig(grad_offsets_size, d);
    ::tensorflow::SetZero
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_offsets_size, grad_offsets);

    const int shared_mem_bytes_per_block = 49152;  // 48 KB * 1024.
    config = ::tensorflow::GetCudaLaunchConfig(grad_backgrounds_size, d, RenderSpritesGrad2DKernel<T>, shared_mem_bytes_per_block, 0);

    RenderSpritesGrad2DKernel<T>
        <<<config.block_count, config.thread_per_block, shared_mem_bytes_per_block, d.stream()>>>(
            sprites, n_sprites, scales, offsets, backgrounds, grad_output,
            grad_sprites, grad_n_sprites, grad_scales, grad_offsets, grad_backgrounds,
            batch_size, max_sprites, sprite_height, sprite_width,
            img_height, img_width, n_channels);
  }
};

template struct RenderSpritesGrad2DFunctor<GPUDevice, float>;

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
