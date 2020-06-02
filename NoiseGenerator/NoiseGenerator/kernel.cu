/*
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//#include "C:\Users\eduar\Documents\lib\glm\glm\glm.hpp"
//#include "C:\Users\eduar\Documents\lib\glm\glm\gtc\noise.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
//#include "C:/Users/eduar/Documents/nubes/NoiseGenerator/Dependencies/Simple OpenGL Image Library/src/SOIL.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



using namespace std;


__device__ float hashFract(float n) {
	float m = sin(n * 43758.5453f);
	return m - floor(m);
}

__device__ float lerp(float start, float end, float interpolator) {
	return start * (1.0 - interpolator) + end * interpolator;
}

__device__ void fracArr3(float* in, float* out) {
	out[0] = in[0] - floor(in[0]);
	out[1] = in[1] - floor(in[1]);
	out[2] = in[2] - floor(in[2]);
}

__device__ void floorArr3(float* in, float* out) {
	out[0] = floor(in[0]);
	out[1] = floor(in[1]);
	out[2] = floor(in[2]);
}

__device__ void mulArr3(float* a, float* b, float* result) {
	result[0] = a[0] * b[0];
	result[1] = a[1] * b[1];
	result[2] = a[2] * b[2];
}

__device__ void mulArr3f(float* a, float t, float* result) {
	result[0] = a[0] * t;
	result[1] = a[1] * t;
	result[2] = a[2] * t;
}

__device__ void subArr3(float* a, float* b, float* result) {
	result[0] = a[0] - b[0];
	result[1] = a[1] - b[1];
	result[2] = a[2] - b[2];
}

__device__ void sumArr3(float* a, float* b, float* result) {
	result[0] = a[0] + b[0];
	result[1] = a[1] + b[1];
	result[2] = a[2] + b[2];
}

__device__ void sumArr3f(float* a, float v, float* result) {
	result[0] = a[0] + v;
	result[1] = a[1] + v;
	result[2] = a[2] + v;
}

__device__ void copyArr3(const float* in, float* out) {
	out[0] = in[0];
	out[1] = in[1];
	out[2] = in[2];
}

__device__ void fillArr3(const float value, float* out) {
	out[0] = value;
	out[1] = value;
	out[2] = value;
}

__device__ float magnitudeArr3(float* a) {
	return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

__device__ float noise(float* position) {
	float rounded[3];
	float fraction[3];
	float two[3] = { 2.0, 2.0, 2.0 };
	float three[3] = { 3.0, 3.0, 3.0 };

	floorArr3(position, rounded);
	fracArr3(position, fraction);

	mulArr3(two, rounded, rounded);
	subArr3(three, rounded, rounded);
	mulArr3(rounded, rounded, rounded);
	
	mulArr3(rounded, rounded, rounded);

	float n = rounded[0] + rounded[1] * 57.0f + rounded[2] * 113.0f;

	return lerp(
		lerp(
			lerp(hashFract(n + 0.0f), hashFract(n + 1.0f), fraction[0]),
			lerp(hashFract(n + 57.0f), hashFract(n + 58.0f), fraction[0]),
			fraction[1]
		),
		lerp(
			lerp(hashFract(n + 113.0f), hashFract(n + 114.0f), fraction[0]),
			lerp(hashFract(n + 170.0f), hashFract(n + 171.0f), fraction[0]),
			fraction[1]
		),
		fraction[2]
	);
}

__device__ float worley(float* position, int cells) {

	float pointPosition[3];
	float currentCell[3];
	float checkingCell[3];
	float pointNoise;
	float distanceToNoise[3];
	float gridSpaceNoise[3];
	float minDistance = 1.0e10;
	// get point position in grid space
	mulArr3f(position, cells, pointPosition);
	// get cell of point
	floorArr3(pointPosition, currentCell);
	// check in 3x3x3 space around cell (adjacent ones)
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			for (int z = -1; z <= 1; z++) {
				// get cell in which its point's distance
				// will be queried
				checkingCell[0] = currentCell[0] + x;
				checkingCell[1] = currentCell[1] + y;
				checkingCell[2] = currentCell[2] + z;
				// get grid space noise position by summing the checking cell position 
				// to the cell space noise vector
				// then get distance between point and noise
				// point - (checkingCell + noise) = point - checkingCell - noise
				sumArr3f(checkingCell, noise(checkingCell), gridSpaceNoise);
				subArr3(pointPosition, gridSpaceNoise, distanceToNoise);
				minDistance = fminf(minDistance, magnitudeArr3(distanceToNoise));
			}
		}
	}
	minDistance = fmaxf(minDistance, 0.0);
	minDistance = fminf(minDistance, 1.0);
	return minDistance;
}

__global__ void pixelNoise(unsigned char * image,  const int width, const int height, const int depth, const int channels, int cells) {
	// y * s_x * s_z + z * s_x + x
	//int i = 3* (blockIdx.y * gridDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x);
	//float position[3] = { threadIdx.x / (float) blockDim.x, blockIdx.y / (float) blockDim.y, blockIdx.x / (float) blockDim.y};
	int i =  blockIdx.x * blockDim.x + threadIdx.x;
	if (i > width * height * depth)
		return;
	//float position[3] = { threadIdx.x / (float) blockDim.x, blockIdx.x / (float) gridDim.x};
	// Get row and col of the pixel
	// Integer division here
	int row = i / (width * depth);
	int col = i % (width * depth);
	// Get Position = {x, y, z} and normalize it
	float position[3] = {
		col % width / (float) width,
		row / (float) height,
		col / width / (float) depth
	};
	i *= channels;
	// worley fractal brownian motion
	float noise = (1 - worley(position, cells)) * 0.625 
		+ (1 - worley(position, cells * 2)) * 0.25 
		+ (1 - worley(position, cells * 4)) * 0.125;
	noise *= 255;
	image[i] = unsigned char ( (int) noise);
	image[i + 1] = unsigned char ( (int) noise);
	image[i + 2] = unsigned char ( (int) noise);


}

cudaError_t fillImage(unsigned char* image, const int width, const int height, const int depth, const int channels)
{
	unsigned char* dev_image = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for output image, no input required
	cudaStatus = cudaMalloc((void**)&dev_image, width * height * depth * channels * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	//dim3 blocks(32, 32);
	int blocks = (int) ceilf(width * height * depth / 512.0);
	printf("blocks %d", blocks);
	pixelNoise <<<blocks,512>>> (dev_image, width, height, depth, channels, 10);
	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	//cudaStatus = cudaMemcpy(dev_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(image, dev_image, width * height * depth * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_image);
	return cudaStatus;
}


int main()
{
  
	int w, h, d, chans;
	// Should be 32x32x32
	w = 32;
	h = 32;
	d = 32;
	chans = 3;
	unsigned char* img = (unsigned char *) malloc(sizeof(unsigned char) * w * h * d * chans);
	printf("%dx%d, %d channels", w, h, chans);
	
    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    cudaError_t cudaStatus = fillImage(img, w, h, d, chans);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fillimage failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	stbi_write_png("sliced3232.png", w * d, h, chans, img, w * d * chans);
	

    return 0;
}