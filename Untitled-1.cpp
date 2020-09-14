#pragma warning(disable:4996)
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <iostream>
//#include "cnn.h"

#define CHECK_ERROR(err) \
   if(err != CL_SUCCESS) { \
      printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      exit(EXIT_FAILURE); \
   }

cl_command_queue queues;
cl_context context;
cl_kernel kernel_convolution_layer;
cl_kernel kernel_reduction_layer;
cl_kernel kernel_ReLU;

static void pooling2x2(float* input, float* output, int N) {
	int i, j, k, l;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			float max = 0;
			for (k = 0; k < 2; k++) {
				for (l = 0; l < 2; l++) {
					float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
					max = (max > pixel) ? max : pixel;
				}
			}
			output[i * N + j] = max;

		}
	}
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float* inputs, float* outputs, int D, int N) {
	int i;
	for (i = 0; i < D; i++) {
		float* input = inputs + i * N * N * 4;
		float* output = outputs + i * N * N;
		pooling2x2(input, output, N);
	}
}

static void convolution3x3(float* input, float* output, float* filter, int N) {
	int i, j, k, l;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			float sum = 0;
			for (k = 0; k < 3; k++) {
				for (l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)
						sum += input[x * N + y] * filter[k * 3 + l];
				}
			}
			output[i * N + j] += sum;
		}
	}
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
#define ReLU(x) (((x)>0)?(x):0)


 //static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
 //   int i, j;
 //   
 //   memset(outputs, 0, sizeof(float) * N * N * D2);
 //
 //   for (j = 0; j < D2; j++) {
 //      for (i = 0; i < D1; i++) {
 //         float* input = inputs + N * N * i;
 //         float* output = outputs + N * N * j;
 //         float* filter = filters + 3 * 3 * (j * D1 + i);
 //         convolution3x3(input, output, filter, N);
 //      }
 //   }
 //
 //   for (i = 0; i < D2; i++) {
 //
 //      float* output = outputs + N * N * i;
 //      float bias = biases[i];
 //      for (j = 0; j < N * N; j++) {
 //         output[j] = ReLU(output[j] + bias);
 //      }
 //   }
 //}

static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
	cl_int err;
	cl_int f_l = 3;


	cl_mem buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N * D1, inputs, &err);
	CHECK_ERROR(err);

	cl_mem buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N * D1 * D2, NULL, &err);
	CHECK_ERROR(err);

	cl_mem buffer3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 9 * D1 * D2, filters, &err);
	CHECK_ERROR(err);

	cl_mem buffer4 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D2, biases, &err);
	CHECK_ERROR(err);

	cl_mem buffer5 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N * D2, NULL, &err);
	CHECK_ERROR(err);


	err = clSetKernelArg(kernel_convolution_layer, 0, sizeof(cl_mem), &buffer1);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 1, sizeof(cl_mem), &buffer2);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 2, sizeof(cl_mem), &buffer3);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 3, 30000, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 4, sizeof(float) * 9, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 5, sizeof(int), &N);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 6, sizeof(int), &D1);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution_layer, 7, sizeof(int), &D2);
	CHECK_ERROR(err);


	size_t global_size[3] = { N , N, D1 };
	size_t local_size[3] = { N / 2,N / 2, 1 };
	if (N < 32)
	{
		local_size[0] = N;
		local_size[1] = N;
		local_size[2] = 1;
	}

	for (int q = 0; q < D2; q++) {
		//convol, setarg

		err = clSetKernelArg(kernel_convolution_layer, 8, sizeof(int), &q);
		CHECK_ERROR(err);


		err = clEnqueueNDRangeKernel(queues, kernel_convolution_layer, 3, NULL, global_size, local_size, 0, NULL, NULL);
		CHECK_ERROR(err);

	}
	err = clEnqueueReadBuffer(queues, buffer2, CL_FALSE, 0, sizeof(float) * N * N * D2, outputs, 0, NULL, NULL);
	CHECK_ERROR(err);
	for (int m = 0; m < 32; m++) {
		for (int n = 0; n < 32; n++) {
			printf("%3d x %2d %f\n", m, n, outputs[m * 32 + n]);
		}
	}
	system("pause");



	//reduction, setarg

	err = clSetKernelArg(kernel_reduction_layer, 0, sizeof(cl_mem), &buffer2);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_reduction_layer, 1, sizeof(cl_mem), &buffer5);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_reduction_layer, 2, sizeof(float) * 1024, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_reduction_layer, 3, sizeof(int), &N);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_reduction_layer, 4, sizeof(int), &D1);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_reduction_layer, 5, sizeof(int), &D2);
	CHECK_ERROR(err);
	for (int r = 0; r < D2; r++) {
		err = clSetKernelArg(kernel_reduction_layer, 6, sizeof(int), &r);
		CHECK_ERROR(err);



		if (D1 != 3) {
			global_size[2] = D1 / 2;
			local_size[0] = 1;
			local_size[1] = 1;
			local_size[2] = D1 / 2;
		}
		else {
			local_size[0] = 1;
			local_size[1] = 1;
			local_size[2] = D1;
		}

		clEnqueueNDRangeKernel(queues, kernel_reduction_layer, 3, NULL, global_size, local_size, 0, NULL, NULL);
	}








	//ReLU, setarg

	err = clSetKernelArg(kernel_ReLU, 0, sizeof(cl_mem), &buffer5);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_ReLU, 1, sizeof(cl_mem), &buffer4);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_ReLU, 2, sizeof(int), &N);
	CHECK_ERROR(err);

	global_size[2] = D2;
	if (N > 16)
	{
		local_size[0] = 16;
		local_size[1] = 16;
		local_size[2] = 1;
	}
	else
	{
		local_size[0] = N;
		local_size[1] = N;
		local_size[2] = 256 / N / N;
	}

	err = clEnqueueNDRangeKernel(queues, kernel_ReLU, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueReadBuffer(queues, buffer5, CL_FALSE, 0, sizeof(float) * N * N * D2, outputs, 0, NULL, NULL);
	CHECK_ERROR(err);



	clReleaseMemObject(buffer1);
	clReleaseMemObject(buffer2);
	clReleaseMemObject(buffer3);
	clReleaseMemObject(buffer4);
	clReleaseMemObject(buffer5);
}


/*
 * M = output size
 * N = input size
 */
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {
	int i, j;
	for (j = 0; j < M; j++) {
		float sum = 0;
		for (i = 0; i < N; i++) {
			sum += input_neuron[i] * weights[j * N + i];
		}
		sum += biases[j];
		output_neuron[j] = ReLU(sum);
	}
}

static void softmax(float* output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float* fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	// nothing to init in the sequential version
}


char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);
	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') {
			cnt++;
		}
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

const char* CLASS_NAME[] = {
   "airplane",
   "automobile",
   "bird",
   "cat",
   "deer",
   "dog",
   "frog",
   "horse",
   "ship",
   "truck"
};

void print_usage_and_exit(char** argv) {
	fprintf(stderr, "Usage: %s <number of image> <output>\n", argv[0]);
	fprintf(stderr, " e.g., %s 3000 result.out\n", argv[0]);
	exit(EXIT_FAILURE);
}

void* read_bytes(const char* fn, size_t n) {
	FILE* f = fopen(fn, "rb");
	void* bytes = malloc(n);
	size_t r = fread(bytes, 1, n, f);
	fclose(f);
	if (r != n) {
		fprintf(stderr,
			"%s: %zd bytes are expected, but %zd bytes are read.\n",
			fn, n, r);
		exit(EXIT_FAILURE);
	}
	return bytes;
}

/*
 * Read images from "cifar10_image.bin".
 * CIFAR-10 test dataset consists of 10000 images with (3, 32, 32) size.
 * Thus, 10000 * 3 * 32 * 32 * sizeof(float) = 122880000 bytes are expected.
 */
const int IMAGE_CHW = 3 * 32 * 32 * sizeof(float);
float* read_images(size_t n) {
	return (float*)read_bytes("cifar10_image.bin", n * IMAGE_CHW);
}

/*
 * Read labels from "cifar10_label.bin".
 * 10000 * sizeof(int) = 40000 bytes are expected.
 */
int* read_labels(size_t n) {
	return (int*)read_bytes("cifar10_label.bin", n * sizeof(int));
}

/*
 * Read network from "network.bin".
 * conv1_1 : weight ( 64,   3, 3, 3) bias ( 64)
 * conv1_2 : weight ( 64,  64, 3, 3) bias ( 64)
 * conv2_1 : weight (128,  64, 3, 3) bias (128)
 * conv2_2 : weight (128, 128, 3, 3) bias (128)
 * conv3_1 : weight (256, 128, 3, 3) bias (256)
 * conv3_2 : weight (256, 256, 3, 3) bias (256)
 * conv3_3 : weight (256, 256, 3, 3) bias (256)
 * conv4_1 : weight (512, 256, 3, 3) bias (512)
 * conv4_2 : weight (512, 512, 3, 3) bias (512)
 * conv4_3 : weight (512, 512, 3, 3) bias (512)
 * conv5_1 : weight (512, 512, 3, 3) bias (512)
 * conv5_2 : weight (512, 512, 3, 3) bias (512)
 * conv5_3 : weight (512, 512, 3, 3) bias (512)
 * fc1     : weight (512, 512) bias (512)
 * fc2     : weight (512, 512) bias (512)
 * fc3     : weight ( 10, 512) bias ( 10)
 * Thus, 60980520 bytes are expected.
 */
const int NETWORK_SIZES[] = {
   64 * 3 * 3 * 3, 64,
   64 * 64 * 3 * 3, 64,
   128 * 64 * 3 * 3, 128,
   128 * 128 * 3 * 3, 128,
   256 * 128 * 3 * 3, 256,
   256 * 256 * 3 * 3, 256,
   256 * 256 * 3 * 3, 256,
   512 * 256 * 3 * 3, 512,
   512 * 512 * 3 * 3, 512,
   512 * 512 * 3 * 3, 512,
   512 * 512 * 3 * 3, 512,
   512 * 512 * 3 * 3, 512,
   512 * 512 * 3 * 3, 512,
   512 * 512, 512,
   512 * 512, 512,
   10 * 512, 10
};

float* read_network() {
	return (float*)read_bytes("network.bin", 60980520);
}

float** slice_network(float* p) {
	float** r = (float**)malloc(sizeof(float*) * 32);
	for (int i = 0; i < 32; ++i) {
		r[i] = p;
		p += NETWORK_SIZES[i];
	}
	return r;
}

int main(int argc, char** argv) {
	using namespace std::chrono;
	using namespace std;

	cl_uint num_platforms;
	cl_platform_id* platforms;
	cl_device_id device;

	char str[1024];
	size_t max_work_group_size;
	cl_uint p, d;
	cl_int err;

	cl_program program;
	const char* source_code = "";
	size_t source_size = strlen(source_code);
	char* kernel_source;
	size_t kernel_source_size;
	cl_mem buffer1, buffer2, buffer3, buffer4;

	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queues = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,
		&kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	kernel_convolution_layer = clCreateKernel(program, "convolution_layer", &err);
	CHECK_ERROR(err);
	kernel_reduction_layer = clCreateKernel(program, "reduction_layer", &err);
	CHECK_ERROR(err);
	kernel_ReLU = clCreateKernel(program, "ReLU", &err);
	CHECK_ERROR(err);


	if (argc != 3) {
		print_usage_and_exit(argv);
	}

	int num_images = atoi(argv[1]);
	float* images = read_images(num_images);
	float* network = read_network();
	float** network_sliced = slice_network(network);
	int* labels = (int*)calloc(num_images, sizeof(int));
	float* confidences = (float*)calloc(num_images, sizeof(float));
	time_t start, end;

	cnn_init();
	start = clock();


	// slice the network into weights and biases
	float* w1_1, * b1_1, * w1_2, * b1_2;
	float* w2_1, * b2_1, * w2_2, * b2_2;
	float* w3_1, * b3_1, * w3_2, * b3_2, * w3_3, * b3_3;
	float* w4_1, * b4_1, * w4_2, * b4_2, * w4_3, * b4_3;
	float* w5_1, * b5_1, * w5_2, * b5_2, * w5_3, * b5_3;
	float* w1, * b1, * w2, * b2, * w3, * b3;
	w1_1 = network_sliced[0]; b1_1 = network_sliced[1];
	w1_2 = network_sliced[2]; b1_2 = network_sliced[3];
	w2_1 = network_sliced[4]; b2_1 = network_sliced[5];
	w2_2 = network_sliced[6]; b2_2 = network_sliced[7];
	w3_1 = network_sliced[8]; b3_1 = network_sliced[9];
	w3_2 = network_sliced[10]; b3_2 = network_sliced[11];
	w3_3 = network_sliced[12]; b3_3 = network_sliced[13];
	w4_1 = network_sliced[14]; b4_1 = network_sliced[15];
	w4_2 = network_sliced[16]; b4_2 = network_sliced[17];
	w4_3 = network_sliced[18]; b4_3 = network_sliced[19];
	w5_1 = network_sliced[20]; b5_1 = network_sliced[21];
	w5_2 = network_sliced[22]; b5_2 = network_sliced[23];
	w5_3 = network_sliced[24]; b5_3 = network_sliced[25];
	w1 = network_sliced[26]; b1 = network_sliced[27];
	w2 = network_sliced[28]; b2 = network_sliced[29];
	w3 = network_sliced[30]; b3 = network_sliced[31];

	//float* c1_1_t;
	// allocate memory for output of each layer
	float* c1_1, * c1_2, * p1;
	float* c2_1, * c2_2, * p2;
	float* c3_1, * c3_2, * c3_3, * p3;
	float* c4_1, * c4_2, * c4_3, * p4;
	float* c5_1, * c5_2, * c5_3, * p5;
	float* fc1, * fc2, * fc3;
	c1_1 = alloc_layer(64 * 32 * 32);
	//c1_1_t = alloc_layer(64 * 32 * 32);
	c1_2 = alloc_layer(64 * 32 * 32);
	p1 = alloc_layer(64 * 16 * 16);
	c2_1 = alloc_layer(128 * 16 * 16);
	c2_2 = alloc_layer(128 * 16 * 16);
	p2 = alloc_layer(128 * 8 * 8);
	c3_1 = alloc_layer(256 * 8 * 8);
	c3_2 = alloc_layer(256 * 8 * 8);
	c3_3 = alloc_layer(256 * 8 * 8);
	p3 = alloc_layer(256 * 4 * 4);
	c4_1 = alloc_layer(512 * 4 * 4);
	c4_2 = alloc_layer(512 * 4 * 4);
	c4_3 = alloc_layer(512 * 4 * 4);
	p4 = alloc_layer(512 * 2 * 2);
	c5_1 = alloc_layer(512 * 2 * 2);
	c5_2 = alloc_layer(512 * 2 * 2);
	c5_3 = alloc_layer(512 * 2 * 2);
	p5 = alloc_layer(512 * 1 * 1);
	fc1 = alloc_layer(512);
	fc2 = alloc_layer(512);
	fc3 = alloc_layer(10);

	// run network
	for (int i = 0; i < num_images; ++i)
	{
		printf(" %d 번쨰 \n", i);
		float* image = images + (i * 3 * 32 * 32);

		//convolution_layer(*input, *output, *filters, *biases, output channels,input channels, input size);
		//pooling_layer(*input, *output, channels,output size);

		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);
		/*for (int m = 0; m < 32; m++) {
		   printf("%f\n",c1_1[m]);
		}
		system("pause");*/


		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
		pooling_layer(c1_2, p1, 64, 16);

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
		pooling_layer(c2_2, p2, 128, 8);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		pooling_layer(c3_3, p3, 256, 4);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		pooling_layer(c4_3, p4, 512, 2);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		pooling_layer(c5_3, p5, 512, 1);

		//fc_layer(*input, *output, *weights, *biases, output size, input size);
		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		softmax(fc3, 10);

		labels[i] = find_max(fc3, 10);
		confidences[i] = fc3[labels[i]];
	}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);


	end = clock();
	printf("Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);

	FILE* of = fopen(argv[2], "w");
	int* labels_ans = read_labels(num_images);
	double acc = 0;
	for (int i = 0; i < num_images; ++i) {
		fprintf(of, "Image %04d: %s %f\n", i, CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i]) ++acc;
	}
	fprintf(of, "Accuracy: %f\n", acc / num_images);
	fclose(of);

	free(images);
	free(network);
	free(network_sliced);
	free(labels);
	free(confidences);
	free(labels_ans);

	return 0;
}
