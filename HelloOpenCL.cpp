#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define MAX_VECTOR_LENGTH 500

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <CL/cl.h>

int main()
{
    std::cout << std::endl << "========== PART 1: Reading platforms and devices ==========" << std::endl;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    cl_platform_id* platform_ids = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platform_ids, &num_platforms);

    for (size_t j = 0; j < num_platforms; j++)
    {
        std::cout << "------------------" << std::endl;
        cl_platform_id platform_id = platform_ids[j];
        size_t platform_info_length = 0;
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, nullptr, &platform_info_length);
        char buf[100]{};
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_info_length, buf, &platform_info_length);
        std::cout << "Platform " << j << ": " << buf << std::endl;
        cl_uint num_devices = 0;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        cl_device_id* device_ids = new cl_device_id[num_devices];
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, &num_devices);

        for (size_t i = 0; i < num_devices; i++)
        {
            cl_device_id device_id = device_ids[i];
            size_t device_info_length = 0;
            clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &device_info_length);
            char buf[100]{};
            clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_info_length, buf, &device_info_length);
            std::cout << "Device " << j << ":" << i << ": " << buf << std::endl;
        }
        delete[] device_ids;
    }

    // clean up
    delete [] platform_ids;

//----------------------------------------------------------------------------------------
    std::cout << std::endl << "========== PART 2: Sending some actual work to the host. ==========" << std::endl;

    // Host input vectors
    float h_a[MAX_VECTOR_LENGTH];
    float h_b[MAX_VECTOR_LENGTH];
    // Host output vector
    float h_c[MAX_VECTOR_LENGTH];

    size_t n = 200;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Initialize vectors on host
    for (size_t i = 0; i < n; i++)
    {
        h_a[i] = sinf((float)i) * sinf((float)i);
        h_b[i] = cosf((float)i) * cosf((float)i);
    }

    size_t globalSize, localSize;
    cl_int err;

    // Number of work items in each local work group
    localSize = 16;

    // Number of total work items - localSize must be devisor
    globalSize = localSize * 20;
    if (globalSize < n)
    {
        std::cout << "Insufficient global size. Exiting.";
        return -1;
    }

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Populate the source buffer from the cl file
    std::ifstream kernelFile("vecAdd.cl");
    std::ostringstream sstr;
    sstr << kernelFile.rdbuf();
    std::string kernelSourceString = sstr.str();
    const char* kernelSourceCode = kernelSourceString.c_str();

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**) & kernelSourceCode, NULL, &err);

    // Build the program executable 
    std::cout << "Building the program executable...";
    auto cl_ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (cl_ret == CL_BUILD_PROGRAM_FAILURE) 
    { 
        size_t len = 10000;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char* log = new char[len]; //or whatever you use
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        std::cout << "CL_BUILD_PROGRAM_FAILURE" << std::endl;
        return cl_ret; 
    }
    else
    {
        std::cout << "Done. Kernel build returned: " << cl_ret << std::endl;
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
    if (err == CL_INVALID_KERNEL_NAME)
    {
        std::cout << "Unable to create kernel (CL_INVALID_KERNEL_NAME). Exiting." << std::endl;
        return err;
    }

    // Create the input and output arrays in device memory for our calculation
    size_t bytes = n * sizeof(float);
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

    //Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;
    for (size_t i = 0; i < n; i++)
    {
        sum += h_c[i];
    }
    std::cout << "Final result: " << sum / n << std::endl;

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}


