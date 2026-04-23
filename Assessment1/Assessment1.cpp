// ================================================================================================
// TO RUN THE PROGRAM, GO TO A "git.bash" TERMINAL!
// ENTER IN TERMINAL:   g++ Assessment1.cpp -o Assessment1.exe -I"C:\OpenCL-SDK\include" -lOpenCL
// THEN ENTER:          ./Assessment1.exe
// ================================================================================================

// ================================================================================================
// Import Libraries
// ================================================================================================
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>



//#include "CImg.h"

// using namespace cimg_library;


// ================================================================================================
// Function to Read and Open the kernel, to check if it has errors opening.
// ================================================================================================
std::string LoadKernelSource(const std::string &filePath) {
    std::ifstream file(filePath);                                                                       // Open and read the kernel file.
    if (!file.is_open()) throw std::runtime_error("Cannot open kernel file: " + filePath);              // Throw an error if the file cannot be opened,
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());         // Otherwise, return file as string, so it can be used in CL::Program
}

// ================================================================================================
// MAIN METHOD
// ================================================================================================
int main() {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Declare variables - these determine the properties for the forest. (random number > probability = true)
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int n = 100;                                                                                        // Size of the row/column of the grid
    const size_t forest_size = n * n;                                                                   // Size of the forest grid.
    float probTree = 0.8f;                                                                              // Probability of genreating a tree
    float probBurning = 0.01f;                                                                          // Probability of a tree burning
    float probImmune = 0.3f;                                                                            // Probability that the tree is immune to burning
    float probLightning = 0.001f;                                                                       // Probability that a tree has been struck by lightning.

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Query all availible OpenCL platforms.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    std::vector<cl::Platform> platforms;                                                                // Vector for storing all availble platforms that use OpenCL.
    cl::Platform::get(&platforms);                                                                      // Retrieve all availible OpenCL Platforms

    if (platforms.empty())  {                                                                           // Check to see that list of platforms is not empty.
        std::cout << "No OpenCL platforms found\n";                                                     // If it is empty inform the user.
        return 1;                                                                                       // End the program
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // For loop iterating through all available platforms.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (auto &platform : platforms) {
        // ---------------------------------------------------------------------
        // Print information about each platform
        // ---------------------------------------------------------------------
        std::cout << "====================\n";
        std::cout << "Platform Name:        " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout << "Platform Vendor:      " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
        std::cout << "Platform Version:     " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
        std::cout << "Profile:              " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
        
        // ---------------------------------------------------------------------
        // Query all availible Devices for each platform
        // ---------------------------------------------------------------------
        std::vector<cl::Device> devices;                                                                // Vector for storing all of the available OpenCL devices on the platform.
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);                                              // Retrieve all available devices for the platform.
       
        if (devices.empty()) {                                                                          // Check if there are any devices on each platform.
            std::cout << "No devices found for this platform.\n";
            continue;                                                                                   // If there aren't any devices for this platform, skip to the next one.
        } 
        else{
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // For loop iterating thorugh all availible devices.
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for (auto &device : devices) {
                // ------------------------------------------------------------
                // Print information about each Device
                // ------------------------------------------------------------
                std::cout << "--------------------\n";
                std::cout << "Device Name:          " << device.getInfo<CL_DEVICE_NAME>() << "\n";
                std::cout << "Device Vendor:        " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
                std::cout << "Max Compute Units:    " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
                std::cout << "Global Memory Size:   " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << "\n";
                
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                // Try Catch to identify/deal with any problems when calling kernels.
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                try {
                    // ------------------------------------------------------------
                    // Create an OpenCL context and Command Queue
                    // ------------------------------------------------------------
                    cl::Context context(device);                                                        // This is basically a workspace in which everything happens.
                    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);                 // This creates a queue to push commands to the chosen device.
                    
                    // ------------------------------------------------------------
                    // Buffers for the forest.                                                          // Allocates memory on the selected device. (memory is called buffer).
                    // ------------------------------------------------------------
                    cl::Buffer d_forestA(context, CL_MEM_READ_WRITE, sizeof(int) * forest_size);        // Buffer, for the current forest grid.
                    cl::Buffer d_forestB(context, CL_MEM_READ_WRITE, sizeof(int) * forest_size);        // Buffer, for the updated forest grid.

                    // ------------------------------------------------------------
                    // Load and Build the Kernals.
                    // ------------------------------------------------------------
                    std::string kernel_source = LoadKernelSource("kernals/forest_codes.cl");            // Call LoadKernelSource helper function to see if it can be called.
                    cl::Program program(context, kernel_source);                                        // Create the program for the kernel code, within the context workspace.
                    program.build({device});                                                            // Build the program, containing kernal code, for each device.
                    
                    // ------------------------------------------------------------
                    // kernel arguments.
                    // ------------------------------------------------------------
                    cl::Kernel kernel(program, "initialise_forest");
                    kernel.setArg(0, d_forestA);                                                        // Sets the current forest grid.
                    kernel.setArg(1, n);                                                                // Sets the size of the forest grid.
                    kernel.setArg(2, probTree);                                                         // Sets the value for the probability of spawning a tree.
                    kernel.setArg(3, probBurning);                                                      // Sets the probability of the tree burning.


                    // ------------------------------------------------------------
                    // Create an event and launch the kernel.
                    // ------------------------------------------------------------
                    cl::Event event;
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(forest_size), cl::NullRange, nullptr, &event);
                    queue.finish();

                    // ------------------------------------------------------------
                    // Display how long it took for the kernel to execute.
                    // ------------------------------------------------------------
                    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                    double elapsedMs = (end - start) * 1.0e-6;
                    std::cout << "Kernel Execution Time:" << elapsedMs << "ms\n"; 
                    
                    // ------------------------------------------------------------
                    // Read back the buffer/kernel output.
                    // ------------------------------------------------------------
                    std::vector<int> f_grid(forest_size);
                    queue.enqueueReadBuffer(d_forestA, CL_TRUE, 0, sizeof(int) * forest_size, f_grid.data());


                }
                catch (cl::BuildError &e) {
                    std::cerr << "Build Error: " << e.getBuildLog()[0].second << std::endl;
                }
                catch(const std::exception& e)  {
                    std::cerr << e.what() << '\n';
                }     
            }
        }
    }

    return 0;
}