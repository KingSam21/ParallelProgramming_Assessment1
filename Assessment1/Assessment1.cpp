// ================================================================================================
// TO RUN THE PROGRAM, GO TO A "git.bash" TERMINAL!
// ENTER IN TERMINAL:   g++ Assessment1.cpp -o Assessment1.exe -I"C:\OpenCL-SDK\include" -lOpenCL -lgdi32
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

#include "CImg.h"
using namespace cimg_library;

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
    int n = 800;                                                                                        // Size of the row/column of the grid
    const size_t forest_size = n * n;                                                                   // Size of the forest grid.
    float probTree = 0.8f;                                                                              // Probability of generating a tree
    float probBurning = 0.01f;                                                                          // Probability of a tree burning
    float probImmune = 0.3f;                                                                            // Probability that the tree is immune to burning
    float probLightning = 0.001f;                                                                       // Probability that a tree has been struck by lightning.
    size_t bytes = forest_size * sizeof(int);
    std::vector<int> f_grid(forest_size);                                                               // vector to contain the values initialised in the forest.

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CImg related Variables - Colours
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    const unsigned char emptyColour[] = {255, 255, 255};                                                // Should be white
    const unsigned char treeColour[] = {0, 255, 0};                                                     // Should be green
    const unsigned char burningColour[] = {255, 0, 0};                                                  // Should be Red
    const unsigned char gridlineColour[] = {0, 0, 0};                                                   // Should be black

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Query all available OpenCL platforms.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    std::vector<cl::Platform> platforms;                                                                // Vector for storing all available platforms that use OpenCL.
    cl::Platform::get(&platforms);                                                                      // Retrieve all available OpenCL Platforms

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
        std::cout << "\n====================\n";
        std::cout << "Platform Name:        " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout << "Platform Vendor:      " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
        std::cout << "Platform Version:     " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
        std::cout << "Profile:              " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
        
        // ---------------------------------------------------------------------
        // Query all available Devices for each platform
        // ---------------------------------------------------------------------
        std::vector<cl::Device> devices;                                                                // Vector for storing all of the available OpenCL devices on the platform.
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);                                              // Retrieve all available devices for the platform.
       
        if (devices.empty()) {                                                                          // Check if there are any devices on each platform.
            std::cout << "No devices found for this platform.\n";
            continue;                                                                                   // If there aren't any devices for this platform, skip to the next one.
        } 
        else{
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // For loop iterating through all available devices.
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
                    // Load and Build the Kernels.
                    // ------------------------------------------------------------
                    std::string kernel_source = LoadKernelSource("kernels/forest_codes.cl");            // Call LoadKernelSource helper function to see if it can be called.
                    cl::Program program(context, kernel_source);                                        // Create the program for the kernel code, within the context workspace.
                    program.build({device});                                                            // Build the program, containing kernal code, for each device.
                    
                    // ------------------------------------------------------------
                    // kernel arguments.
                    // ------------------------------------------------------------
                    cl::Kernel kernel(program, "initialise_forest");
                    kernel.setArg(0, d_forestA);                                                        // Sets the current size of a row/column in forest grid.
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
                    std::cout << "Kernel Execution Time: " << elapsedMs << "ms\n"; 
                    
                    // ------------------------------------------------------------
                    // Read back the buffer/kernel output.
                    // ------------------------------------------------------------

                    queue.enqueueReadBuffer (
                                                d_forestA,                                              // The location in which the data is.
                                                CL_TRUE,                                                // Means that the device will wait until the kernel is complete, before moving on.
                                                0,                                                      // The range, 0 means to start from the beginning.
                                                bytes,                                                  // The size of the data.
                                                f_grid.data()                                           // The destination for the result from the kernel.
                    );
                    queue.finish();


                    std::cout << "Simulation Completed!" << std::endl;                                  // Print a statement to the console when the simulation has completed.

                    // ------------------------------------------------------------
                    // Display the initialised state of the forest.
                    // ------------------------------------------------------------
                    CImg<unsigned char> visualiseForest (                                               // Create a new image (In this case, a grid for the forest).
                                                            800,                                        // Width of the window. 
                                                            800,                                        // Height of the window.
                                                            1,                                          // Depth (1 for 2d images)
                                                            3                                           // Channel (colour components: 1 is grayscale, 3 is RGB).
                    ); 

                    for (int y = 0; y < n; y++) {
                        for (int x = 0; x < n; x++) {
                            int treeState = f_grid[y * n + x];                                          // Create a variable recording the current cell position
                            switch (treeState){                                                         // Switch statement to select the appropriate colour according to the cells value.
                                case 0:
                                    visualiseForest.draw_point(x, y, emptyColour);
                                    break;
                                case 1:
                                    visualiseForest.draw_point(x, y, treeColour);
                                    break;
                                case 2:
                                    visualiseForest.draw_point(x, y, burningColour);
                                    break;
                                default:
                                    std::cout << "There was a poblem with cell: (" << x << "," << y << ")." << std::endl; 
                            }
                        }
                    }
                    CImgDisplay ForestFireSimulation(800, 800, "This is a Forest Fire Sim", 0);         // Turn the display window into a variable so it can be used in the while loop.
                    //visualiseForest.display("This should be the first stage of the forest fire sim"); // Display the forest, using CImg Library. (PREVIOUS METHOD)
                    
                    int cellSize = 800/n;
                    
                    // ------------------------------------------------------------
                    // While loop to keep updating the forest fire simulation.
                    // ------------------------------------------------------------
                    cl::Kernel kernel2(program, "update_forest");                                       // Change the current kernel function, to be the 'update_forest'.
                    
                    bool useForestA = true;                                                             // Boolean determining which forest is visualise to prevent race conditions.
                    
                    while (!ForestFireSimulation.is_closed()) {                                         // While the display window is open.
                        // ------------------------------------------------------------
                        // Set the kernel Arguments, changing which forest is A and B.
                        // ------------------------------------------------------------
                        kernel2.setArg(0, useForestA ? d_forestA : d_forestB);                          // The current forest grid.
                        kernel2.setArg(1, useForestA ? d_forestB : d_forestA);                          // The Updated forest grid.
                        kernel2.setArg(2, n);                                                           // The size of the forest grid.
                        kernel2.setArg(3, probImmune);                                                  // Sets the value for the probability of a tree being immune.
                        kernel2.setArg(4, probLightning);                                               // Sets the probability of the tree getting struck by lightning.

                        // ------------------------------------------------------------
                        // Create an event and launch the kernel.
                        // ------------------------------------------------------------
                        queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(forest_size), cl::NullRange);


                        // ------------------------------------------------------------
                        // Read back the results of the kernel.
                        // ------------------------------------------------------------
                        queue.enqueueReadBuffer(useForestA ? d_forestB : d_forestA, CL_TRUE, 0, sizeof(int) * forest_size, f_grid.data());
                        queue.finish();

                        // ------------------------------------------------------------
                        // Assign the appropriate colours by looping through the results.
                        // ------------------------------------------------------------
                        for (int y = 0; y < n; y++) {
                            for (int x = 0; x < n; x++) {
                                int treeState = f_grid[y * n + x];                                      // Create a variable recording the current cell position
                                switch (treeState){                                                     // Switch statement to select the appropriate colour according to the cells value.
                                    case 0:
                                        visualiseForest.draw_rectangle(x * cellSize, y * cellSize, (x + 1) * cellSize - 1, (y + 1) * cellSize - 1, emptyColour);
                                        break;
                                    case 1:
                                        visualiseForest.draw_rectangle(x * cellSize, y * cellSize, (x + 1) * cellSize - 1, (y + 1) * cellSize - 1, treeColour);
                                        break;
                                    case 2:
                                        visualiseForest.draw_rectangle(x * cellSize, y * cellSize, (x + 1) * cellSize - 1, (y + 1) * cellSize - 1, burningColour);
                                        break;
                                    default:
                                        std::cout << "There was a poblem with cell: (" << x << "," << y << ")." << std::endl; 
                                }
                            }
                        }

                        visualiseForest.get_resize(800, 800, 1, 3, 1).display(ForestFireSimulation);

                        useForestA = !useForestA;                                                       // Switch which forest is the current one, for the next iteration.
                        cimg::wait(1000);                                                               // Sets the time in [ms] between iterations.
                    }

                }
                catch (cl::BuildError &e) {
                    std::cerr << "Build Error: " << e.getBuildLog()[0].second << std::endl;             // Print a build log if the kernel fails to compile.
                }
                catch(const std::exception& e)  {
                    std::cerr << e.what() << '\n';
                }     
            }
        }
    }
    return 0;
}