#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	//image input variable
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		//read the image 
		CImg<unsigned char> image_input(image_filename.c_str());
		//variable for displaying the image
		CImgDisplay disp_input(image_input,"input");

		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//create debugging event for profilling 
		cl::Event prof_event;
		cl::Event memory_event1;
		cl::Event memory_event2;
		cl::Event memory_event3;
		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//get the size of the image
		vector<unsigned char> image_size(image_input.size());

		//defining the vector type - int for most tasks - float for normalisation
		typedef int mytype;
		typedef float myfloattype;

		// ----- Intensity histogram code -----

		//output vector
		std::vector<mytype> Intensity_vector(255, 0);
		//size of vector in bytes
		size_t Intensity_output_size = Intensity_vector.size() * sizeof(mytype);

		//device - buffers 
		//The input of the buffer needs to be the greyscaled image provided the output buffer should be the output vector created earlier
		cl::Buffer Intensity_Input_Buffer(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer Intensity_Output_Buffer(context, CL_MEM_READ_WRITE, Intensity_output_size); 

		//Intialise input and output on device memoryy
		queue.enqueueWriteBuffer(Intensity_Input_Buffer, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &memory_event1);
		queue.enqueueFillBuffer(Intensity_Output_Buffer, 0, 0, Intensity_output_size);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel Intensity_Kernal = cl::Kernel(program, "hist_intensity");
		//set kernal arguments
		Intensity_Kernal.setArg(0, Intensity_Input_Buffer);
		Intensity_Kernal.setArg(1, Intensity_Output_Buffer);

		//Calls all kernals in a sequence 
		queue.enqueueNDRangeKernel(Intensity_Kernal, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		//copy result from device to host and get output
		queue.enqueueReadBuffer(Intensity_Output_Buffer, CL_TRUE, 0, Intensity_output_size, &Intensity_vector[0], NULL, &memory_event2);

		//display vector output to user
		std::cout << "Intensity Vector" << Intensity_vector << std::endl;

		//print the profilling information about the algorithm - tells the user information about the kernals 
		std::cout << "Intesity kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "All Intensity kernal data: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Memory transfer time [ns]:" << memory_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() + memory_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		// ----- Cummaliative histogram -----

		//output vector
		std::vector<mytype> Cumalative_vector(255, 0);
		//size of vector in bytes
		size_t Cumalative_output_size = Intensity_vector.size() * sizeof(mytype);

		//device - buffers 
		//The input of the buffer needs to be the intensity vector provided the output buffer should be the output vector created earlier
		cl::Buffer Cumalative_Input_Buffer(context, CL_MEM_READ_ONLY, Intensity_output_size);
		cl::Buffer Cumalative_Output_Buffer(context, CL_MEM_READ_WRITE, Cumalative_output_size);

		//Intialise input and output on device memoryy
		queue.enqueueWriteBuffer(Cumalative_Input_Buffer, CL_TRUE, 0, Intensity_output_size, &Intensity_vector[0], NULL, &memory_event1);
		queue.enqueueFillBuffer(Cumalative_Output_Buffer, 0, 0, Cumalative_output_size);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel Cumalative_Kernal = cl::Kernel(program, "hist_cumalative");
		//set kernal arguments
		Cumalative_Kernal.setArg(0, Cumalative_Input_Buffer);
		Cumalative_Kernal.setArg(1, Cumalative_Output_Buffer);

		//Calls all kernals in a sequence 
		queue.enqueueNDRangeKernel(Cumalative_Kernal, cl::NullRange, cl::NDRange(Cumalative_output_size), cl::NullRange, NULL, &prof_event);
		//copy result from device to host and get output
		queue.enqueueReadBuffer(Cumalative_Output_Buffer, CL_TRUE, 0, Cumalative_output_size, &Cumalative_vector[0], NULL, &memory_event2);

		//display vector output to user
		std::cout << "Cumalative vector" << Cumalative_vector << std::endl;

		//print the profilling information about the algorithm - tells the user information about the kernals 
		std::cout << "Cumalative kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "All Cumalative kernal data: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Memory transfer time [ns]:" << memory_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() + memory_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>()<< std::endl;

		// ----- Normalised Histogram -----
		//A noramlised histogram is a histogram where the sum of all the frequencies is exactly 1 - 
		//this means that each frquency must be represented as a percentage - Input will need to be divided
		//The equation is described as:
		//"The normalized count is the count in the class divided by the number of observations times the class width" - https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm
		//Within my code this would be the cumalative histogram output / number of pixels * the maxiumum range for the vector/image (255 (RGB))
		//The output of the cumalative vector tells me how many pixels are in the image at "699392"
		

		//output vector 
		//since the normalisation deals with percentages and decimal numbers we use a float instead of an int
		std::vector<myfloattype> Normalisation_vector(255, 0);
		//size of vector in bytes
		size_t Normalisation_output_size = Intensity_vector.size() * sizeof(myfloattype);

		//device - buffers 
		//The input of the buffer needs to be the cumalative vector the output buffer should be the output vector created earlier
		cl::Buffer Normalisation_Input_Buffer(context, CL_MEM_READ_ONLY, Cumalative_output_size);
		cl::Buffer Normalisation_Output_Buffer(context, CL_MEM_READ_WRITE, Normalisation_output_size);

		//Intialise input and output on device memoryy
		queue.enqueueWriteBuffer(Normalisation_Input_Buffer, CL_TRUE, 0, Cumalative_output_size, &Cumalative_vector[0], NULL, &memory_event1);
		queue.enqueueFillBuffer(Normalisation_Output_Buffer, 0, 0, Normalisation_output_size);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel Normalisation_Kernal = cl::Kernel(program, "hist_normalise");
		//set kernal arguments
		Normalisation_Kernal.setArg(0, Normalisation_Input_Buffer);
		Normalisation_Kernal.setArg(1, Normalisation_Output_Buffer);

		//Calls all kernals in a sequence 
		queue.enqueueNDRangeKernel(Normalisation_Kernal, cl::NullRange, cl::NDRange(Normalisation_output_size), cl::NullRange, NULL, &prof_event);
		//copy result from device to host and get output
		queue.enqueueReadBuffer(Normalisation_Output_Buffer, CL_TRUE, 0, Normalisation_output_size, &Normalisation_vector[0], NULL, &memory_event2);

		//display vector output to user
		std::cout << "Normalisation vector" << Normalisation_vector << std::endl;

		//print the profilling information about the algorithm - tells the user information about the kernals 
		std::cout << "Normalisation Kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "All normalisation kernal data: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Memory transfer time [ns]:" << memory_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() + memory_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		// ----- Lookup Table section ----	
		//Iterate through each gray scale pixel values in the original image and replace their values with the their corresponding normalized values 
		//This basically means we have to use the ID to iterate through the pixels in the image and replace those with the normalised cumalative output from the previous kernals
		//After we apply this to an image 
		
		//The input of the buffer needs to be the greyscaled image provided the output buffer should be the output vector created earlier
		cl::Buffer image_Input_Buffer(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer LUT(context, CL_MEM_READ_ONLY, Normalisation_output_size);
		cl::Buffer image_Output_Buffer(context, CL_MEM_READ_WRITE, image_input.size());

		//Intialise input and output on device memoryy
		queue.enqueueWriteBuffer(image_Input_Buffer, CL_TRUE, 0, image_input.size(), &image_input[0], NULL, &memory_event1);
		queue.enqueueWriteBuffer(LUT, 0, 0, Normalisation_output_size, &Normalisation_vector[0], NULL, &memory_event3);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel LUT_Kernal = cl::Kernel(program, "LUT");
		//set kernal arguments
		//Create 3 kernals for the LUT as we need one input and two outputs (vector and image)
		LUT_Kernal.setArg(0, image_Input_Buffer);
		LUT_Kernal.setArg(1, LUT);
		LUT_Kernal.setArg(2, image_Output_Buffer);

		//Calls all kernals in a sequence 
		queue.enqueueNDRangeKernel(LUT_Kernal, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);

		//copy result from device to host and get output
		queue.enqueueReadBuffer(image_Output_Buffer, CL_TRUE, 0, image_size.size(), &image_size.data()[0], NULL, &memory_event2);

		//print the profilling information about the algorithm - tells the user information about the kernals 
		std::cout << "LUT Kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "All LUT kernal data: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Memory execution time [ns]:" << memory_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() + memory_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() + memory_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>()  << std::endl;
		//image output 
		CImg<unsigned char> output_image(image_size.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//display image output
		CImgDisplay disp_output(output_image, "output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

		//Your program should also report memory transfer, kernel execution, and total program execution times for performance assessment.
		//In such a case, your program should runand display execution times for different variants of your algorithms.

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
