//a very simple histogram implementation
kernel void hist_intensity(global const uchar* A, global int* H)
{
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//a very simple histogram implementation
kernel void hist_cumalative(global const int* A, global int* H)
{
	//get the global id of the kernal
	int id = get_global_id(0);
	//get the global size of the kernal
	int N = get_global_size(0);

	//atomically add the next output of the vector
	for (int i = id + 1; i < N && id < N; i++)
		atomic_add(&H[i], A[id]);
}

kernel void hist_normalise(global const int* A, global float* H)
{
	//get the global id of the kernal
	int id_count = get_global_id(0);
	float max_pixels = 699392;
	int max_range = 255;
	
	/*The equation is described as :
	"The normalized count is the count in the class divided by the number of observations times the class width"
	Within my code this would be the cumalative histogram output / number of pixels * the maxiumum range for the vector / image(255 (RGB))
	*/
	H[id_count] = A[id_count] / max_pixels * max_range;

}

kernel void LUT(global const uchar* A, global float* Normalisation_array, global uchar* H)
{
	int pixel_id = get_global_id(0);
	
	//Iterate through each pixel value in the original image and replace their values with the their corresponding normalized values
	H[pixel_id] = Normalisation_array[A[pixel_id]];
}


