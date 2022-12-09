/*image segmentation
: threshold/ adaptive thresholding/ mean filter를 uniform mean filter이용*/

#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat adaptive_thres(const Mat input, int n, float b);

int main() {

	Mat input = imread("./image/coin_A4_nice_black.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;

	cvtColor(input, input_gray, CV_RGB2GRAY); // Converting image to gray

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	float resize_ratio = 0.1;
	namedWindow("Grayscale", WINDOW_NORMAL);
	resizeWindow("Grayscale", input.cols * resize_ratio, input.rows * resize_ratio); //원본비율 유지한채 사이즈 조절
	imshow("Grayscale", input_gray);

	output = adaptive_thres(input_gray, 2, 0.9); //Fix with uniform mean filtering with zero paddle

	namedWindow("Adaptive_threshold", WINDOW_NORMAL);
	resizeWindow("Adaptive_threshold", input.cols * resize_ratio, input.rows * resize_ratio);
	imshow("Adaptive_threshold", output);


	waitKey(0);

	return 0;
}


Mat adaptive_thres(const Mat input, int n, float bnumber) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);

	//(uniform mean filtering) Initialiazing Kernel Matrix
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	float kernelvalue = kernel.at<float>(0, 0);  // To simplify, as the filter is uniform. All elements of the kernel value are same.

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) { //for each pixel in the output
		for (int j = 0; j < col; j++) {

			/*STEP1. finds the mean intensity using uniform mean filtering with zero paddle border process.*/
			// m(i,j) = (i,j)에서 uniform mean filter와 convolution
			float sum1 = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {

					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
					}
				}
			}

			// T(i,j) = b * m(i,j) 
			float temp = bnumber * (G)sum1;

			/*STEP2. find output using adaptive thresholding*/
			// g(i,j) = { if( I(i,j) > T(i,j) 255, else 0}
			// Fill code that makes output image's pixel intensity to 255 if the intensity of the input image is bigger than the temp value, else 0.
			if (input.at<G>(i, j) > temp)
				output.at<G>(i, j) = 255;
			else
				output.at<G>(i, j) = 0;
		}
	}
	return output;
}
