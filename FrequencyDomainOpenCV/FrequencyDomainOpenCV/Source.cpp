#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

Mat generateLowPassFilterMask(Mat complexI, int radius);
Mat generatehighPassFilterMask(Mat complexI, int radius);
Mat generateBandPassFilterMask(Mat complexI, int radius, int radius2);
Mat multiplyDFT(Mat complexI, Mat kernel_spec);
Mat calculateIDFT(Mat complexI);
Mat getFilterKernel(Mat FilterMask);
void shift(Mat magI);
Mat gaussianMask(Mat complexI, double sigma, int ksize);

int main()
{
	//const char* filename = argc >= 2 ? argv[1] : "lena.jpg";

	Mat I = imread("utb1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (I.empty())
		return -1;

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;

	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix



	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	shift(magI);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).

	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);


	/*    Generating Filter Masks, Multiplying with DFT's of image with DFT of mask and then calculating IDFT and displaying   */
	Mat complexILowPass = complexI.clone();
	Mat complexIHighPass = complexI.clone();
	Mat complexIBandPass = complexI.clone();
	Mat complexGaussFilter = complexI.clone();

	Mat LowPassFilterMask = generateLowPassFilterMask(complexILowPass, 45);
	Mat highPassFilterMask = generatehighPassFilterMask(complexIHighPass, 10);
	Mat bandPassFilterMask = generateBandPassFilterMask(complexIBandPass, 35, 85);
	Mat gaussMask = gaussianMask(complexI, -1,200);

	Mat lowPassKernel = getFilterKernel(LowPassFilterMask);
	Mat highPassKernel = getFilterKernel(highPassFilterMask);
	Mat bandPassKernel = getFilterKernel(bandPassFilterMask);
	Mat gaussFilterKernel = getFilterKernel(gaussMask);

	Mat lowPassDFTproduct = multiplyDFT(complexILowPass, lowPassKernel);
	Mat highPassDFTproduct = multiplyDFT(complexIHighPass, highPassKernel);
	Mat bandPassDFTproduct = multiplyDFT(complexIBandPass, bandPassKernel);
	Mat gaussDFTproduct = multiplyDFT(complexGaussFilter, gaussFilterKernel);

	Mat LowPassIDFT = calculateIDFT(lowPassDFTproduct);
	Mat highPassIDFT = calculateIDFT(highPassDFTproduct);
	Mat bandPassIDFT = calculateIDFT(bandPassDFTproduct);
	Mat gaussFilterIDFT = calculateIDFT(gaussDFTproduct);

	imshow("Low Pass Filtered", LowPassIDFT);
	imshow("High Pass Filtered", highPassIDFT);
	imshow("Band Pass Filtered", bandPassIDFT);
	imshow("Gaussian Filtered", gaussFilterIDFT);




	waitKey();

	return 0;
}


void shift(Mat magI) {
	// crop if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat multiplyDFT(Mat complexI, Mat kernel_spec)
{
	shift(complexI);
	//Multiplying the DFT of image with DFT of the filter 
	mulSpectrums(complexI, kernel_spec, complexI, DFT_ROWS);

	return complexI;
}
Mat calculateIDFT(Mat complexI)
{
	shift(complexI);
	Mat work;

	idft(complexI, work);
	//  dft(complex, work, DFT_INVERSE + DFT_SCALE);
	Mat planes1[] = { Mat::zeros(complexI.size(), CV_32F), Mat::zeros(complexI.size(), CV_32F) };
	split(work, planes1);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

	magnitude(planes1[0], planes1[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	normalize(work, work, 0, 1, NORM_MINMAX);
	//cvtColor(work, work, CV_GRAY2BGR);
	// imshow("result", work);

	return work;
}

Mat getFilterKernel(Mat FilterMask)
{

	Mat planes2[] = { Mat_<float>(FilterMask), Mat::zeros(FilterMask.size(), CV_32F) };
	Mat kernel_spec;
	merge(planes2, 2, kernel_spec);

	return kernel_spec;
}


Mat generateLowPassFilterMask(Mat complexI, int radius)
{
	//Create a circular mask -- Low Pass
	//Mat img = Mat::zeros(complexI.rows, complexI.cols, CV_32F);
	Mat lowPassFilterMask=Mat::zeros(complexI.rows, complexI.cols, CV_32F);
	circle(lowPassFilterMask, Point(lowPassFilterMask.cols / 2, lowPassFilterMask.rows / 2), radius, Scalar(1), -1, 8);

	imshow("Low Pass Filter Mask", lowPassFilterMask);
	return lowPassFilterMask;
}


Mat generatehighPassFilterMask(Mat complexI, int radius)
{
	//Create a circular mask -- High Pass
	Mat highPassFilterMask=Mat::ones(complexI.rows, complexI.cols, CV_32F);
	circle(highPassFilterMask, Point(highPassFilterMask.cols / 2, highPassFilterMask.rows / 2), radius, Scalar(0), -1, 8);

	imshow("High Pass Filter Mask", highPassFilterMask);
	return highPassFilterMask;
}


Mat generateBandPassFilterMask(Mat complexI, int radius, int radius2)
{
	//Create a circular mask -- Band Pass
	Mat bandPassFilterMask = generateLowPassFilterMask(complexI, radius2) - generateLowPassFilterMask(complexI, radius);

	imshow("Band Pass Filter Mask", bandPassFilterMask);
	return bandPassFilterMask;

}

Mat gaussianMask(Mat complexI,double sigma,int ksize)
{

	int x = complexI.cols / 2;
	int y = complexI.rows / 2;
	// call openCV gaussian kernel generator
	//double sigma = -1;
	Size mask_size = complexI.size();
	
	Mat kernelX = getGaussianKernel(ksize, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(ksize, sigma, CV_32F);
	// create 2d gaus
	Mat kernel = kernelX * kernelY.t();
	// create empty mask
	Mat mask = Mat::zeros(mask_size, CV_32F);
	Mat maski = Mat::zeros(mask_size, CV_32F);

	// copy kernel to mask on x,y
	Mat pos(mask, Rect(x - ksize / 2, y - ksize / 2, ksize, ksize));
	kernel.copyTo(pos);

	// create mirrored mask
	Mat posi(maski, Rect((mask_size.width - x) - ksize / 2, (mask_size.height - y) - ksize / 2, ksize, ksize));
	kernel.copyTo(posi);
	// add mirrored to mask
	add(mask, maski, mask);

	// transform mask to range 0..1
	
	normalize(mask, mask, 0, 1, NORM_MINMAX);

		imshow("Gaussian Filter Mask", mask);
		return mask;

}


