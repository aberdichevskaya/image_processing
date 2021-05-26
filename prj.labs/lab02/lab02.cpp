#include <opencv2/opencv.hpp>

int main() {
	cv::Mat initial_img = cv::imread("../data/data_cross_0256x0256.png");
	cv::imwrite("cross_0256x0256_025.jpg", initial_img, { cv::IMWRITE_JPEG_QUALITY, 25 });
	cv::Mat low_quality_img = cv::imread("cross_0256x0256_025.jpg");

	cv::Mat chanels_initial[3];
	cv::split(initial_img, chanels_initial);
	int heigh = initial_img.rows;
	int width = initial_img.cols;
	cv::Mat res1[3];
	res1[0] = cv::Mat::zeros(2 * heigh, 2 * width, CV_8UC1);
	res1[1] = cv::Mat::zeros(2 * heigh, 2 * width, CV_8UC1);
	res1[2] = cv::Mat::zeros(2 * heigh, 2 * width, CV_8UC1);

	chanels_initial[0].copyTo(res1[0](cv::Rect(0, 0, width, heigh)));
	chanels_initial[0].copyTo(res1[0](cv::Rect(width, heigh, width, heigh)));

	chanels_initial[1].copyTo(res1[1](cv::Rect(0, 0, width, heigh)));
	chanels_initial[1].copyTo(res1[1](cv::Rect(0, heigh, width, heigh)));

	chanels_initial[2].copyTo(res1[2](cv::Rect(0, 0, width, heigh)));
	chanels_initial[2].copyTo(res1[2](cv::Rect(width, 0, width, heigh)));


	cv::Mat result1;
	cv::merge(res1, 3, result1);
	cv::imwrite("cross_0256x0256_png_channels.png", result1);


	cv::Mat chanels_low_quality[3];
	cv::split(low_quality_img, chanels_low_quality);
	heigh = low_quality_img.rows;
	width = low_quality_img.cols;
	cv::Mat res2[3];
	res2[0] = cv::Mat::zeros(2 * heigh, 2 * width, CV_8UC1);
	res2[1] = cv::Mat::zeros(2 * heigh, 2 * width, CV_8UC1);
	res2[2] = cv::Mat::zeros(2 * heigh, 2 * width, CV_8UC1);

	chanels_low_quality[0].copyTo(res2[0](cv::Rect(0, 0, width, heigh)));
	chanels_low_quality[0].copyTo(res2[0](cv::Rect(width, heigh, width, heigh)));

	chanels_low_quality[1].copyTo(res2[1](cv::Rect(0, 0, width, heigh)));
	chanels_low_quality[1].copyTo(res2[1](cv::Rect(0, heigh, width, heigh)));

	chanels_low_quality[2].copyTo(res2[2](cv::Rect(0, 0, width, heigh)));
	chanels_low_quality[2].copyTo(res2[2](cv::Rect(width, 0, width, heigh)));


	cv::Mat result2;
	cv::merge(res2, 3, result2);
	cv::imwrite("cross_0256x0256_jpg_channels.png", result2);
	
	int histSize = 256;
	float range[] = { 0, 256 }; 
	const float* histRange = { range };
	cv::Mat b_hist_initial, g_hist_initial, r_hist_initial;
	calcHist(&chanels_initial[0], 1, 0, cv::Mat(), b_hist_initial, 1, &histSize, &histRange);
	calcHist(&chanels_initial[1], 1, 0, cv::Mat(), g_hist_initial, 1, &histSize, &histRange);
	calcHist(&chanels_initial[2], 1, 0, cv::Mat(), r_hist_initial, 1, &histSize, &histRange);

	cv::Mat b_hist_low_quality, g_hist_low_quality, r_hist_low_quality;
	calcHist(&chanels_low_quality[0], 1, 0, cv::Mat(), b_hist_low_quality, 1, &histSize, &histRange);
	calcHist(&chanels_low_quality[1], 1, 0, cv::Mat(), g_hist_low_quality, 1, &histSize, &histRange);
	calcHist(&chanels_low_quality[2], 1, 0, cv::Mat(), r_hist_low_quality, 1, &histSize, &histRange);

	int hist_w = 256, hist_h = 200;
	cv::Mat histImage (2*hist_h, 3*hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

	normalize(b_hist_initial, b_hist_initial, 0, histImage.rows/2, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist_initial, g_hist_initial, 0, histImage.rows/2, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist_initial, r_hist_initial, 0, histImage.rows/2, cv::NORM_MINMAX, -1, cv::Mat());

	normalize(b_hist_low_quality, b_hist_low_quality, 0, histImage.rows/2, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist_low_quality, g_hist_low_quality, 0, histImage.rows/2, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist_low_quality, r_hist_low_quality, 0, histImage.rows/2, cv::NORM_MINMAX, -1, cv::Mat());

	for (ptrdiff_t i = 0; i < histSize; i = i+1) {
		line(histImage(cv::Rect(0, 0, hist_w, hist_h)),
			cv::Point(i, hist_h - cvRound(b_hist_initial.at<float>(i))), cv::Point(i, hist_h),
			cv::Scalar(255, 0, 0));
		line(histImage(cv::Rect(hist_w, 0, hist_w, hist_h)),
			cv::Point(i, hist_h - cvRound(g_hist_initial.at<float>(i))), cv::Point(i, hist_h),
			cv::Scalar(0, 255, 0));
		line(histImage(cv::Rect(2*hist_w, 0, hist_w, hist_h)),
			cv::Point(i, hist_h - cvRound(r_hist_initial.at<float>(i))), cv::Point(i, hist_h),
			cv::Scalar(0, 0, 255));

		line(histImage(cv::Rect(0, hist_h, hist_w, hist_h)),
			cv::Point(i, hist_h - cvRound(b_hist_low_quality.at<float>(i))), cv::Point(i, hist_h),
			cv::Scalar(255, 0, 0));
		line(histImage(cv::Rect(hist_w, hist_h, hist_w, hist_h)),
			cv::Point(i, hist_h - cvRound(g_hist_low_quality.at<float>(i))), cv::Point(i, hist_h),
			cv::Scalar(0, 255, 0));
		line(histImage(cv::Rect(2 * hist_w, hist_h, hist_w, hist_h)),
			cv::Point(i, hist_h - cvRound(r_hist_low_quality.at<float>(i))), cv::Point(i, hist_h),
			cv::Scalar(0, 0, 255));
	}
	
	cv::imwrite("cross_0256x0256_hists.png", histImage);

	return 0;
}
