#include <opencv2/opencv.hpp>
#include <ctime>
#include <iostream>

int main() {
	uint32_t heigh = 60;
	uint32_t width = 768;
	cv::Mat i1(heigh, width, CV_8UC1);
	for (ptrdiff_t i(0); i < heigh; i += 1) {
		for (ptrdiff_t j(0); j < width; j += 1) {
			uint8_t color = j / 3;
			i1.at<uint8_t>(i, j) = color;
		}
	}

	float gamma = 2.3f;

	float start_g1 = clock();
	cv::Mat g1;
	i1.convertTo(g1, CV_32FC1, 1.0f / 255.0f);
	cv::pow(g1, gamma, g1);
	g1.convertTo(g1, CV_8UC1, 255.0f, 0.5);
	float time_g1 = clock() - start_g1;
	std::cout << "Time for G1 = " << time_g1 << std::endl;

	float start_g2 = clock();
	cv::Mat g2;
	i1.copyTo(g2);
	for (ptrdiff_t i(0); i < heigh; i += 1) {
		for (ptrdiff_t j(0); j < width; j += 1) {
			g2.at<uint8_t>(i, j) = static_cast<uint8_t>(cv::pow(static_cast<float>(g2.at<uint8_t>(i, j)) / 255.0f, gamma)*255.0f + 0.5f);
		}
	}
	float time_g2 = clock() - start_g2;
	std::cout << "Time for G2 = " << time_g2 << std::endl;
	

	cv::Mat result(3*heigh, width, CV_8UC1);
	i1.copyTo(result(cv::Rect(0, 0, width, heigh)));
	g1.copyTo(result(cv::Rect(0, heigh, width, heigh)));
	g2.copyTo(result(cv::Rect(0, 2 * heigh, width, heigh)));
	cv::imwrite("lab01.png", result);
	cv::imshow("RESULT", result);
	cv::waitKey(0);

	return 0;
}

