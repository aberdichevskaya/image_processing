#include <opencv2/opencv.hpp>
#include <cmath>

uint32_t brightness_conversion(const uint32_t color) {
	uint32_t rez = static_cast<uint32_t>(255 * 10 * fabs(sin(static_cast<double>(color) / (255.0 * 3))) + 0.5) % 256;
	return rez;
}

int main() {
	cv::Mat initial_img = cv::imread("../data/data_cross_0256x0256.png");
	cv::imwrite("lab03_rgb.png", initial_img);

	cv::Mat greyscale_img;
	cv::cvtColor(initial_img, greyscale_img, cv::COLOR_BGR2GRAY);
	cv::imwrite("lab03_gre.png", greyscale_img);

	cv::Mat look_up_table(1, 256, CV_8UC1);
	for (ptrdiff_t i = 0; i < 256; i = i + 1) {
		look_up_table.at<uint8_t>(0, i) = brightness_conversion(i);
	}
	
	cv::Mat result_for_initial, result_for_greyscale;
	cv::LUT(initial_img, look_up_table, result_for_initial);
	cv::LUT(greyscale_img, look_up_table, result_for_greyscale);

	cv::imwrite("lab03_gre_res.png", result_for_greyscale);
	cv::imwrite("lab03_rgb_res.png", result_for_initial);

	cv::Mat funktion_visualization(512, 512, CV_8UC1, 255);
	for (ptrdiff_t i = 0; i < 512; i = i + 2) {
		funktion_visualization.at<uint8_t>(512 - 2*look_up_table.at<uint8_t>(0, i / 2) - 1, i ) = 0;
	}
	cv::imwrite("lab03_viz_func.png", funktion_visualization);
	cv::imshow("   ", funktion_visualization);
	cv::imshow("lab03_gre_res.png", result_for_greyscale);
	cv::imshow("lab03_rgb_res.png", result_for_initial);
	cv::waitKey(0);

	return 0;
}