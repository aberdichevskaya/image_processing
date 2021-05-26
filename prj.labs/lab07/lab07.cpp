#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


double euclidean_distance(cv::Point a, cv::Point b) {
	return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double points_estimation(const std::vector<cv::Point>& found, const std::vector<cv::Point>& etalon) {
	double score = 0.0;
	double max_len = std::max(euclidean_distance(etalon[0], etalon[2]), euclidean_distance(etalon[1], etalon[3]));
	for (ptrdiff_t i = 0; i < 4; i = i + 1) {
		score += (1.0 - euclidean_distance(found[i], etalon[i]) / max_len);
	}
	return score / 4;
}

double intersection_over_union(const cv::Mat& orig, const std::vector<cv::Point>& found,
	const std::vector<cv::Point>& etalon_pts) {
	cv::Mat etalon = cv::Mat::zeros(cv::Size(orig.cols, orig.rows), CV_8U);
	cv::Mat my_result = cv::Mat::zeros(cv::Size(orig.cols, orig.rows), CV_8U);
	cv::fillPoly(etalon, etalon_pts, 255, 150, 0);
	cv::fillPoly(my_result, found, 255, 150, 0);
	std::vector<cv::Point> FP;
	std::vector<cv::Point> FN;
	double intersection = 0.0;
	double union_ = 0.0;
	for (ptrdiff_t i = 0; i < orig.cols; i++) {
		for (ptrdiff_t j = 0; j < orig.rows; j++) {
			if (etalon.at<uchar>(j, i) == 255 || my_result.at<uchar>(j, i) == 255) {
				union_ += 1.0;
			}
			if (etalon.at<uchar>(j, i) == 255 && my_result.at<uchar>(j, i) == 255) {
				intersection += 1.0;
			}
		}
	}
	double res = intersection / union_;
	cv::polylines(orig, etalon_pts, true, cv::Scalar(255, 0, 8, 0), 2, 150, 0);
	cv::polylines(orig, found, true, cv::Scalar(0, 0, 255, 0), 2, 150, 0);
	return res;
}

void gamma_correction(cv::Mat& img) {
	float gamma = 2.7f;
	img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
	cv::pow(img, gamma, img);
	img.convertTo(img, CV_8UC3, 255.0f, 0.5);
}

void process_image(cv::Mat& orig, const std::vector<cv::Point>& etalon_pts, const size_t idx, double& point_res) {
	gamma_correction(orig);
	cv::imwrite("after_gamma.png", orig);
	cv::medianBlur(orig, orig, 45);
	cv::imwrite("after_blur.png", orig);

	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); 
	cv::Mat imgLaplacian;
	filter2D(orig, imgLaplacian, CV_32F, kernel);
	cv::Mat sharp;
	orig.convertTo(sharp, CV_32F);
	cv::Mat imgResult = sharp - imgLaplacian;
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	cv::imwrite("after_laplas.png", imgLaplacian);

	cv::Mat bw;
	cv::cvtColor(orig, bw, cv::COLOR_BGR2GRAY);
	threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::imwrite("binary.png", bw);

	cv::Mat dist;
	cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
	cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
	cv::imwrite("dist.png", dist);

	cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
	cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	cv::imwrite("dist_threshhold.png", dist);


	cv::Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	std::vector<std::vector<cv::Point> > contours;
	findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
	for (ptrdiff_t i = 0; i < contours.size(); i++) {
		drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
	}
	cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
	cv::Mat markers8u;
	markers.convertTo(markers8u, CV_8U, 10);
	cv::imwrite("markers.png", markers8u);


	cv::watershed(imgResult, markers);
	std::vector<cv::Vec3b> colors;
	for (ptrdiff_t i = 0; i < contours.size(); i++) {
		int b = cv::theRNG().uniform(0, 256);
		int g = cv::theRNG().uniform(0, 256);
		int r = cv::theRNG().uniform(0, 256);
		colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	
	cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
	for (ptrdiff_t i = 0; i < markers.rows; i++) {
		for (ptrdiff_t j = 0; j < markers.cols; j++) {
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size())) {
				dst.at<cv::Vec3b>(i, j) = colors[index - 1];
			}
		}
	}
	cv::imwrite("Final Result.png", dst);

}


int main() {
	cv::Mat scan_img = cv::imread("../data/lab05.scan.png");
	const uint32_t num_img = 5;
	cv::Mat photos[num_img];
	photos[0] = cv::imread("../data/lab05.photo1.jpg");
	photos[1] = cv::imread("../data/lab05.photo2.jpg");
	photos[2] = cv::imread("../data/lab05.photo3.jpg");
	photos[3] = cv::imread("../data/lab05.photo4.jpg");
	photos[4] = cv::imread("../data/lab05.photo5.jpg");

	std::vector<std::vector<cv::Point2i>> special_pts = { {{380,484},{2209,511},{2306,3215},{305,3232}},
									{{400,611}, {2009,609},{2031,2938} ,{379, 2918}},
									{{534,1966},{449,294}, {3175,61},{3088,2165}},
									{{612,1996},{694,488}, {3020,203},{3127,2168}},
									{{2839,450}, {2786,2169},{472,2060}, {469,454}} };



	cv::Mat edges[num_img];
	double average_score_iou = 0.0;
	double average_score_pts = 0.0;
	for (ptrdiff_t i = 0; i < num_img; i = i + 1) {
		double pts_score = 0.0;
		process_image(photos[i], special_pts[i], i, pts_score);
		average_score_pts += pts_score;
	}
	double pts_score = 0.0;
	process_image(photos[0], special_pts[0], 0, pts_score);
	std::cout << "Average score for points: " << average_score_pts / num_img << std::endl;
	std::cout << "Average intersiction over union score: " << average_score_iou / num_img;


	return 0;
}