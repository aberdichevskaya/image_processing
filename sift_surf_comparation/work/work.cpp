#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ctime>
#include <iostream>
#include <vector>
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

std::vector< cv::KeyPoint> get_points(const json& json_file) {
    std::vector< cv::KeyPoint> points;
    json json_points = json_file["quad"];
    for (ptrdiff_t i_point = 0; i_point < json_points.size(); i_point += 1) {
        points.push_back({ json_points[i_point][0], json_points[i_point][1], 0.1});
    }
    return points;
}

std::vector<cv::KeyPoint> SIFT(const cv::Mat& img, const uint32_t idx) {
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);
    cv::Mat output;
    cv::drawKeypoints(img, keypoints, output);
    cv::imwrite("sift_result" + std::to_string(idx) + ".png", output);
    detector.release();
    return keypoints;
}


std::vector<cv::KeyPoint> SURF(const cv::Mat& img, const uint32_t idx) {
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints);
    cv::imwrite("surf_result" + std::to_string(idx) + ".png", img_keypoints);
    detector.release();
    return keypoints;
}

double dist(cv::KeyPoint a, cv::KeyPoint b) {
    double x1 = a.pt.x;
    double y1 = a.pt.y;
    double x2 = b.pt.x;
    double y2 = b.pt.y;
    double dist2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    return sqrt(dist2);
}

double asesment(const std::vector<cv::KeyPoint>& etalon, const std::vector<cv::KeyPoint>& found_pts) {
    double threshold = 5.0 - 1e-9;
    double TP = 0.0;
    double FP = found_pts.size();
    double FN = etalon.size();
    for (ptrdiff_t i = 0; i < found_pts.size(); i += 1) {
        for (ptrdiff_t j = 0; j < etalon.size(); j += 1) {
            if (dist(found_pts[i], etalon[j]) < threshold) {
                TP += 1;
                FN -= 1;
                FP -= 1;
                break;
            }
        }
    }
    double pre = TP / (TP + FP);
    double rec = TP / (TP + FN);
    double F1 = 2.0 * pre * rec / (pre + rec);
    return F1;
}

int main() {
    uint32_t cnt = 40;
    std::vector<cv::Mat> img(cnt);
    std::vector<std::vector<cv::KeyPoint>> etalon(cnt);
    double sift_f1 = 0.0;
    double surf_f1 = 0.0;
    double sift_time = 0.0;
    double surf_time = 0.0;
    for (int i = 0; i < cnt; i += 1) {
        json json_file;
        std::ifstream ifstream("../data/jsons/etalon (" + std::to_string(i) + ").json");
        ifstream >> json_file;
        etalon[i] = get_points(json_file);
        img[i] = cv::imread("../data/imgs/img (" + std::to_string(i) + ").jpg", 0);
        double time = clock();
        std::vector<cv::KeyPoint> sift_kp = SIFT(img[i], i);
        sift_time += (clock() - time);
        time = clock();
        std::vector<cv::KeyPoint> surf_kp = SURF(img[i], i);
        surf_time += (clock() - time);
        sift_f1 += asesment(etalon[i], sift_kp);
        surf_f1 += asesment(etalon[i], surf_kp);
    }
    sift_f1 /= cnt;
    surf_f1 /= cnt;
    sift_time /= cnt;
    surf_time /= cnt;
    std::cout << "SIFT F1-score:\n" << sift_f1 << "\n" << "SURF F1-score:\n" << surf_f1 << "\n\n";
    std::cout << "SIFT time:\n" << sift_time << "\n" << "SURF time:\n" << surf_time;
    return 0;
}


