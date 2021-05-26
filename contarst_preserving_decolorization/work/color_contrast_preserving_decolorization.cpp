#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>



inline double scalar_mul(const cv::Vec3d& A, const cv::Vec3d& B) {
	double res = 0;
	for (ptrdiff_t i = 0; i < 3; i = i + 1) {
		res += A[i] * B[i];
	}
	return res;
}

inline double delta(const cv::Mat& lab_img, const int i1, const int j1, const int i2, const int j2) {
	double L1 = lab_img.at<cv::Vec3b>(i1, j1)[0];
	double a1 = lab_img.at<cv::Vec3b>(i1, j1)[1];
	double b1 = lab_img.at<cv::Vec3b>(i1, j1)[2];

	double L2 = lab_img.at<cv::Vec3b>(i2, j2)[0];
	double a2 = lab_img.at<cv::Vec3b>(i2, j2)[1];
	double b2 = lab_img.at<cv::Vec3b>(i2, j2)[2];

	double res2 = (L1 - L2) * (L1 - L2) + (a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2);
	return sqrt(res2);
}


inline cv::Vec3d lambda(const cv::Mat& rgb_img, const int i1, const int j1, const int i2, const int j2) {
	double r1 = rgb_img.at<cv::Vec3b>(i1, j1)[0];
	double g1 = rgb_img.at<cv::Vec3b>(i1, j1)[1];
	double b1 = rgb_img.at<cv::Vec3b>(i1, j1)[2];

	double r2 = rgb_img.at<cv::Vec3b>(i2, j2)[0];
	double g2 = rgb_img.at<cv::Vec3b>(i2, j2)[1];
	double b2 = rgb_img.at<cv::Vec3b>(i2, j2)[2];

	cv::Vec3d res({ r1 - r2, g1 - g2, b1 - b2 });
	return res;
}

inline double L(const cv::Mat& lab_img, const int i, const int j) {
	return lab_img.at<cv::Vec3b>(i, j)[0];
}

int8_t gamma(const cv::Mat& rgb_img, const int i1, const int j1, const int i2, const int j2, const cv::Vec3d& w) {
	double eps = 1e-9;

	cv::Vec3d la = lambda(rgb_img, i1, j1, i2, j2);
	double res = scalar_mul(w, la);
	int8_t ans = 1;
	if (fabs(res) < eps)
		ans = 0;
	else if (res < -eps)
		ans = -1;
	return ans;
}

std::vector<std::vector<float>> Invertible(std::vector<std::vector<float>>& A) {
	const int n = A.size();
	std::vector<std::vector<float>> I(n, std::vector<float>(3, 0));
	for (int i = 0; i < n; ++i) {
		I[i][i] = 1.0;
	}
	for (int i = 0; i < n; ++i) {
		double t = A[i][i];
		for (int j = 0; j < n; ++j) {
			A[i][j] /= t;
			I[i][j] /= t;
		}
		for (int j = i + 1; j < n; ++j) {
			double t = A[j][i];
			for (int k = 0; k < n; ++k) {
				A[j][k] -= A[i][k] * t;
				I[j][k] -= I[i][k] * t;
			}
		}
	}
	for (int i = n - 1; i >= 0; --i) {
		for (int j = i - 1; j >= 0; --j) {
			double t = A[j][i];
			for (int k = n - 1; k >= 0; --k) {
				A[j][k] -= A[i][k] * t;
				I[j][k] -= I[i][k] * t;
			}
		}
	}
	return I;
}

cv::Vec3d mul_matrix_vec(const cv::Mat& A, const cv::Vec3d& b) {
	cv::Vec3d w;
	for (ptrdiff_t i = 0; i < A.rows; i = i + 1) {
		double wi = 0.0;
		for (ptrdiff_t j = 0; j < A.cols; j = j + 1) {
			wi += A.at<float>(i, j) * b[j];
		}
		w[i] = wi;
	}
	return w;
}

double A_E1(const cv::Mat& img, const uint8_t i, const uint8_t j) {
	cv::Mat rgb_img;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
	double res = 0.0;
	for (ptrdiff_t x = 0; x < rgb_img.rows; x = x + 1) {
		for (ptrdiff_t y = 0; y < rgb_img.cols; y = y + 1) {
			cv::Vec3b I = rgb_img.at<cv::Vec3b>(x, y);
			res += I[i] * I[j];
		}
	}
	return res;
}

double A_E2(const cv::Mat& img, const uint8_t i, const uint8_t j) {
	double res = 0.0;
	cv::Mat rgb_img;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
	for (ptrdiff_t x = 0; x < img.rows; x = x + 1) {
		for (ptrdiff_t y = 0; y < img.cols; y = y + 1) {
			if (x > 0) {
				cv::Vec3d la = lambda(rgb_img, x, y, x - 1, y);
				res += la[i] * la[j];
			}
			if (y > 0) {
				cv::Vec3d la = lambda(rgb_img, x, y, x, y - 1);
				res += la[i] * la[j];
			}
			if (x < img.rows - 1) {
				cv::Vec3d la = lambda(rgb_img, x, y, x + 1, y);
				res += la[i] * la[j];
			}
			if (y < img.cols - 1) {
				cv::Vec3d la = lambda(rgb_img, x, y, x, y + 1);
				res += la[i] * la[j];
			}
			if (x > 0 && y > 0) {
				cv::Vec3d la = lambda(rgb_img, x, y, x - 1, y - 1);
				res += la[i] * la[j];
			}
			if (x < img.rows - 1  && y < img.cols - 1) {
				cv::Vec3d la = lambda(rgb_img, x, y, x + 1, y + 1);
				res += la[i] * la[j];
			}
			if (x > 0 && y < img.cols - 1) {
				cv::Vec3d la = lambda(rgb_img, x, y, x - 1, y + 1);
				res += la[i] * la[j];
			}
			if (x < img.rows - 1 && y > 0) {
				cv::Vec3d la = lambda(rgb_img, x, y, x + 1, y - 1);
				res += la[i] * la[j];
			}
		}
	}
	return res;
}

double get_Aij(const cv::Mat& img, const double la1, const uint8_t i, const uint8_t j) {
	double e1 = A_E1(img, i, j);
	double e2 = A_E2(img, i, j);
	return la1 * e1 + e2;
}

std::vector<std::vector<float>> get_A(const cv::Mat& img, const double la1) {
	std::vector<std::vector<float>> A(3, std::vector<float>(3));
	for (ptrdiff_t i = 0; i < 3; i = i + 1) {
		for (ptrdiff_t j = 0; j < 3; j = j + 1) {
			A[i][j] = get_Aij(img, la1, i, j);
		}
	}
	return A;
}

double b_E1(const cv::Mat& img, const uint8_t j) {
	cv::Mat rgb_img, lab_img;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
	cv::cvtColor(img, lab_img, cv::COLOR_BGR2Lab);
	double res = 0.0;
	for (ptrdiff_t x = 0; x < rgb_img.rows; x = x + 1) {
		for (ptrdiff_t y = 0; y < rgb_img.cols; y = y + 1) {
			cv::Vec3b I = rgb_img.at<cv::Vec3b>(x, y);
			double Lp = L(lab_img, x, y);
			res += Lp * I[j];
		}
	}
	return res;
}


double b_E2(const cv::Mat& img, const uint8_t j, const cv::Vec3d& w) {
	double res = 0.0;
	cv::Mat lab_img, rgb_img;
	cv::cvtColor(img, lab_img, cv::COLOR_BGR2Lab);
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
	for (ptrdiff_t x = 0; x < img.rows; x = x + 1) {
		for (ptrdiff_t y = 0; y < img.cols; y = y + 1) {
			double cur_res = 0.0;
			if (x > 0) {
				uint8_t g = gamma(rgb_img, x, y, x - 1, y, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x - 1, y);
				double d = delta(lab_img, x, y, x - 1, y);
				cur_res += g * la[j] * d;
			}
			if (y > 0) {
				uint8_t g = gamma(rgb_img, x, y, x, y - 1, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x, y - 1);
				double d = delta(lab_img, x, y, x, y - 1);
				cur_res += g * la[j] * d;
			}
			if (x < img.rows - 1) {
				uint8_t g = gamma(rgb_img, x, y, x + 1, y, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x + 1, y);
				double d = delta(lab_img, x, y, x + 1, y);
				cur_res += g * la[j] * d;
			}
			if (y < img.cols - 1) {
				uint8_t g = gamma(rgb_img, x, y, x, y + 1, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x, y + 1);
				double d = delta(lab_img, x, y, x, y + 1);
				cur_res += g * la[j] * d;
			}
			if (x > 0 && y > 0) {
				uint8_t g = gamma(rgb_img, x, y, x - 1, y - 1, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x - 1, y - 1);
				double d = delta(lab_img, x, y, x - 1, y - 1);
				cur_res += g * la[j] * d;
			}
			if (x < img.rows - 1 && y < img.cols - 1) {
				uint8_t g = gamma(rgb_img, x, y, x + 1, y + 1, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x + 1, y + 1);
				double d = delta(lab_img, x, y, x + 1, y + 1);
				cur_res += g * la[j] * d;
			}
			if (x > 0 && y < img.cols - 1) {
				uint8_t g = gamma(rgb_img, x, y, x - 1, y + 1, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x - 1, y + 1);
				double d = delta(lab_img, x, y, x - 1, y + 1);
				cur_res += g * la[j] * d;
			}
			if (x < img.rows - 1 && y > 0) {
				uint8_t g = gamma(rgb_img, x, y, x + 1, y - 1, w);
				cv::Vec3d la = lambda(rgb_img, x, y, x + 1, y - 1);
				double d = delta(lab_img, x, y, x + 1, y - 1);
				cur_res += g * la[j] * d;
			}
			res += cur_res;
		}
	}
	return res;
}


double get_bj(const cv::Mat& img, const double la1, const cv::Vec3d& w, const uint8_t j) {
	double e1 = b_E1(img, j);
	double e2 = b_E2(img, j, w);
	return la1 * e1 + e2;
}

cv::Vec3d get_b(const cv::Mat& img, const double la1, const cv::Vec3d& w) {
	cv::Vec3d b;
	for (ptrdiff_t j = 0; j < 3; j = j + 1) {
		b[j] = get_bj(img, la1, w, j);
	}
	return b;
}

cv::Vec3d count_w(cv::Mat img, const uint32_t num_iterations, const double la1) {
	cv::Vec3d w = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
	std::vector<std::vector<float>> A = get_A(img, la1);

	std::vector<std::vector<float>> inv_A = Invertible(A);

	cv::Mat inv(3, 3, CV_32FC1);
	for (ptrdiff_t i = 0; i < 3; i = i + 1) {
		for (ptrdiff_t j = 0; j < 3; j = j + 1) {
			inv.at<float>(i, j) = inv_A[i][j];
		}
	}

	for (ptrdiff_t k = 0; k < num_iterations; k = k + 1) {
		cv::Vec3d b = get_b(img, la1, w);
		cv::Vec3d wk = mul_matrix_vec(inv, b);
		double norm = wk[0] + wk[1] + wk[2];
		wk[0] /= norm;
		wk[1] /= norm;
		wk[2] /= norm;
		swap(w, wk);
	}

	return w;
}


cv::Mat prosecc_img(cv::Mat img, const uint32_t num_iterations, double la1) {
	cv::Mat rgb_img;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
	cv::Vec3d w = count_w(img, num_iterations, la1);
	cv::Mat res(rgb_img.rows, rgb_img.cols, CV_8UC1);
	for (ptrdiff_t i = 0; i < rgb_img.rows; i = i + 1) {
		for (ptrdiff_t j = 0; j < rgb_img.cols; j = j + 1) {
			cv::Vec3b I = rgb_img.at<cv::Vec3b>(i, j);
			double val = scalar_mul(w, I);
			res.at<uint8_t>(i, j) = static_cast<uint8_t>(val);
		}
	}
	return res;
}


double CCPR(const std::vector<cv::Mat>& orig, const std::vector<cv::Mat>& res, uint32_t tau) {
	double eps = 1e-9;
	uint32_t cnt = orig.size();
	double ccpr = 0.0;

	for (ptrdiff_t num = 0; num < cnt; num += 1) {
		cv::Mat lab_orig;
		cv::cvtColor(orig[num], lab_orig, cv::COLOR_RGB2Lab);
		uint32_t W = 0;
		uint32_t G = 0;
		for (ptrdiff_t x = 0; x < lab_orig.rows; x += 1) {
			for (ptrdiff_t y = 0; y < lab_orig.cols; y += 1) {
				if (x > 0) {
					double d = delta(lab_orig, x, y, x - 1, y);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x - 1, y)) >= tau) {
							G++;
						}
					}			
				}
				if (y > 0) {
					double d = delta(lab_orig, x, y, x, y - 1);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x, y - 1)) >= tau) {
							G++;
						}
					}
				}
				if (x < lab_orig.rows - 1) {
					double d = delta(lab_orig, x, y, x + 1, y);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x + 1, y)) >= tau) {
							G++;
						}
					}
				}
				if (y < lab_orig.cols - 1) {
					double d = delta(lab_orig, x, y, x, y + 1);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x, y + 1)) >= tau) {
							G++;
						}
					}
				}
				if (x > 0 && y > 0) {
					double d = delta(lab_orig, x, y, x - 1, y - 1);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x - 1, y - 1)) >= tau) {
							G++;
						}
					}
				}
				if (x < lab_orig.rows - 1 && y < lab_orig.cols - 1) {
					double d = delta(lab_orig, x, y, x + 1, y + 1);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x + 1, y + 1)) >= tau) {
							G++;
						}
					}
				}
				if (x > 0 && y < lab_orig.cols - 1) {
					double d = delta(lab_orig, x, y, x - 1, y + 1);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x - 1, y + 1)) >= tau) {
							G++;
						}
					}
				}
				if (x < lab_orig.rows - 1 && y > 0) {
					double d = delta(lab_orig, x, y, x + 1, y - 1);
					if (d + eps >= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x + 1, y - 1)) >= tau) {
							G++;
						}
					}
				}
			}
		}
		if (W) {
			ccpr += static_cast<double>(G) / static_cast<double>(W);
		}
	}
	ccpr /= cnt;
	std::cout << "cnt " << cnt << "\n ccpr " << ccpr << "\n";
	return ccpr;
}

double CCFR(const std::vector<cv::Mat>& orig, const std::vector<cv::Mat>& res, uint32_t tau) {
	double eps = 1e-9;
	uint32_t cnt = orig.size();
	double ccfr = 0.0;

	for (ptrdiff_t num = 0; num < cnt; num += 1) {
		cv::Mat lab_orig;
		cv::cvtColor(orig[num], lab_orig, cv::COLOR_RGB2Lab);
		uint32_t W = 0;
		uint32_t G = 0;
		for (ptrdiff_t x = 0; x < lab_orig.rows; x += 1) {
			for (ptrdiff_t y = 0; y < lab_orig.cols; y += 1) {
				if (x > 0) {
					double d = delta(lab_orig, x, y, x - 1, y);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x - 1, y)) >= tau) {
							G++;
						}
					}
				}
				if (y > 0) {
					double d = delta(lab_orig, x, y, x, y - 1);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x, y - 1)) >= tau) {
							G++;
						}
					}
				}
				if (x < lab_orig.rows - 1) {
					double d = delta(lab_orig, x, y, x + 1, y);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x + 1, y)) >= tau) {
							G++;
						}
					}
				}
				if (y < lab_orig.cols - 1) {
					double d = delta(lab_orig, x, y, x, y + 1);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x, y + 1)) >= tau) {
							G++;
						}
					}
				}
				if (x > 0 && y > 0) {
					double d = delta(lab_orig, x, y, x - 1, y - 1);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x - 1, y - 1)) >= tau) {
							G++;
						}
					}
				}
				if (x < lab_orig.rows - 1 && y < lab_orig.cols - 1) {
					double d = delta(lab_orig, x, y, x + 1, y + 1);
					if (d + eps <= tau) {
						W++;
					if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x + 1, y + 1)) >= tau) {
							G++;
						}
					}
				}
				if (x > 0 && y < lab_orig.cols - 1) {
					double d = delta(lab_orig, x, y, x - 1, y + 1);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x - 1, y + 1)) >= tau) {
							G++;
						}
					}
				}
				if (x < lab_orig.rows - 1 && y > 0) {
					double d = delta(lab_orig, x, y, x + 1, y - 1);
					if (d + eps <= tau) {
						W++;
						if (abs(res[num].at<uint8_t>(x, y) - res[num].at<uint8_t>(x + 1, y - 1)) >= tau) {
							G++;
						}
					}
				}
			}
		}
		if(W)
			ccfr += (1.0 - static_cast<double>(G) / static_cast<double>(W));
	}
	ccfr /= cnt;
	
	return ccfr;
}


double E_score(const double ccpr, double ccfr) {
	double num = 2.0 * ccpr * ccfr;
	double den = ccpr + ccfr;
	double E = num / den;
	return E;
}


void vary_la1(std::vector<cv::Mat>& img) {
	double eps = 1e-9;
	uint32_t num_iterations = 3;

	uint32_t max_tau = 40;

	std::ofstream out;        
	out.open("vary_la1.txt");

	for (double la1 = 1.0 / 1048576.0; la1 < 1.0/1024.0 + eps; la1 *= 2) {
		std::vector<cv::Mat> res(img.size());
		for (ptrdiff_t i = 0; i < img.size(); i += 1) {
			res[i] = prosecc_img(img[i], num_iterations, la1);
		}
		double CCPR_by_la1 = CCPR(img, res, max_tau);
		double CCFR_by_la1 = CCFR(img, res, max_tau);
		double E_by_la1 = E_score(CCPR_by_la1, CCFR_by_la1);
		out << la1 << " " << CCPR_by_la1 << " " << CCFR_by_la1 << " " << E_by_la1 << "\n";
	}

	out.close();
}


void vary_num_iterations(std::vector<cv::Mat>& img) {
	std::cout << img.size() << "\n";
	double eps = 1e-9;
	double la1 = 1.0 / 1024.0;

	uint32_t max_tau = 40;

	std::ofstream out;
	out.open("vary_num_iterations.txt");

	std::vector<std::vector<cv::Mat>> num_res(31, std::vector<cv::Mat>(img.size()));
	
	for (ptrdiff_t i = 0; i < img.size(); i += 1) {
		cv::Mat rgb_img;
		cv::cvtColor(img[i], rgb_img, cv::COLOR_BGR2RGB);
		cv::Vec3d w = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
		std::vector<std::vector<float>> A = get_A(img[i], la1);
		std::vector<std::vector<float>> inv_A = Invertible(A);

		cv::Mat inv(3, 3, CV_32FC1);
		for (ptrdiff_t i = 0; i < 3; i = i + 1) {
			for (ptrdiff_t j = 0; j < 3; j = j + 1) {
				inv.at<float>(i, j) = inv_A[i][j];
			}
		}

		for (uint32_t num = 0; num <= 30; num += 1) {
			cv::Vec3d b = get_b(img[i], la1, w);
			cv::Vec3d wk = mul_matrix_vec(inv, b);
			double norm = wk[0] + wk[1] + wk[2];
			wk[0] /= norm;
			wk[1] /= norm;
			wk[2] /= norm;
			swap(w, wk);

			num_res[num][i] = cv::Mat(rgb_img.rows, rgb_img.cols, CV_8UC1);
			for (ptrdiff_t i_row = 0; i_row < rgb_img.rows; i_row = i_row + 1) {
				for (ptrdiff_t j_col = 0; j_col < rgb_img.cols; j_col = j_col + 1) {
					cv::Vec3b I = rgb_img.at<cv::Vec3b>(i_row, j_col);
					double val = scalar_mul(w, I);
					num_res[num][i].at<uint8_t>(i_row, j_col) = static_cast<uint8_t>(val);
				}
			}
		}


	}
		
	for(ptrdiff_t num = 0; num <= 30; num += 1) {
		double CCPR_by_num = CCPR(img, num_res[num], max_tau);
		double CCFR_by_num = CCFR(img, num_res[num], max_tau);
		double E_by_num = E_score(CCPR_by_num, CCFR_by_num);
		out << num << " " << CCPR_by_num << " " << CCFR_by_num << " " << E_by_num << "\n";
	}

	out.close();
}

cv::Mat join(const cv::Mat& orig, const cv::Mat& my_res) {
	uint32_t heigh = orig.rows;
	uint32_t width = orig.cols;
	cv::Mat my_res_3c;
	cv::cvtColor(my_res, my_res_3c, cv::COLOR_GRAY2RGB);
	cv::Mat result(heigh, 2*width, CV_8UC3);
	orig.copyTo(result(cv::Rect(0, 0, width, heigh)));
	my_res_3c.copyTo(result(cv::Rect(width, 0, width, heigh)));
	return result;
}

int main() {
	try {
		
		uint32_t cnt = 12;
		uint32_t num_iterations = 3;
		double la1 = 1.0 / 1024.0;

		std::vector<cv::Mat> img(cnt);
		std::vector<cv::Mat> res(cnt);
		for (ptrdiff_t i = 0; i < cnt; i += 1) {
			img[i] = cv::imread("../coursework/data/COLOR250/images/" + std::to_string(i + 1) + ".png");
			res[i] = prosecc_img(img[i], num_iterations, la1);
		}

	
		//uint32_t cnt = 250;
		//for (ptrdiff_t i = 0; i < cnt; i += 1) {
		//	img[i] = cv::imread("../coursework/data/img" + std::to_string(i + 1) + ".png");
		//	res[i] = prosecc_img(img[i], num_iterations, la1);
		//	cv::imwrite("result_grayscale" + std::to_string(i + 1) + ".png", join(img[i], res[i]));
		//}

		//vary_num_iterations(img);
		//vary_la1(img);

	}
	catch (cv::Exception e) {
		std::cerr << e.what();
	}

	return 0;
}

