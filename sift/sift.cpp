#include "sift.h"

mapper_cv::SIFT::SIFT(cv::Mat &image) {
    octaves_num = 5;
    octave_scale_size = 5;
    scale_space_pyr = vector<cv::Mat>(octaves_num * (3 + octave_scale_size));
    DoG_pyr = vector<cv::Mat>(octaves_num * (2 + octave_scale_size));
    sigma = 0.8;

    cv::cvtColor(image, base_img, cv::COLOR_BGR2GRAY);
    cv::resize(base_img, base_img, cv::Size(128, 128), 0, 0, cv::INTER_NEAREST);
    
    cv::cvtColor(image, base_img_color, cv::COLOR_BGR2GRAY);
    // this image will be used to display the keypoints, thus its size is x2 of the input because
    // according to the algorithm, the image is up-sampled.
    cv::resize(image, base_img_color, cv::Size(128*2, 128*2), 0, 0, cv::INTER_NEAREST);
}

void mapper_cv::SIFT::displayScaleSpace() {
    detectFeatures();
    std::cout << "end" << std::endl;
}

void mapper_cv::SIFT::detectFeatures() {
    calc_scale_space();
    std::cout << "scale_space_pyr size: " << scale_space_pyr.size() << std::endl;

    build_DoG();
    std::cout << "number of images in DOG space: " << DoG_pyr.size() << std::endl;

    std::cout << "finding keypoints" << std::endl;
    find_features();
    std::cout << "found " << keypoints.size() << " keypoints" << std::endl;
    std::cout << "drawing keypoints" << std::endl;
    displayKeypoints();
}

void mapper_cv::SIFT::displayKeypoints() {
    for (auto& kpt : keypoints) {
        cv::circle( base_img_color, kpt.pt, 2, cv::Scalar(0,255,0) );
    }
    cv::imshow("keypoints", base_img_color);
    cv::waitKey(0);
}

void mapper_cv::SIFT::prep_process_image() {
    cv::resize(base_img, base_img, cv::Size(base_img.cols * 2, base_img.rows * 2), 0, 0, cv::INTER_LINEAR);
    cv::GaussianBlur(base_img, base_img, cv::Size(3, 3), 0.5f);

    // Normalize pixel value to be within [0, 1]
    double normalize_factor = 1.0/255;
    cv::Mat src_cpy = base_img.clone();
    src_cpy.convertTo(base_img, CV_32F, normalize_factor, 0);

}

void mapper_cv::SIFT::calc_scale_space() {
    prep_process_image();

    // according to the paper, must produce s+3 in the stack of blurred images for each octave (s = octave_scale_size).
    size_t octave_size = 3 + octave_scale_size;

    // calc the Gaussian sigmas per octave space
    sigmas = vector<double>(octave_size);

    double k = pow(2., 1. / octave_scale_size); // k is a param from the paper denoting the blurring factor.
    sigmas[0] = sigma;
    for (size_t i(0); i < octave_size; ++i) {
        double sig_prev = pow(k, (double)(i - 1)) * sigma;
        double sig_total = sig_prev * k;
        sigmas[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
    }

    for (size_t octave(0); octave < octaves_num; ++octave) {
        for (size_t layer(0); layer < octave_size; ++layer) {
            cv::Mat &dst = scale_space_pyr[(octave * octave_size) + layer];
            if (octave == 0 && layer == 0) {
                dst = base_img;
            } else if (layer == 0) {
                const cv::Mat &src =
                    scale_space_pyr[((octave - 1) * octave_size) + octave_scale_size - 1];
                cv::resize(src, dst, cv::Size(src.cols / 2, src.rows / 2),
                           0, 0, cv::INTER_NEAREST);
            } else {
                const cv::Mat &src =
                    scale_space_pyr[(octave * octave_size) + layer - 1];
                GaussianBlur(src, dst, cv::Size(), sigmas[layer], sigmas[layer]);
            }
        }
    }
}

void mapper_cv::SIFT::build_DoG() {
    size_t octave_size = 2 + octave_scale_size;
    for (size_t octave(0); octave < octaves_num; ++octave) {
        for (size_t layer(0); layer < octave_size; ++layer) {
            const cv::Mat &src1 = scale_space_pyr[octave * (octave_scale_size + 3) + layer];
            const cv::Mat &src2 = scale_space_pyr[octave * (octave_scale_size + 3) + layer + 1];
            cv::Mat &dst = DoG_pyr[octave * octave_size + layer];
            cv::subtract(src2, src1, dst, cv::noArray(), cv::DataType<float>::type);
        }
    }
}

void mapper_cv::SIFT::find_features() {
    int octave_size = 2 + octave_scale_size;
    cv::KeyPoint kpt;
    for (size_t octave = 0; octave < octaves_num; ++octave) {
        for (size_t layer(0); layer < octave_size; ++layer) {
            Mat &curr_DoG = DoG_pyr.at(octave * octave_size + layer);
            for (size_t row = IMG_BORDER; row < curr_DoG.rows - IMG_BORDER; ++row) {
                for (size_t col = IMG_BORDER; col < curr_DoG.cols - IMG_BORDER; ++col) {
                    bool is_stable = curr_DoG.at<float>(row, col) >= CONSTRAST_THRESH;
                    if (is_stable && is_local_extrema(DoG_pyr, octave_size, row, col, octave, layer)) {
                        int row_approx(row), col_approx(col), layer_approx(layer);
                        float DoG_estimate(0);
                        cv::Vec3f approx_offset;
                        if (adjust_local_extrema(
                                DoG_pyr, SIFT_MAX_INTERP_STEPS, col_approx, row_approx, layer_approx,
                                DoG_estimate, approx_offset, octave, layer, octave_size)) {
                                    get_keypoint(
                                        DoG_pyr, kpt, sigma, octave, row_approx, 
                                        col_approx, layer_approx, approx_offset, octave_size);
                                    const cv::Mat& img = scale_space_pyr[octave_scale_size * octave + layer];
                                    if (calc_descriptor(img, kpt, layer, octave)) {
                                        keypoints.push_back(kpt);
                                    }
                        }
                    }
                }
            }
        }
    }
}

bool mapper_cv::SIFT::calc_descriptor(const cv::Mat& img, cv::KeyPoint& kpt, size_t layer, size_t octave) {   
    return true;
}
