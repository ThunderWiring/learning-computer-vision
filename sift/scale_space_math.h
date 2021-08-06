#ifndef __IMAGE_MATH_H__
#define __IMAGE_MATH_H__

#include "precomp.h"

#define INTER_STABILITY_THR 0.5f
#define EDGE_THRESH 10.0f
#define CONSTRAST_THRESH 0.03f
#define IMG_BORDER 5

using cv::KeyPoint;
using cv::Mat;
using cv::Matx22f;
using cv::Matx31f;
using cv::Matx33f;
using cv::Vec3f;
using std::vector;

namespace mapper_cv {
/**
 * Contains math functions to do matrix calculations on the image, as well as derivatives.
 * Most those functions are wrapper for OpenCV math functions.
*/

/**
 * @brief returns true if the pixel at (row, col) is a local extrema, that is, it's less
 * or bigger than its 26 neighbours, which are the 8 nieghbours and the 9 pixels in the 
 * octave before and the octave after that are centered in (row, col)
 * @param scale_space scale space pyramids (multiple octaves)
 * @param octave_size number of images per octave
 * @param row, @param col  the coordinates of the pixel we're checking if it's a extremum
 * @param octave the idx of the octave for the image which we're cheking the point for extremum
 * @param scale the scale of the image of interest in the specified octave.
*/
static bool is_local_extrema(
    vector<Mat> &scale_space, int octave_size, int row, int col, int octave, int scale) {
    int curr_image_idx = octave * octave_size + scale;
    if (curr_image_idx >= scale_space.size() - 1 || curr_image_idx <= 0) {
        return false; // idxs out of range
    }
    const Mat curr = scale_space.at(curr_image_idx);
    const Mat next = scale_space.at(curr_image_idx + 1);
    const Mat prev = scale_space.at(curr_image_idx - 1);

    auto is_valid_idx = [](const Mat &img, int row, int col) {
        bool is_valid_row = row > 0 && row + 1 < img.rows;
        bool is_valid_col = col > 0 && col + 1 < img.cols;
        return is_valid_col && is_valid_row;
    };
    float val = curr.at<float>(row, col);

    auto is_larger = [is_valid_idx, val, row, col](const Mat &img) {
        return is_valid_idx(img, row, col) &&
               val >= img.at<float>(row - 1, col - 1) && val >= img.at<float>(row - 1, col) && val >= img.at<float>(row - 1, col + 1) && val >= img.at<float>(row, col - 1) && val >= img.at<float>(row, col) && val >= img.at<float>(row, col + 1) && val >= img.at<float>(row + 1, col - 1) && val >= img.at<float>(row + 1, col) && val >= img.at<float>(row + 1, col + 1);
    };
    auto is_less = [is_valid_idx, val, row, col](const Mat &img) {
        return is_valid_idx(img, row, col) &&
               val <= img.at<float>(row - 1, col - 1) && val <= img.at<float>(row - 1, col) && val <= img.at<float>(row - 1, col + 1) && val <= img.at<float>(row, col - 1) && val <= img.at<float>(row, col) && val <= img.at<float>(row, col + 1) && val <= img.at<float>(row + 1, col - 1) && val <= img.at<float>(row + 1, col) && val <= img.at<float>(row + 1, col + 1);
    };

    bool is_maxima = is_larger(curr) && is_larger(prev) && is_larger(next);
    bool is_minima = is_less(curr) && is_less(prev) && is_less(next);

    return is_maxima || is_minima;
}

/**
 * @brief Returns 3x3 Hessian matrix for the scale space function D(x, y, sigma), 
 * at the given point (row, col).
*/
static Matx33f calc_scale_space_hessian(
    vector<Mat> &DoG_scale_space, int row, int col, int octave, int scale, int octave_size) {
    int idx = octave * (octave_size + 2) + scale;
    const Mat &curr = DoG_scale_space[idx];
    const Mat &prev = DoG_scale_space[idx - 1];
    const Mat &next = DoG_scale_space[idx + 1];

    float dxx = next.at<float>(row, col) + prev.at<float>(row, col) - 2 * curr.at<float>(row, col);
    float dyy = curr.at<float>(row + 1, col) + curr.at<float>(row - 1, col) - 2 * curr.at<float>(row, col);
    float dss = curr.at<float>(row, col + 1) + curr.at<float>(row, col - 1) - 2 * curr.at<float>(row, col);

    float dxy = (next.at<float>(row + 1, col) - next.at<float>(row - 1, col) - prev.at<float>(row + 1, col) + prev.at<float>(row - 1, col)) * 0.25;
    float dxs = (next.at<float>(row, col + 1) - next.at<float>(row, col - 1) - prev.at<float>(row, col + 1) + prev.at<float>(row, col - 1)) * 0.25;
    float dys = (curr.at<float>(row + 1, col + 1) - curr.at<float>(row + 1, col - 1) - curr.at<float>(row - 1, col + 1) + curr.at<float>(row - 1, col - 1)) * 0.25;

    // the matrix is symmtric
    return Matx33f(
        dxx, dxy, dxs,
        dxy, dyy, dys,
        dxs, dys, dss);
}

/**
 * @brief Returns a 3x1 vector for (dx, dx, d_sigma) - derivative vector for the scale space function D(x, y, sigma), 
 * at the given point (row, col).
*/
static Vec3f calc_gradient(vector<Mat> &DoG_scale_space, int row, int col,
                           size_t octave, size_t scale, size_t octave_size) {
    int idx = octave * (octave_size + 2) + scale;
    const Mat &curr = DoG_scale_space[idx];
    const Mat &prev = DoG_scale_space[idx - 1];
    const Mat &next = DoG_scale_space[idx + 1];
    return Vec3f(
        (next.at<float>(row, col) - prev.at<float>(row, col)) * 0.5f, // der in sigma dim
        (curr.at<float>(row + 1, col) - curr.at<float>(row - 1, col)) * 0.5f,
        (curr.at<float>(row, col + 1) - curr.at<float>(row, col - 1)) * 0.5f);
}

/**
 * @brief Returns false if the candid keypoint location is on an edge, or if 
 * it's a low contrast.
 * 
*/
static bool is_keypoint_stable(vector<Mat> &DoG_scale_space, Vec3f approx, int &col, int &row, int octave, size_t scale,
                               int octave_size) {
    Mat H_3x3 = Mat(calc_scale_space_hessian(DoG_scale_space, row, col, octave, scale, octave_size));
    float dxx = H_3x3.at<float>(0, 0);
    float dyy = H_3x3.at<float>(1, 1);
    float dxy = H_3x3.at<float>(0, 1);
    Matx22f H = Matx22f(
        dxx, dxy,
        dxy, dyy);

    float tr = cv::trace(H);
    float det = cv::determinant(H);
    float edginess = (tr * tr) / det;
    float edge_thr = (EDGE_THRESH + 1) * (EDGE_THRESH + 1) / EDGE_THRESH;
    if (edginess >= edge_thr) {
        return false;
    }

    int idx = octave * (octave_size + 2) + scale;
    const Mat &curr = DoG_scale_space[idx];
    Vec3f grad = calc_gradient(DoG_scale_space, row, col, octave, scale, octave_size);
    float t = grad.dot(Matx31f(approx[0], approx[1], approx[2]));
    float contrast = curr.at<float>(row, col) * scale + t * 0.5; // the Taylor approx
    
    return std::abs(contrast) >= 0.8 * CONSTRAST_THRESH;
}

static float get_contrast(vector<Mat> &DoG_scale_space, Vec3f approx, int &col, int &row, size_t octave, size_t scale,
                          size_t octave_size) {
    int idx = octave * (octave_size + 2) + scale;
    const Mat &curr = DoG_scale_space[idx];
    Vec3f grad = calc_gradient(DoG_scale_space, row, col, octave, scale, octave_size);
    float t = grad.dot(Matx31f(approx[0], approx[1], approx[2]));
    return curr.at<float>(row, col) * scale + t * 0.5; // the Taylor approx
}

/**
 * @brief performs approximation on the local extrema point to get a more precise location
 * in the continuious space.
*/
static bool adjust_local_extrema(
    vector<Mat> &DoG_scale_space, int iterations, int &col, int &row,
    int &layer_approx, float &DoG_estimate, Vec3f &approx_offset, int octave, int scale,
    int octave_size) {
    int interpolation_step = 0;
    for (; interpolation_step < iterations; ++interpolation_step) {
        try {
            if (scale < 1 || scale > octave_size) {
                return false;
            }
            
            int idx = octave * (octave_size + 2) + scale;
            if (idx < 1 || idx > DoG_scale_space.size() - 1) {
                return false;
            }

            const Mat &curr = DoG_scale_space[idx];
            if (col < IMG_BORDER || col >= curr.cols - IMG_BORDER ||
                row < IMG_BORDER || row >= curr.rows - IMG_BORDER) {
                return false;
            }

            // calc hessian
            Matx33f H = calc_scale_space_hessian(DoG_scale_space, row, col, octave, scale, octave_size);

            // calc gradient
            Vec3f g = calc_gradient(DoG_scale_space, row, col, octave, scale, octave_size);

            // calc approximation
            approx_offset = H.solve(g, cv::DECOMP_LU);

            // if approx within range, break.
            if (approx_offset[0] < INTER_STABILITY_THR &&
                approx_offset[1] < INTER_STABILITY_THR &&
                approx_offset[2] < INTER_STABILITY_THR) {
                break;
            } else if (std::abs(approx_offset[0]) > (float)(INT_MAX / 3) ||
                       std::abs(approx_offset[1]) > (float)(INT_MAX / 3) ||
                       std::abs(approx_offset[2]) > (float)(INT_MAX / 3)) {
                return false;
            }

            col -= cvRound(approx_offset[2]);
            row -= cvRound(approx_offset[1]);
            scale -= cvRound(approx_offset[0]);

        } catch (std::exception &e) {
            std::cout << "exception" << std::endl;
            return false;
        }
    }
    if (interpolation_step >= iterations) {
        return false;
    }
    return is_keypoint_stable(
        DoG_scale_space, approx_offset, col, row, octave, scale, octave_size);
}

static void get_keypoint(
    vector<Mat> &DoG_scale_space, KeyPoint &kpt, float sigma, int octave, int row, int col,
    int scale, Vec3f &approx_offset, int octave_size) {
    // see equation 15 in http://www.ipol.im/pub/art/2014/82/article.pdf to understand the math
    kpt.pt.x = (col + approx_offset[0]) * (1 << octave);
    kpt.pt.y = (row + approx_offset[1]) * (1 << octave);
    kpt.octave = octave + (scale << 8) + (cvRound((approx_offset[2] + 0.5) * 255) << 16);
    float contrast = get_contrast(DoG_scale_space, approx_offset, col, row, octave, scale, octave_size);
    kpt.response = std::abs(contrast);
    // see equation 15 in http://www.ipol.im/pub/art/2014/82/article.pdf to understand the math
    kpt.size = sigma *powf(2.f, (scale + approx_offset[2]) / octave_size)*(1 << octave)*2;
}

}; // namespace mapper_cv

#endif /*__IMAGE_MATH_H__*/