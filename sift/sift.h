#ifndef SIFT_H
#define SIFT_H

#include "scale_space_math.h"

using std::vector;

// SIFT paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
// high level explaination of the algo: http://weitz.de/sift/index.html?size=large
// How Histogram of Oriented Gradients work: https://learnopencv.com/histogram-of-oriented-gradients/

namespace mapper_cv {
class SIFT {
private:
    // maximum steps of keypoint interpolation before failure
    static const size_t SIFT_MAX_INTERP_STEPS = 5;

    /** Original image (grayscale) without blurring */
    cv::Mat base_img;
    cv::Mat base_img_color;

    /** Number of octaves there are in the scale space */
    size_t octaves_num;

    /** Number of images per octave (denoted by S in the paper)*/
    size_t octave_scale_size;

    /** initial blurring degree for 1st image in the 1st octave. */
    float sigma;
    
    vector<double> sigmas;

    /** 
     * Contains the Gaussian pyramid representation of the image with its octave scales. 
     * Size of this vector is [octaves_num * (3 + octave_scale_size)]
    */
    vector<cv::Mat> scale_space_pyr;

    /** Diffrencial of Gaussian pyramid. */
    vector<cv::Mat> DoG_pyr;

    /** Contains the keypoints of the image. */
    vector<cv::KeyPoint> keypoints;

    cv::Mat descriptors;

    /**
     * @brief The seed image which is passed to the c'tor should be upsampled and 
     * then blurred with sigma_min = 0.5
    */
    void prep_process_image();

    /**
     * @brief calculates the scale space of the image. A sclae space is basically 
     * blurring different scales of the image. Each scale level is called octave.
     * So a scale space is basically a mapping between a scale factor to a set of blurred images:
     * @code
     *      {
     *           0.50x -> octave_scale_0.50
     *           0.25x -> octave_scale_0.25
     *           . . .
     *      }
     * 
     * Where octave_scale_0.XX is basically a set of a blurred images of different level
     * scaled by 0.XX of the original image size.
     * 
     * According to the paper, the number of blurred images per octave is:
     * (octave_scale_size + 3)
     * 
     * In short, it's creating a Gaussian pyramid representation of the image and 
     * blurring each scale level (octave) multiple times.
     * 
    */
    void calc_scale_space();

    /**
     * @brief builds the diffrencial of gaussian space from the scale space 
     * (i.e. gaussian pyramid which was built by @see calc_scale_space)
     * 
     * This function must not be called before calling calc_scale_space()
     * 
    */
    void build_DoG();

    /**
     * @brief Finds candidate keypoints in the image.
     * 1. Find local discrete extremas in the 3D space (rows, cols, sigma)
     * 2. Refine the extrema position using Taylor series of the DoG function
     * 
    */
    void find_features();

    /**
     * @brief Returns true if the keypoint neighbourhood is within the image borders.
     * @param img to extract the pixel patch neighbourhood of the keypoint.
    */
    bool calc_descriptor(const cv::Mat& img, cv::KeyPoint& kpt, size_t layer, size_t octave);

public:
    /**
    * @param image Original image to calculate keypoints and descriptors for.
   */
    explicit SIFT(cv::Mat &image);

    /**
     * @brief Detects the keypoints and descriptors for the image passed in the c'tor.
    */
    void detectFeatures();

    void displayScaleSpace();
    void displayKeypoints();

    // todo: add function to do matching between 2 descriptors
};

}; // namespace mapper_cv

#endif /*SIFT_H*/
