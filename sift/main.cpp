
#include "precomp.h"
#include <iostream>
#include <string>
#include <thread>

#include "sift.h"

using std::cout;
using std::endl;
using std::vector;

int main() {

    auto image_paths = vector<std::string>({
        "C:\\Users\\bassam\\Desktop\\backup\\samurai.jpg",
        "C:\\Users\\bassam\\Desktop\\backup\\me.jpg",
        "C:\\Users\\bassam\\Desktop\\backup\\cat.jpg",
        "C:\\Users\\bassam\\Desktop\\backup\\samurai2.jpg",
    });
    for (auto &path : image_paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            cout << "cannot read image at path: " << path << endl;
            continue;
        }
        cout << "Getting features for image at path: " << path << endl;
        cout << "rows: " << img.rows << ", cols: " << img.cols << endl;

        mapper_cv::SIFT sift = mapper_cv::SIFT(img);
        sift.displayScaleSpace();
    }

    return 0;
}