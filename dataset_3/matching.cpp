#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void matchUsingSIFT(const Mat& img, const Mat& templ, const string& imageName) {
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypointsImg, keypointsTempl; 
    Mat descriptorsImg, descriptorsTempl; 

    detector->detectAndCompute(img, noArray(), keypointsImg, descriptorsImg);
    detector->detectAndCompute(templ, noArray(), keypointsTempl, descriptorsTempl);

    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches; 
    matcher.knnMatch(descriptorsTempl, descriptorsImg, knnMatches, 2);

    const float ratioThresh = 0.6;
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    if (goodMatches.size() >= 4) {
        vector<Point2f> templPoints, imgPoints;
        for (size_t i = 0; i < goodMatches.size(); i++) {
            templPoints.push_back(keypointsTempl[goodMatches[i].queryIdx].pt);
            imgPoints.push_back(keypointsImg[goodMatches[i].trainIdx].pt);
        }

        Mat H = findHomography(templPoints, imgPoints, RANSAC);
        if (!H.empty()) {
            Mat imgMatches;
            drawMatches(templ, keypointsTempl, img, keypointsImg, goodMatches, imgMatches,
                        Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            string outputPath = "sift_result_" + imageName;
            imwrite(outputPath, imgMatches);
        }
    }
}

int main() {
    string archivePath = "archive";
    string templatePath = "template";
    for (const auto& templateEntry : fs::directory_iterator(templatePath)) {
        Mat templ = imread(templateEntry.path().string(), IMREAD_COLOR);
        for (const auto& archiveEntry : fs::directory_iterator(archivePath)) {
            Mat img = imread(archiveEntry.path().string(), IMREAD_COLOR);
            matchUsingSIFT(img, templ, archiveEntry.path().filename().string());
        }
    }

    return 0;
}
