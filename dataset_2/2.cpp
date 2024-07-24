#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void detectZebraCrossing(const string& imgPath, const string& maskPath, int brightnessReduction = 60, double grayThreshold = 50) {

    Mat image = imread(imgPath);
    image = image - Scalar(brightnessReduction, brightnessReduction, brightnessReduction);
    image = max(image, Scalar(0, 0, 0)); 
    int height = image.rows;
    int width = image.cols;
    int cutoffHeight = static_cast<int>(height * 0.4);
    rectangle(image, Point(0, 0), Point(width, cutoffHeight), Scalar(0, 0, 0), FILLED);


    Mat mask = imread(maskPath, IMREAD_GRAYSCALE);
    resize(mask, mask, image.size());
    image.setTo(Scalar(0, 0, 0), mask);


    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);


    Mat grayThresh;
    threshold(gray, grayThresh, grayThreshold, 255, THRESH_BINARY);


    Mat medianBlurred;
    medianBlur(grayThresh, medianBlurred, 5);


    Mat morph;
    Mat closeKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat openKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(medianBlurred, morph, MORPH_CLOSE, closeKernel); 
    morphologyEx(morph, morph, MORPH_OPEN, openKernel); 


    Mat eroded, dilated;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(morph, eroded, kernel);
    dilate(eroded, dilated, kernel);


    imshow("效果图 - " + imgPath, dilated);


    string outputFileName = "final_" + imgPath.substr(imgPath.find_last_of("/\\") + 1);
    imwrite(outputFileName, dilated);

    waitKey(0);
}

int main() {
    string maskPath = "car_mask.png"; 
    for (int i = 1; i <= 5; ++i) {
        string imgPath = to_string(i) + ".jpg"; 
        if (imgPath == "3.jpg") {
            detectZebraCrossing(imgPath, maskPath, 110, 80);
        } else {
            detectZebraCrossing(imgPath, maskPath);
        }
    }
    return 0;
}
