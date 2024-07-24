#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void processImage(const string& imgPath, double scale, const string& interpolation, int dx, int dy, const string& rotationCenter, double angle) {
    Mat image = imread(imgPath);
    if (image.empty()) {
        cerr << "无法打开或找到图片 " << imgPath << endl;
        return;
    }

    // 缩放图像
    Mat scaledImage;
    int interpolationMethod = (interpolation == "NEAREST") ? INTER_NEAREST : INTER_LINEAR;
    resize(image, scaledImage, Size(), scale, scale, interpolationMethod);

    // 平移图像
    Mat translatedImage;
    Mat translationMatrix = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    warpAffine(scaledImage, translatedImage, translationMatrix, scaledImage.size(), interpolationMethod, BORDER_CONSTANT, Scalar(255, 255, 255));

    // 裁剪平移后黑色边框
    Rect cropRect(dx > 0 ? dx : 0, dy > 0 ? dy : 0, translatedImage.cols - abs(dx), translatedImage.rows - abs(dy));
    translatedImage = translatedImage(cropRect);

    // 旋转图像
    Point2f center = (rotationCenter == "center") ? Point2f(translatedImage.cols / 2.0, translatedImage.rows / 2.0) : Point2f(0, 0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);

    Rect bbox = RotatedRect(center, translatedImage.size(), angle).boundingRect();
    rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    Mat rotatedImage;
    warpAffine(translatedImage, rotatedImage, rotationMatrix, bbox.size(), interpolationMethod, BORDER_CONSTANT, Scalar(255, 255, 255));

    // 创建掩膜以将背景设置为白色
    Mat mask;
    cvtColor(rotatedImage, mask, COLOR_BGR2GRAY);
    threshold(mask, mask, 254, 255, THRESH_BINARY);
    rotatedImage.setTo(Scalar(255, 255, 255), mask);

    // 保存处理后的图像
    fs::path outputPath = "processed_" + fs::path(imgPath).filename().string();
    imwrite(outputPath.string(), rotatedImage);
}

void processImagesFromFile(const string& csvFilePath) {
    ifstream file(csvFilePath);
    if (!file.is_open()) {
        cerr << "无法打开 CSV 文件" << endl;
        return;
    }

    string line;
    getline(file, line); // 跳过表头

    while (getline(file, line)) {
        stringstream s(line);
        string imgName, interpolation, rotationCenter;
        double imgScale;
        int imgHorizontal, imgVertical;
        double rotationAngle;
        
        if (getline(s, imgName, ',') &&
            s >> imgScale && s.ignore() &&
            getline(s, interpolation, ',') &&
            s >> imgHorizontal && s.ignore() &&
            s >> imgVertical && s.ignore() &&
            getline(s, rotationCenter, ',') &&
            s >> rotationAngle) {
            
            processImage(imgName, imgScale, interpolation, imgHorizontal, imgVertical, rotationCenter, rotationAngle);
        } else {
            cerr << "解析行出错: " << line << endl;
        }
    }
}

int main() {
    processImagesFromFile("experiment1.csv");
    return 0;
}

