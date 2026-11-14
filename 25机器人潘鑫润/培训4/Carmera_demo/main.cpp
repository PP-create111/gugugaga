#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

/*===============参数配置==============*/
const Size  CHESSBOARD_SIZE(10, 7);        // 内角点: 宽10 高7
const float SQUARE_SIZE = 5.0f;            // 方格边长(mm)
const string READ_DIR      = "./cam_mix_img";  

/*===============辅助函数（来自文件2 风格）==============*/
// 7 色调色板
static const Scalar kPalette[7] = {
    Scalar( 60,  60, 255),   // 红
    Scalar( 60, 150, 255),   // 橙
    Scalar( 60, 220, 255),   // 黄
    Scalar( 80, 220, 120),   // 绿
    Scalar(230, 220,  80),   // 青
    Scalar(255, 160,  70),   // 蓝
    Scalar(255,  90, 220),   // 品红
};

// 把角点按 7 行 × 10 列画出来：行内连线、每个点画圆、标 1~70 编号
static void drawNumberedCorners(Mat& canvas, const vector<Point2f>& corners)
{
    const int kRows = CHESSBOARD_SIZE.height;
    const int kCols = CHESSBOARD_SIZE.width;
    for (int r = 0; r < kRows; ++r)
    {
        const Scalar color = kPalette[r % 7];
        for (int c = 0; c < kCols; ++c)
        {
            const int idx = r * kCols + c;
            if (idx >= (int)corners.size()) return;
            const Point p(cvRound(corners[idx].x), cvRound(corners[idx].y));
            if (c > 0)
            {
                const Point q(cvRound(corners[idx - 1].x), cvRound(corners[idx - 1].y));
                line(canvas, q, p, color, 1, LINE_AA);
            }
            circle(canvas, p, 3, color, -1, LINE_AA);
            putText(canvas, to_string(idx + 1), p + Point(4, -4),
                    FONT_HERSHEY_SIMPLEX, 0.38, color, 1, LINE_AA);
        }
    }
}

// 左右拼两张图，并加上 before / after 标签
static Mat sideBySide(const Mat& a, const Mat& b)
{
    Mat out(a.rows, a.cols * 2 + 8, a.type(), Scalar(30, 30, 30));
    a.copyTo(out(Rect(0, 0, a.cols, a.rows)));
    b.copyTo(out(Rect(a.cols + 8, 0, b.cols, b.rows)));
    putText(out, "before", Point(10, 22), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0, 255, 255), 2, LINE_AA);
    putText(out, "after", Point(a.cols + 18, 22), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0, 255, 255), 2, LINE_AA);
    return out;
}

int main(int argc, char** argv)
{
    
    string readDir = READ_DIR;     // 默认用常量
    vector<string> images;
    if (!fs::exists(readDir))
    {
        cout << "图片目录不存在：" << readDir << endl
             << "用法: calib_demo <图片目录> [--show]\n";
        return -1;
    }

    for (const auto &entry : fs::directory_iterator(readDir))
    {
        if (!entry.is_regular_file()) continue;   // 跳过子目录

        string path = entry.path().string();
        string ext  = entry.path().extension().string();

        // 后缀转小写，兼容 .JPG/.PNG
        transform(ext.begin(), ext.end(), ext.begin(),
                  [](unsigned char c){ return tolower(c); });

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            images.push_back(path);
    }

    if (images.empty())
    {
        cerr << "未找到图片：" << readDir << endl;
        return -1;
    }

    sort(images.begin(), images.end());   // 排序，保证顺序稳定
    cout << "共找到 " << images.size() << " 张图片，目录 " << readDir << "\n" << endl;

    if (images.size() < 6)
    {
        cout << "图片太少，至少要 6 张才能标定\n";
        return 1;
    }

    //======================准备世界坐标====================
    vector<Point3f> objs;
    // 参考图中的角点顺序：右下角为起点(0,0)，往左递增，然后往上换行。
    // 所以我们反转 X 和 Y 的坐标生成逻辑，让序号1落在右下角
    for (int i = 0; i < CHESSBOARD_SIZE.height; i++)
    {
        for (int j = 0; j < CHESSBOARD_SIZE.width; j++)
        {
            objs.push_back(Point3f((CHESSBOARD_SIZE.width - 1 - j) * SQUARE_SIZE,
                                   (CHESSBOARD_SIZE.height - 1 - i) * SQUARE_SIZE,
                                   0));
        }
    }

    vector<vector<Point3f>> objpoints;
    vector<vector<Point2f>> imagepoints;
    vector<string> valid_images;

    // 亚像素优化的终止条件
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);

    // 记录第一张成功检出的图与角点（用于画 corners.jpg）
    Mat firstHitImage;
    vector<Point2f> firstHitCorners;
    Size imageSize;

    //======================提取角点====================
    int failed = 0;
    for (const auto &fname : images)
    {
        Mat img = imread(fname);
        if (img.empty())
        {
            cout << "该图片无法读取：" << fs::path(fname).filename().string() << endl;
            ++failed;
            continue;
        }

        Mat Gray;
        cvtColor(img, Gray, COLOR_BGR2GRAY);

        vector<Point2f> corner;

        bool found = findChessboardCorners(
            Gray, CHESSBOARD_SIZE, corner,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if (found)
        {
            // 亚像素精细化
            cornerSubPix(Gray, corner, Size(11, 11), Size(-1, -1), criteria);

            objpoints.push_back(objs);
            imagepoints.push_back(corner);
            valid_images.push_back(fname);

            // 记录第一张成功检出的图与角点
            if (firstHitImage.empty())
            {
                imageSize       = Gray.size();
                firstHitImage   = img.clone();
                firstHitCorners = corner;
            }

            cout << "[OK]   " << fs::path(fname).filename().string()
                 << "  角点: " << corner.size() << endl;
        }
        else
        {
            cout << "[FAIL] " << fs::path(fname).filename().string()
                 << "  未检测到角点" << endl;
            ++failed;
        }
    }

    cout << "\n角点检出: " << valid_images.size() << " / " << images.size()
         << " 张（" << failed << " 张检不出来，正常）\n" << endl;

    if (valid_images.size() < 6)
    {
        cerr << "有效图片不足 6 张，无法标定。检查标定板尺寸是不是写成了 11x8\n";
        return 2;
    }

    //======================相机标定====================
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(
        objpoints, imagepoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs);

    cout << "\n==========标定结果===========" << endl;
    printf("重投影误差 (RMS): %.4f 像素\n", rms);
    cout << "\n内参矩阵 K:\n" << cameraMatrix << endl;
    cout << "\n畸变系数 [k1, k2, p1, p2, k3]:\n" << distCoeffs << endl;

    /*
       内参矩阵:
            ┌ fx   0   cx ┐
        K = │  0  fy   cy │
            └  0   0    1 ┘
    */
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);
    printf("\nfx = %.4f, fy = %.4f\n", fx, fy);
    printf("cx = %.4f, cy = %.4f\n", cx, cy);

    //======================逐图重投影误差 + Top3====================
    vector<pair<double,int>> errs;
    for (size_t i = 0; i < objpoints.size(); ++i)
    {
        vector<Point2f> projected;
        projectPoints(objpoints[i], rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, projected);
        double sum = 0.0;
        for (size_t k = 0; k < projected.size(); ++k)
        {
            Point2f d = projected[k] - imagepoints[i][k];
            sum += d.x * d.x + d.y * d.y;
        }
        errs.emplace_back(sqrt(sum / projected.size()), (int)i);
    }
    sort(errs.rbegin(), errs.rend());   // 降序
    cout << "\n误差最大的三张（可考虑剔除后再标一次）:\n";
    for (int k = 0; k < 3 && k < (int)errs.size(); ++k)
        printf("  #%d  %.4f 像素\n", k + 1, errs[k].first);

    //======================写 corners.jpg（第一张成功图）====================
    Mat cornersCanvas = firstHitImage.clone();
    drawNumberedCorners(cornersCanvas, firstHitCorners);
    imwrite("corners.jpg", cornersCanvas);
    cout << "\n已写出 corners.jpg\n";

    //======================去畸变（只处理第一张）====================
    Mat newMatrix = getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize, 1, imageSize);

    Mat before = firstHitImage;
    Mat after;
    // 用 optimal 新内参 + initUndistortRectifyMap/remap，保留全视野
    Mat mapx, mapy;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            newMatrix, imageSize, CV_32FC1, mapx, mapy);
    remap(before, after, mapx, mapy, INTER_LINEAR);

    imwrite("undistort_before.jpg", before);
    imwrite("undistort_after.jpg",  after);
    imwrite("undistort_compare.jpg", sideBySide(before, after));
    cout << "已写出 undistort_before.jpg / undistort_after.jpg / undistort_compare.jpg\n";

    imshow("corners", cornersCanvas);
    imshow("before / after", sideBySide(before, after));
    waitKey(0);
    destroyAllWindows();
    return 0;
}
