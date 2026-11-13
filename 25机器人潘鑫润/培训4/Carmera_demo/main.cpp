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
const string READ_DIR      = "./cam_mix_img";  // 输入图片目录
const string BEFORE_CORNER = "./before_corners";       // 保存: 带角点的图
const string AFTER_CORNER  = "./after_corners";   // 保存: 去畸变后的图

int main()
{
    //======================收集图片====================
    vector<string> images;
    if (!fs::exists(READ_DIR))
    {
        cout << "图片目录不存在：" << READ_DIR << endl;
        return -1;
    }

    for (const auto &entry : fs::directory_iterator(READ_DIR))
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
        cerr << "未找到图片：" << READ_DIR << endl;
        return -1;
    }

    sort(images.begin(), images.end());   // 排序，保证顺序稳定
    cout << "共找到 " << images.size() << " 张图片\n" << endl;

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

    // 创建输出目录
    if (!fs::exists(BEFORE_CORNER)) fs::create_directories(BEFORE_CORNER);
    if (!fs::exists(AFTER_CORNER))  fs::create_directories(AFTER_CORNER);

    //======================提取角点====================
    for (const auto &fname : images)
    {
        Mat img = imread(fname);
        if (img.empty())
        {
            cout << "该图片无法读取：" << fs::path(fname).filename().string() << endl;
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

            // 画角点（第3参数传当前这张图的角点）
            Mat imgColor = img.clone();
            drawChessboardCorners(imgColor, CHESSBOARD_SIZE, corner, true);  

            // 保存带角点的图
            string name    = fs::path(fname).filename().string();
            string outPath = (fs::path(BEFORE_CORNER) / name).string();
            if (imwrite(outPath, imgColor))
                cout << "角点图已保存: " << outPath << endl;
            else
                cerr << "角点图保存失败: " << outPath << endl;

            imshow("corners", imgColor);
            waitKey(50);   
        }
    }

    if (valid_images.empty())
    {
        cerr << "没有一张图片成功找到角点，无法标定。" << endl;
        return -1;
    }

    //======================相机标定====================
    Mat img = imread(valid_images[0]);
    Size imageSize = img.size();

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    // 参数1：objpoints        世界坐标 vector<vector<Point3f>>
    // 参数2：imgpoints        图像坐标 vector<vector<Point2f>>
    // 参数3：imageSize        图像尺寸 Size(width, height)
    // 参数4：cameraMatrix     输出，3x3 内参矩阵 K
    // 参数5：distCoeffs       输出，畸变系数 [k1,k2,p1,p2,k3]
    // 参数6：rvecs            输出，每张图旋转向量 vector<Mat>
    // 参数7：tvecs            输出，每张图平移向量 vector<Mat>
    // 返回：rms 整体重投影误差（像素）
    double rms = calibrateCamera(   
        objpoints, imagepoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs);

    cout << "\n==========标定结果===========" << endl;
    cout << "重投影误差 (RMS): " << rms << " 像素" << endl;
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
    cout << "\nfx = " << fx << ", fy = " << fy << endl;
    cout << "cx = " << cx << ", cy = " << cy << endl;

    //======================去畸变====================
  
    Mat newMatrix = getOptimalNewCameraMatrix(  
        cameraMatrix, distCoeffs, imageSize, 1, imageSize);

    Mat mapx, mapy;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            newMatrix, imageSize, CV_32FC1, mapx, mapy);
                            
    for (size_t i = 0; i < valid_images.size(); i++)
    {
        Mat src = imread(valid_images[i]);
        if (src.empty()) continue;

        Mat undistorted;
        remap(src, undistorted, mapx, mapy, INTER_LINEAR);

        // 角点坐标同步去畸变（关键：第5参数传 newMatrix 才输出像素坐标）
        vector<Point2f> cornersUndist;
        undistortPoints(imagepoints[i], cornersUndist,
                        cameraMatrix, distCoeffs, noArray(), newMatrix);

        // 在去畸变图上画角点并标号（完全复刻参考图效果）
        Mat undistortedShow = undistorted.clone();
        
        // 定义 7 种颜色 (B, G, R)，从下到上依次对应 红、橙、黄、绿、蓝、紫、青
        vector<Scalar> colors = {
            Scalar(0, 0, 255),     // 第1行：红 (点 1~10)
            Scalar(0, 165, 255),   // 第2行：橙 (点 11~20)
            Scalar(0, 255, 255),   // 第3行：黄 (点 21~30)
            Scalar(0, 255, 0),     // 第4行：绿 (点 31~40)
            Scalar(255, 0, 0),     // 第5行：蓝 (点 41~50)
            Scalar(255, 0, 255),   // 第6行：紫 (点 51~60)
            Scalar(255, 255, 0)    // 第7行：青 (点 61~70)
        };

        int cols = CHESSBOARD_SIZE.width;  // 每行 10 个点

        // 1. 先画每行的连线
        for (int r = 0; r < CHESSBOARD_SIZE.height; r++)
        {
            Scalar color = colors[r % colors.size()];
            for (int c = 0; c < cols - 1; c++) // 连到倒数第二个点即可
            {
                int idx1 = r * cols + c;
                int idx2 = r * cols + c + 1;
                // 画一条线宽为 1 的实线
                line(undistortedShow, cornersUndist[idx1], cornersUndist[idx2], color, 1, LINE_AA);
            }
        }

        // 2. 再画圆点和数字
        for (size_t k = 0; k < cornersUndist.size(); k++)
        {
            Point2f pt = cornersUndist[k];
            
            int row = k / cols; 
            Scalar color = colors[row % colors.size()]; 

            // 画实心圆点
            circle(undistortedShow, pt, 3, color, -1, LINE_AA);
            
            // 标数字序号
            string text = to_string(k + 1);
            
           
            putText(undistortedShow, text, pt + Point2f(-6, 14),
                    FONT_HERSHEY_SIMPLEX, 0.4, color, 1, LINE_AA);
        }

        // 保存去畸变图
        string name    = fs::path(valid_images[i]).filename().string();
        string outPath = (fs::path(AFTER_CORNER) / name).string();
        if (imwrite(outPath, undistortedShow))
            cout << "已保存: " << outPath << endl;
        else
            cerr << "保存失败: " << outPath << endl;

        imshow("undistorted + corners", undistortedShow);
        waitKey(500);   
    }
    
    destroyAllWindows();
    return 0;
}