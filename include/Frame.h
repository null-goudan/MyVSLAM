#ifndef FRAME_H
#define FRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBExtractor.h"

#include <opencv2/opencv.hpp>

using namespace std;

namespace Goudan_SLAM
{
    #define FRAME_GRID_ROWS 48
    #define FRAME_GRID_COLS 64
    class MapPoint;
    class KeyFrame;

    class Frame
    {
    public:
        Frame();
        Frame(const Frame &frame);
        Frame(const cv::Mat &imGray, const double &timeStamp, ORBExtractor *extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

        // 提取的关键点存放在mvKeys和mDescriptors中 ORB是直接调orbExtractor提取的
        void ExtractORB(const cv::Mat &im);
        // 计算Bag of words特征并存放在mBowVec中
        void ComputeBoW();
        // 设置相机姿态(用Tcw 更新 mTcw)
        void SetPose(cv::Mat Tcw);
        // 计算旋转，平移和相机中心矩阵
        void UpdatePoseMatrices();

        // 返回相机中心相对于世界坐标系的变换矩阵
        inline cv::Mat GetCameraCenter()
        {
            return mOw.clone();
        }

        // 返回旋转矩阵的逆
        inline cv::Mat GetRotationInverse()
        {
            return mRwc.clone();
        }

        // 判断路标点是否在视野中
        bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

        // 计算出特征点所属单元
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        // 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
        vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1, const int maxLevel = -1) const;


    public:
        // Vocabulary used for relocalization.
        ORBVocabulary* mpORBvocabulary;

        // 特征提取器
        ORBExtractor *mpORBextractorLeft;

        // 此帧时间点
        double mTimeStamp;

        // 标定的内参，以及OpenCV矫正畸变的参数
        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;
        cv::Mat mDistCoef;

        float mb;   // baseline
        float mbf;  // baseline multiplied by fx

        float mThDepth;

        // KeyPoints 数量
        int N;

        // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
        // In the stereo case, mvKeysUn is redundant as images must be rectified.
        // In the RGB-D case, RGB images can be distorted.
        // mvKeys:原始左图像提取出的特征点（未校正）
        // mvKeysRight:原始右图像提取出的特征点（未校正）
        // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
        std::vector<cv::KeyPoint> mvKeys;
        std::vector<cv::KeyPoint> mvKeysUn;

        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // ORB descriptor, each row associated to a keypoint.
        // 左目摄像头和右目摄像头特征点对应的描述子
        cv::Mat mDescriptors;

        // MapPoints associated to keypoints, NULL pointer if no association.
        // 每个特征点对应的MapPoint
        std::vector<MapPoint *> mvpMapPoints;

        // Flag to identify outlier associations.
        // 观测不到Map中的3D点
        std::vector<bool> mvbOutlier;

        // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
        // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
        // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
        // FRAME_GRID_ROWS 48
        // FRAME_GRID_COLS 64
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        // Camera pose.
        cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵

        // Current and Next Frame id.
        static long unsigned int nNextId; ///< Next Frame id.
        long unsigned int mnId;           ///< Current Frame id.

        // Reference Keyframe.
        KeyFrame *mpReferenceKF; // 指针，指向参考关键帧

        // 图像金字塔尺度参数
        int mnScaleLevels;      // 图像提金字塔的层数
        float mfScaleFactor;    // 图像提金字塔的尺度因子
        float mfLogScaleFactor; //
        vector<float> mvScaleFactors;
        vector<float> mvInvScaleFactors;
        vector<float> mvLevelSigma2;
        vector<float> mvInvLevelSigma2;

        // Undistorted Image Bounds (computed once).
        // 用于确定画格子时的边界
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;

        static bool mbInitialComputations;

    private:
        // Undistort keypoints given OpenCV distortion parameters.
        // (called in the constructor).
        void UndistortKeyPoints();

        // Computes image bounds for the undistorted image (called in the constructor).
        void ComputeImageBounds(const cv::Mat &imLeft);

        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        void AssignFeaturesToGrid();

        // Rotation, translation and camera center
        cv::Mat mRcw; ///< Rotation from world to camera
        cv::Mat mtcw; ///< Translation from world to camera
        cv::Mat mRwc; ///< Rotation from camera to world
        cv::Mat mOw;  ///< mtwc,Translation from camera to world
    };

}

#endif