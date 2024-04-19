#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Frame.h"
#include "Map.h"
#include "ORBExtractor.h"

#include <mutex>
#include <set>

namespace Goudan_SLAM
{
    class Map;
    class MapPoint;
    class Frame;
    // class KeyFrameDatabase;

    class KeyFrame
    {
    public:
        // KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);
        KeyFrame(Frame &F, Map *pMap);

        // Pose functions
        void SetPose(const cv::Mat &Tcw);
        cv::Mat GetPose();
        cv::Mat GetPoseInverse();
        cv::Mat GetCameraCenter();
        cv::Mat GetRotation();
        cv::Mat GetTranslation();

        // 计算 bag of wards
        void ComputeBow();

        // 共视图函数
        void AddConnection(KeyFrame *pKF, const int &weight);
        void EraseConnection(KeyFrame *pKF);
        void UpdateConnections();
        void UpdateBestCovisibles();
        std::set<KeyFrame *> GetConnectedKeyFrames();
        std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();
        std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);
        std::vector<KeyFrame *> GetConvisiblesByWeight(const int &w);
        int GetWeight(KeyFrame *pKF);

        // 关键帧函数
        std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;

        // :TODO

    public:
        // The following variables are accesed from only 1 thread or never change (no mutex needed).
        // nNextID -> nLastID
        static long unsigned int nNextId;
        long unsigned int mnID; // 当前的关键帧的ID号
        // 每个关键帧由一个普通帧初始化而来， mnFrameId记录该KeyFrame是由哪个Frame初始化的
        const long unsigned int mnFrameId;

        const double mTimeStamp;

        // Grid (to speed up feature matching)
        // 和Frame类中的定义相同
        const int mnGridCols;
        const int mnGridRows;
        const float mfGridElementWidthInv;
        const float mfGridElementHeightInv;

        // Tracking线程使用变量
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnFuseTargetForKF;

        // 局部优化线程使用变量
        long unsigned int mnBALocalForKF;
        long unsigned int mnBAFixedForKF;

        // 关键帧数据库使用变量
        long unsigned int mnLoopQuery;
        int mnLoopWords;
        float mLoopScore;
        long unsigned int mnRelocQuery;
        int mnRelocWords;
        float mRelocScore;

        // 回环检测使用变量
        cv::Mat mTcwGBA;
        cv::Mat mTcwBefGBA;
        long unsigned int mnBAGlobalForKF;

        // 矫正畸变参数
        const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

        // 关键点的数量
        const int N;

        // // BoW
        // DBoW2::BowVector mBowVec;      ///< Vector of words to represent images
        // DBoW2::FeatureVector mFeatVec; ///< Vector of nodes with indexes of local features

        // Pose relative to parent (this is computed when bad flag is activated)
        cv::Mat mTcp;

        // Scale
        const int mnScaleLevels;
        const float mfScaleFactor;
        const float mfLogScaleFactor;
        const std::vector<float> mvScaleFactors; // 尺度因子，scale^n，scale=1.2，n为层数
        const std::vector<float> mvLevelSigma2;  // 尺度因子的平方
        const std::vector<float> mvInvLevelSigma2;

        // Image bounds and calibration
        const int mnMinX;
        const int mnMinY;
        const int mnMaxX;
        const int mnMaxY;
        const cv::Mat mK;

        // 以下变量需要通过锁进行线程安全
    // protected:
        
    };

}

#endif