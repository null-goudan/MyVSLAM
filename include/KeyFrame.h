#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBExtractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>

namespace Goudan_SLAM
{
    class Map;
    class MapPoint;
    class Frame;
    class KeyFrameDatabase;

    class KeyFrame
    {
    public:
        KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);
        // KeyFrame(Frame &F, Map *pMap);

        // Pose functions
        void SetPose(const cv::Mat &Tcw);
        cv::Mat GetPose();
        cv::Mat GetPoseInverse();
        cv::Mat GetCameraCenter();
        cv::Mat GetRotation();
        cv::Mat GetTranslation();

        // 计算 bag of wards
        void ComputeBoW();

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

        // Spanning tree functions
        void AddChild(KeyFrame* pKF);
        void EraseChild(KeyFrame* pKF);
        void ChangeParent(KeyFrame* pKF);
        std::set<KeyFrame* > GetChilds();
        KeyFrame* GetParent();
        bool hasChild(KeyFrame *pKF);

        // LoopEdges
        void AddLoopEdge(KeyFrame* pKF);
        std::set<KeyFrame*> GetLoopEdges();

        // MapPoint observation functions
        void AddMapPoint(MapPoint* pMp, const size_t &idx);
        void EraseMapPointMatch(const size_t &idx);
        void EraseMapPointMatch(MapPoint* pMp);
        void ReplaceMapPointMatch(const size_t &idx, MapPoint* mMp);
        std::set<MapPoint*> GetMapPoints();
        std::vector<MapPoint*> GetMapPointMatches();
        int TrackedMapPoints(const int &minOns);
        MapPoint* GetMapPoint(const size_t &idx);
    
        // 关键帧函数
        std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;

        // Image
        bool IsInImage(const float &x, const float &y) const;

        // Enable/Disable bad flag changes
        void SetNotErase();
        void SetErase();

        // Set/check bad flag
        void SetBadFlag();
        bool isBad();

        // Compute Scene Depth (q=2 median). Used in monocular
        float ComputeSceneMedianDepth(const int q);

        static bool weightComp(int a, int b){
            return a>b;
        }

        static bool lId(KeyFrame* pKF1, KeyFrame* pKF2)
        {
            return pKF1->mnID<pKF2->mnID;
        }

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

        // 和Frame类中的定义相同
        const std::vector<cv::KeyPoint> mvKeys;
        const std::vector<cv::KeyPoint> mvKeysUn;
        const std::vector<float> mvuRight; // negative value for monocular points
        const std::vector<float> mvDepth; // negative value for monocular points
        const cv::Mat mDescriptors;

        // // BoW
        DBoW2::BowVector mBowVec;      ///< Vector of words to represent images
        DBoW2::FeatureVector mFeatVec; ///< Vector of nodes with indexes of local features

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
    protected:
        // SE3 Pose and camera center
        cv::Mat Tcw;
        cv::Mat Twc;
        cv::Mat Ow;

        cv::Mat Cw;

        std::vector<MapPoint*> mvpMapPoints;

        // Bow 
        KeyFrameDatabase* mpKeyFrameDB;
        ORBVocabulary* mpORBvocabulary;

        // 加速匹配的方式（类似桶排序）
        std::vector< std::vector <std::vector<size_t> > > mGrid;

        // 共视图
        std::map<KeyFrame*,int> mConnectedKeyFrameWeights; ///< 与该关键帧连接的关键帧与权重
        std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; ///< 排序后的关键帧
        std::vector<int> mvOrderedWeights; ///< 排序后的权重(从大到小)

        // Bad flags
        bool mbNotErase;
        bool mbToBeErased;
        bool mbBad;

        float mHalfBaseline;

        Map* mpMap;

        std::mutex mMutexPose;
        std::mutex mMutexConnections;
        std::mutex mMutexFeatures;
        
    };

}

#endif