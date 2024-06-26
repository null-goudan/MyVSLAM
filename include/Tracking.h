#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "ORBExtractor.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "LocalMapping.h"
#include "LoopClosing.h"

#include <mutex>

namespace Goudan_SLAM{
    class Viewer;
    class FrameDrawer;
    class Map;
    class System;
    class LocalMapping;
    
    class Tracking{
    public:
        Tracking(System *pSys, ORBVocabulary* pVoc , FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,  
            KeyFrameDatabase *pKFDB, const std::string &strSettingPath);
        
        cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

        void SetLocalMapper(LocalMapping* pLocalMapper);
        void SetLoopClosing(LoopClosing* pLoopClosing);
        void SetViewer(Viewer* pViewer);

        void ChangeCalibration(const string &strSettingPath);
        void InformOnlyTracking(const bool &flag);
        // member
    public:
        enum eTrackingState{
            SYSTEM_NOT_READY=-1,
            NO_IMAGES_YET=0,
            NOT_INITIALIZED=1,
            OK=2,
            LOST=3
        };

        eTrackingState mState;
        eTrackingState mLastProcessedState;

        // 当前帧
        Frame mCurrentFrame;
        cv::Mat mImGray;

        // 记录相机整个轨迹以及最后执行的一些列表，最基础的是我们要记录每一帧的Reference Keyframe 
        // 并且和他相关的转换矩阵
        list<cv::Mat> mlRelativeFramePoses;
        list<KeyFrame*> mlpReferences;
        list<double> mlFrameTimes;
        list<bool> mlbLost;

        // 初始化相关变量 （单目相机）
        // 初始化时前两帧的相关变量
        std::vector<int> mvIniLastMatches;
        std::vector<int> mvIniMatches; // 跟踪初始化时的前两帧之间的匹配
        std::vector<cv::Point2f> mvbPrevMatched;
        std::vector<cv::Point3f> mvIniP3D;
        Frame mInitialFrame;


        bool mbOnlyTracking;        // 只定位模式
        void Reset();
    
    protected:
        void Track();

        void MonocularInitialization();
        void CreateInitialMapMonocular();

        void CheckReplacedInLastFrame();
        bool TrackReferenceKeyFrame();
        void UpdateLastFrame();
        bool TrackWithMotionModel();

        bool Relocalization();

        void UpdateLocalMap();
        void UpdateLocalPoints();
        void UpdateLocalKeyFrames();

        bool TrackLocalMap();
        void SearchLocalPoints();

        bool NeedNewKeyFrame();
        void CreateNewKeyFrame();

        bool mbVO;

        //Other Thread Pointers
        LocalMapping* mpLocalMapper;
        LoopClosing* mpLoopClosing;

        ORBExtractor* mpORBextractorLeft;
        ORBExtractor* mpIniORBextractor;

        // Bow
        ORBVocabulary* mpORBVocabulary;
        KeyFrameDatabase* mpKeyFrameDB;

        // 初始化器
        Initializer* mpInitializer;

        // Local Map
        KeyFrame* mpReferenceKF;
        std::vector<KeyFrame *> mvpLocalKeyFrames;
        std::vector<MapPoint *> mvpLocalMapPoints;

        // System
        System* mpSystem;

        // Drawers
        Viewer* mpViewer;
        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;

        // Map
        Map* mpMap;

        // 标定矩阵参数
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;

        // Threshold close/far points
        float mThDepth;

        // 新的滑动窗口范围?
        int mMinFrames;
        int mMaxFrames;

        // 现在已经匹配到的关键点的个数
        int mnMatchesInliers;

        // Last Frame, KeyFream and Reloaclisation Info
        KeyFrame* mpLastKeyFrame;
        Frame mLastFrame;
        unsigned int mnLastKeyFrameId;
        unsigned int mnLastRelocFrameId;

        //运动模型
        cv::Mat mVelocity;

        //Color order (true RGB, false BGR, ignored if grayscale)
        bool mbRGB;

        list<MapPoint*> mlpTemporalPoints;
    };
}

#endif