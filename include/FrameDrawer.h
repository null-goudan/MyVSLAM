#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <mutex>

namespace Goudan_SLAM{
    class Tracking;
    class Viewer;

    class FrameDrawer
    {
    public:
        FrameDrawer(Map *pMap);

        // 从上一个被处理的帧更新信息
        void Update(Tracking *pTracker);
        cv::Mat DrawFrame(); // 绘制最后一个处理的帧
    
    protected:

        void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

        // 某帧需要绘图的信息
        cv::Mat mIm;
        int N;
        vector<cv::KeyPoint> mvCurrentKeys;
        vector<bool> mvbMap, mvbVO;
        bool mbOnlyTracking;
        int mnTracked, mnTrackedVO;
        vector<cv::KeyPoint> mvIniKeys;
        vector<int> mvIniMatches;
        int mState;
        
        Map* mpMap;

        std::mutex mMutex;
    };
}

#endif