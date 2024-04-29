#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>

namespace Goudan_SLAM
{
    class MapPoint;
    class KeyFrame;

    class Map
    {
    public:
        Map();

        void AddKeyFrame(KeyFrame *pKF);
        void AddMapPoint(MapPoint *pMP);
        void EraseMapPoint(MapPoint *pMP);
        void EraseKeyFrame(KeyFrame *pKF);
        void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

        std::vector<KeyFrame *> GetAllKeyFrames();
        std::vector<MapPoint *> GetAllMapPoints();
        std::vector<MapPoint *> GetReferenceMapPoints();

        long unsigned int MapPointsInMap();
        long unsigned KeyFramesInMap();

        long unsigned int GetMaxKFid();

        void clear();

        vector<KeyFrame *> mvpKeyFrameOrigins;

        std::mutex mMutexMapUpdate;
        std::mutex mMutexPointCreation;

    protected:
        std::set<MapPoint *> mspMapPoints; ///< MapPoints
        std::set<KeyFrame *> mspKeyFrames; ///< Keyframs

        std::vector<MapPoint *> mvpReferenceMapPoints;

        long unsigned int mnMaxKFid;

        std::mutex mMutexMap;
    };
}

#endif