#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
#include "Map.h"
#include "MapDrawer.h"
#include "FrameDrawer.h"
#include "Viewer.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"

using namespace std;

namespace Goudan_SLAM
{
    class Tracking;
    class Viewer;
    class FrameDrawer;
    class Map;
    class LocalMapping;
    class LoopClosing;

    class System
    {
    public:
        System(const string &strVocFile, const string &strSettingsFile);

        // Tracker
        cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

        // This stops local mapping thread (map building) and performs only camera tracking.
        void ActivateLocalizationMode();
        // This resumes local mapping thread and performs SLAM again.
        void DeactivateLocalizationMode();

        void Reset();

        void Shutdown();

        void SaveTrajectoryKITTI(const string &filename);

    private:
        ORBVocabulary *mpVocabulary;
        KeyFrameDatabase *mpKeyFrameDatabase;

        Map *mpMap;

        Tracking *mpTracker;

        LocalMapping *mpLocalMapper;

        LoopClosing* mpLoopCloser;

        Viewer *mpViewer;

        FrameDrawer *mpFrameDrawer;
        MapDrawer *mpMapDrawer;

        std::thread *mptViewer;
        std::thread* mptLoopClosing;
        std::thread *mptLocalMapping;

        // Reset flag
        std::mutex mMutexReset;
        bool mbReset;

        // Change mode flags
        std::mutex mMutexMode;
        bool mbActivateLocalizationMode;
        bool mbDeactivateLocalizationMode;
    };
}

#endif