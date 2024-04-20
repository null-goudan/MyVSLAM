#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "Map.h"
#include "MapDrawer.h"
#include "FrameDrawer.h"
#include "Viewer.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"

using namespace std;

namespace Goudan_SLAM{
    class Tracking;
    class Viewer;
    class FrameDrawer;
    class Map;

    class System{
    public:
        System(const string &strVocFile, const string &strSettingsFile);

        // Tracker
        cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);
    private:
    
        ORBVocabulary* mpVocabulary;
        KeyFrameDatabase* mpKeyFrameDatabase;

        Map* mpMap;

        Tracking* mpTracker;

        Viewer* mpViewer;

        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;


        std::thread* mptViewer;
    };
}

#endif