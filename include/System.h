#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"

using namespace std;

namespace Goudan_SLAM{
    class Tracking;

    class System{
    public:
        // :TODO
        System(const string &strSettingsFile);

        // Tracker
        cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);
    private:
        Tracking* mpTracker;
    };
}

#endif