#include "LocalMapping.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>


namespace Goudan_SLAM{
    LocalMapping::LocalMapping(Map *pMap): mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
    {
    }

}