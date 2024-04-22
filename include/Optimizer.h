#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace Goudan_SLAM{
    // 利用优化的算法
    class Optimizer
    {
    public:
        // 重投影误差法解决 3D-2D问题 
        int static PoseOptimization(Frame* pFrame);
    };
}

#endif