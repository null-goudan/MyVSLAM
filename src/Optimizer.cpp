#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace Goudan_SLAM{
    int Optimizer::PoseOptimization(Frame *pFrame)
    {
        // 该优化函数主要用于Tracking线程中
        // 构造g2o优化器
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // 添加顶点: 待优化当前帧的Tcw
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        const int N = pFrame->N;

        // 单目边
        vector<g2o::EdgeSE3ProjectXYZOnlyPose* > vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        const float deltaMono = sqrt(5.991);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);
            for(int i = 0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;

                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
        if(nInitialCorrespondences<3)
            return 0;

        // 开始优化，优化四次， 每次优化后，将观测分为outlier和inliner, outliner 不参与下次优化
        // 由于每次优化之后是对所有的观测进行outliner 和 inlier的判别，因此之前被判别为outlier有可能变成inlier，反之亦然
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        // const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
        const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

        int nBad = 0;
        for(size_t it = 0; it<4; it++)
        {
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0); //对level为0的边进行优化
            optimizer.optimize(its[it]);

            nBad = 0;
            for(size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];
                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();   //卡方检验

                if(chi2 > chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);     //设置outliner
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);     //设置inliner 
                }

                if(it==2)
                    e->setRobustKernel(0);  // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
            }

            if(optimizer.edges().size()<10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences-nBad;
    }
}