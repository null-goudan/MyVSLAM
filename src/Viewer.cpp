#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>
#include <opencv2/core/core.hpp>


namespace Goudan_SLAM
{
    Viewer::Viewer(System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath) : mpSystem(pSystem), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpTracker(pTracking),
                                                                                                                                           mbFinishRequested(false), mbFinished(true), mbStopped(false), mbStopRequested(false)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        float fps = fSettings["Camera.fps"];
        if (fps < 1)
            fps = 30;
        mT = 1e3 / fps;

        mImageWidth = fSettings["Camera.width"];
        mImageHeight = fSettings["Camera.height"];
        if (mImageWidth < 1 || mImageHeight < 1)
        {
            mImageWidth = 640;
            mImageHeight = 480;
        }

        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];
    }

    void Viewer::Run()
    {
        mbFinished = false;

        pangolin::CreateWindowAndBind("Goudan-SLAM: Map Viewer", 1024, 768);
        // 启动深度测试，OpenGL只绘制最前面的一层，绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
        glEnable(GL_DEPTH_TEST);

        // 在OpenGL中使用颜色混合
        glEnable(GL_BLEND);
        // 选择混合选项
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // cout << "[Viewer] Create Buttons -----" <<endl;
        // 新建按钮和选择框，第一个参数为按钮的名字，第二个为默认状态，第三个为是否有选择框
        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
        pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false, true);
        pangolin::Var<bool> menuReset("menu.Reset", false, false);

        // 定义相机投影模型：ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
        // 定义观测方位向量：观测点位置：(mViewpointX mViewpointY mViewpointZ)
        //                观测目标位置：(0, 0, 0)
        //                观测的方位向量：(0.0,-1.0, 0.0)
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

        // 定义显示面板大小，o有左右两个面板，左边显示一些按钮，右边显示图形
        // 前两个参数（0.0, 1.0）表明宽度和面板纵向宽度和窗口大小相同
        // 中间两个参数（pangolin::Attach::Pix(175), 1.0）表明右边所有部分用于显示图形
        // 最后一个参数（-1024.0f/768.0f）为显示长宽比
        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        cv::namedWindow("Goudan-SLAM: Current Frame");

        bool bFollow = true;
        bool bLocalizationMode = false;

        // cout << "------ Start Show viewer -----" <<endl;
        while (1)
        {
            // 清除缓冲区中的当前可写的颜色缓冲 和 深度缓冲
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // cout << "Viewer: CurrentOpenGLCameraMatrix:" <<endl<<Twc<<endl;
            // 步骤1：得到最新的相机位姿
            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

            // 步骤2：根据相机的位姿调整视角
            // menuFollowCamera为按钮的状态，bFollow为真实的状态
            if (menuFollowCamera && bFollow)
            {
                s_cam.Follow(Twc);
            }
            else if (menuFollowCamera && !bFollow)
            {
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            }
            else if (!menuFollowCamera && bFollow)
            {
                bFollow = false;
            }

            if (menuLocalizationMode && !bLocalizationMode)
            {
                // mpSystem->ActivateLocalizationMode();
                bLocalizationMode = true;
            }
            else if (!menuLocalizationMode && bLocalizationMode)
            {
                // mpSystem->DeactivateLocalizationMode();
                bLocalizationMode = false;
            }

            d_cam.Activate(s_cam);
            // 步骤3：绘制地图和图像
            // 设置为白色，glClearColor(red, green, blue, alpha），数值范围(0, 1)
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            mpMapDrawer->DrawCurrentCamera(Twc);
            if (menuShowKeyFrames || menuShowGraph)
                mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
            if (menuShowPoints)
                mpMapDrawer->DrawMapPoints();

            pangolin::FinishFrame();

            cv::Mat im = mpFrameDrawer->DrawFrame();
            cv::imshow("Goudan-SLAM: Current Frame", im);
            cv::waitKey(mT);

            // if (menuReset)
            // {
            //     menuShowGraph = true;
            //     menuShowKeyFrames = true;
            //     menuShowPoints = true;
            //     menuLocalizationMode = false;
            //     if (bLocalizationMode)
            //         mpSystem->DeactivateLocalizationMode();
            //     bLocalizationMode = false;
            //     bFollow = true;
            //     menuFollowCamera = true;
            //     mpSystem->Reset();
            //     menuReset = false;
            // }

            if (Stop())
            {
                while (isStopped())
                {
                    // usleep(3000);
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
            }

            if (CheckFinish())
                break;
        }

        SetFinish();
    }

    void Viewer::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool Viewer::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Viewer::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool Viewer::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    void Viewer::RequestStop()
    {
        unique_lock<mutex> lock(mMutexStop);
        if (!mbStopped)
            mbStopRequested = true;
    }

    bool Viewer::isStopped()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    bool Viewer::Stop()
    {
        unique_lock<mutex> lock(mMutexStop);
        unique_lock<mutex> lock2(mMutexFinish);

        if (mbFinishRequested)
            return false;
        else if (mbStopRequested)
        {
            mbStopped = true;
            mbStopRequested = false;
            return true;
        }

        return false;
    }

    void Viewer::Release()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopped = false;
    }
}
