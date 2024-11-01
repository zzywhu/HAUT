#pragma once
#include "common.h"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unordered_map>
#include <pcl/point_types.h>
#include <cmath>
#include"loadconfig.h"
////////////////////////////////////////////////
#include <pcl/point_types.h>
#include <pcl/common/common.h>
///////////////////////////////////////////////


#define calib
#define online

#include <random>
////////////////////////////////////////////////////////global param
extern int IterNum;
extern int dataProcessingNum;
////////////////////////////////////////////////////////
std::default_random_engine generator;
cv::Vec3b randomRGBColor()
{
    int minValue = 1, maxValue = 256;
    cv::Vec3b Color(generator(), generator(), generator());
    for (int i = 0; i < 3; ++i)
    {
        Color[i] *= (maxValue - minValue);
        Color[i] += minValue;
    }
    return Color;
}

void saveRGBPointCloud(std::string name, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc)
{
    pcl::io::savePCDFileASCII(name, *pc);
}

void FitPlaneSVD(Plane& single_plane)//用整体点云直接提平面，可能会遗留后患！！！
{
    // 1、计算质心	// 2、去质心
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_buf(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud(single_plane.cloud, *cloud_buf);
    Eigen::MatrixXf cloudMat = cloud_buf->getMatrixXfMap(3, 4, 0);
    Eigen::VectorXf centroid(cloudMat.rows());
    for (int i = 0; i < cloudMat.rows(); i++)
    {
        centroid(i) = cloudMat.row(i).mean();
        cloudMat.row(i) = cloudMat.row(i) - (Eigen::VectorXf::Ones(cloudMat.cols()).transpose() * centroid(i));
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cloudMat, Eigen::ComputeFullU);
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::VectorXf S = svd.singularValues();

    // std::cout<<std::endl<<U.col(2)<<std::endl;//model.coefficients.topRows(3).dot(centroid);
    // std::cout<<U.col(2).dot(centroid)<<std::endl;
    // std::cout<<single_plane.p_center<<std::endl;
    // std::cout<<centroid<<std::endl;

    single_plane.normal[0] = U.col(2)[0];
    single_plane.normal[1] = U.col(2)[1];
    single_plane.normal[2] = U.col(2)[2];
    single_plane.p_center.getVector3fMap() = centroid;
}


void mergePlanes(std::vector<Plane>& PlaneList)
{
    int CurrentLength = PlaneList.size();
    int BaseIndex = 0;

    while (1)
    {
        CurrentLength = PlaneList.size();
        if (BaseIndex >= CurrentLength)
        {
            break;
        }
        Eigen::Vector3d normal = PlaneList[BaseIndex].normal;
        pcl::PointXYZRGB center_PCL = PlaneList[BaseIndex].p_center;
        Eigen::Vector3d center(center_PCL.x, center_PCL.y, center_PCL.z);
        PlaneList[BaseIndex].index = BaseIndex;

        int key = 0;
        while (1)
        {
            if (key + BaseIndex + 1 >= CurrentLength)
                break;
            Eigen::Vector3d normal_t = PlaneList[key + BaseIndex + 1].normal;
            pcl::PointXYZRGB center_tPCL = PlaneList[key + BaseIndex + 1].p_center;
            Eigen::Vector3d center_t(center_tPCL.x, center_tPCL.y, center_tPCL.z);

            float D_t = -(normal_t.dot(center_t));//second plane ABCD
            float cosAngle = abs(normal.dot(normal_t));
            float Distance = abs(normal_t.dot(center) + D_t);

            if ((cosAngle > 0.7) && (Distance < 0.05))//0.1
            {
                pcl::copyPointCloud(PlaneList[BaseIndex].cloud + PlaneList[key + BaseIndex + 1].cloud, PlaneList[BaseIndex].cloud);
                FitPlaneSVD(PlaneList[BaseIndex]);
                PlaneList.erase(PlaneList.begin() + key + BaseIndex + 1);
                // cout<<"fit"<<BaseIndex<<" "<<key + BaseIndex + 1;
                CurrentLength = PlaneList.size();
                // PlaneList[index]
            }
            else
            {
                // cout<<"pass"<<BaseIndex<<" "<<key + BaseIndex + 1;
                key++;
            }
        }
        PlaneList[BaseIndex].index = BaseIndex;
        BaseIndex++;
    }
}


void savePlanes(std::vector<Plane> plane_list, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc)
{
    for (int i = 0; i < plane_list.size(); i++)
    {
        std::vector<unsigned int> colors;
        colors.push_back(static_cast<unsigned int>(rand() % 256));
        colors.push_back(static_cast<unsigned int>(rand() % 256));
        colors.push_back(static_cast<unsigned int>(rand() % 256));
        for (size_t j = 0; j < plane_list[i].cloud.points.size(); j++)
        {
            //plane_list[i].cloud.points[j].g
            pcl::PointXYZRGB p;
            p.getArray3fMap() = plane_list[i].cloud.points[j].getArray3fMap();
            p.r = colors[0];
            p.g = colors[1];
            p.b = colors[2];
            pc->push_back(p);
        }
    }
}

//////////////////////////////////////////////////////////
class Calibration {
public:
    enum ProjectionType { DEPTH, INTENSITY, BOTH };
    enum Direction { UP, DOWN, LEFT, RIGHT };

    int rgb_edge_minLen_ = 200;
    int rgb_canny_threshold_ = 20;
    int min_depth_ = 0.1;
    int max_depth_ = 50;
    int plane_max_size_ = 5;
    float detect_line_threshold_ = 0.02;
    int line_number_ = 0;
    int color_intensity_threshold_ = 5;
    Eigen::Vector3d adjust_euler_angle_;
    Calibration(const std::string& image_file, const std::string& pcd_file, const std::string& calib_config_file, vector<double> camera_matrix, vector<double>dist_coeffs);

    void loadImgAndPointcloud(const std::string bag_path, pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_cloud, cv::Mat& rgb_img);

    bool loadCameraConfig(const std::string& camera_file);

    bool loadCalibConfig(const std::string& config_file);

    bool loadConfig(const std::string& configFile);

    bool checkFov(const cv::Point2d& p);

    void colorCloud(const Vector6d& extrinsic_params, const int density, const cv::Mat& rgb_img, const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color_cloud);

    void edgeDetector(const int& canny_threshold, const int& edge_threshold, const cv::Mat& src_img, cv::Mat& edge_img,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_cloud);

    void projection(const Vector6d& extrinsic_params, const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
        const ProjectionType projection_type, const bool is_fill_img,
        cv::Mat& projection_img);

    void calcLine(const std::vector<Plane>& plane_list, const double voxel_size, const Eigen::Vector3d origin,
        std::vector<pcl::PointCloud<pcl::PointXYZI>>& line_cloud_list);
    void calcLineLin(std::vector<Plane>& plane_list, std::vector<pcl::PointCloud<pcl::PointXYZI>>& line_cloud_list);

    cv::Mat fillImg(const cv::Mat& input_img, const Direction first_direct, const Direction second_direct);

    void buildPnp(const Vector6d& extrinsic_params, const int dis_threshold, const bool show_residual,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cam_edge_cloud_2d,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
        std::vector<PnPData>& pnp_list);
    void
        buildVPnp(const Vector6d& extrinsic_params, const int dis_threshold, const bool show_residual,
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cam_edge_cloud_2d,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
            std::vector<VPnPData>& pnp_list);

    cv::Mat
        getConnectImg(const int dis_threshold, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_edge_cloud,
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& depth_edge_cloud);

    cv::Mat getProjectionImg(const Vector6d& extrinsic_params);

    void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        const float voxel_size,
        std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map);

    void LiDAREdgeExtraction(const std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map, const float ransac_dis_thre, const int plane_size_threshold,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d);

    void calcDirection(const std::vector<Eigen::Vector2d>& points, Eigen::Vector2d& direction);

    void calcResidual(const Vector6d& extrinsic_params, const std::vector<VPnPData> vpnp_list,
        std::vector<float>& residual_list);

    void calcCovarance(const Vector6d& extrinsic_params, const VPnPData& vpnp_point, const float pixel_inc,
        const float range_inc, const float degree_inc, Eigen::Matrix2f& covarance);

    // 相机内参
    float fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_, s_;
    int width_, height_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Mat init_extrinsic_;

    int is_use_custom_msg_;
    float voxel_size_ = 1.0;
    // float down_sample_size_ = 0.02;
    float ransac_dis_threshold_ = 0.02;
    float plane_size_threshold_ = 60;
    float theta_min_;
    float theta_max_;
    float direction_theta_min_;
    float direction_theta_max_;
    float SearchForPointDis_ = 0.03;
    // float min_line_dis_threshold_ = 0.03;
    // float max_line_dis_threshold_ = 0.06;

    cv::Mat rgb_image_;
    cv::Mat image_;
    cv::Mat grey_image_;
    // 裁剪后的灰度图像
    cv::Mat cut_grey_image_;

    // 初始旋转矩阵
    Eigen::Matrix3d init_rotation_matrix_;
    // 初始平移向量
    Eigen::Vector3d init_translation_vector_;

    // 存储从pcd/bag处获取的原始点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud_;

    // 存储平面交接点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
    std::vector<int> plane_line_number_;
    // 存储RGB图像边缘点的2D点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_egde_cloud_;
    // 存储LiDAR Depth/Intensity图像边缘点的2D点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr lidar_edge_cloud_;
};
void ConvertRGB2GRAY(const cv::Mat& image, cv::Mat& imageGray)
{
    if (!image.data || image.channels() != 3)
    {
        return;
    }
    //创建一张单通道的灰度图像
    imageGray = cv::Mat::zeros(image.size(), CV_8UC1);
    //取出存储图像像素的数组的指针
    uchar* pointImage = image.data;
    uchar* pointImageGray = imageGray.data;
    //取出图像每行所占的字节数
    size_t stepImage = image.step;
    size_t stepImageGray = imageGray.step;


    for (int i = 0; i < imageGray.rows; i++)
    {
        for (int j = 0; j < imageGray.cols; j++)
        {
            //opencv的通道顺序是BGR，而不是我们常说的RGB顺序
            if (pointImage[i * stepImage + 3 * j] > pointImage[i * stepImage + 3 * j + 1] && pointImage[i * stepImage + 3 * j] > pointImage[i * stepImage + 3 * j + 2])
            {
                pointImageGray[i * stepImageGray + j] = 255;
            }
            if (pointImage[i * stepImage + 3 * j + 1] > pointImage[i * stepImage + 3 * j] && pointImage[i * stepImage + 3 * j + 1] > pointImage[i * stepImage + 3 * j + 2])
            {
                pointImageGray[i * stepImageGray + j] = 125;
            }
            if (pointImage[i * stepImage + 3 * j + 2] > pointImage[i * stepImage + 3 * j] && pointImage[i * stepImage + 3 * j + 2] > pointImage[i * stepImage + 3 * j + 1])
            {
                pointImageGray[i * stepImageGray + j] = 0;
            }
            /*pointImageGray[i * stepImageGray + j] =
                (uchar)(0.333 * pointImage[i * stepImage + 3 * j] +
                    0.333 * pointImage[i * stepImage + 3 * j + 1] +
                    0.333 * pointImage[i * stepImage + 3 * j + 2]);*/
        }
    }
}
Calibration::Calibration(const std::string& image_file,
    const std::string& pcd_file,
    const std::string& calib_config_file, vector<double> camera_matrix, vector<double>dist_coeffs)
{
    loadCameraConfig(calib_config_file);
    /*fx_ = camera_matrix[0];
    cx_ = camera_matrix[2];
    fy_ = camera_matrix[4];
    cy_ = camera_matrix[5];
    k1_ = dist_coeffs[0];
    k2_ = dist_coeffs[1];
    p1_ = dist_coeffs[2];
    p2_ = dist_coeffs[3];
    k3_ = dist_coeffs[4];*/
    int isdist;
    string configfile2= "param\\config_multi.yaml";
    //cv::FileStorage fSettings(calib_config_file2, cv::FileStorage::READ);
    isdist = parseIntFromYAML(configfile2, "undist"); ;       //canny param 10
    cv::Mat k1 = cv::Mat::zeros(3, 3, CV_32F); cv::Mat k2 = cv::Mat::zeros(3, 3, CV_32F); cv::Mat c1 = cv::Mat::zeros(4, 1, CV_32F);
    k1.at<float>(0, 0) = fx_; k1.at<float>(0, 1) = 0; k1.at<float>(0, 2) = cx_;
    k1.at<float>(1, 0) = 0; k1.at<float>(1, 1) = fy_; k1.at<float>(1, 2) = cy_;
    k1.at<float>(2, 0) = 0; k1.at<float>(2, 1) = 0; k1.at<float>(2, 2) = 1;
    k2.at<float>(0, 0) = fx_*0.8; k2.at<float>(0, 1) = 0; k2.at<float>(0, 2) = cx_;
    k2.at<float>(1, 0) = 0; k2.at<float>(1, 1) = fy_*0.8; k2.at<float>(1, 2) = cy_;
    k2.at<float>(2, 0) = 0; k2.at<float>(2, 1) = 0; k2.at<float>(2, 2) = 1;
    c1.at<float>(0) = k1_; c1.at<float>(1) = k2_; c1.at<float>(2) = p1_; c1.at<float>(3) = p2_; //c1.at<float>(4) = k3_;
    loadCalibConfig(calib_config_file);//load the config_indoor.yaml, including initial extrinsic parameters
    cv::Mat image_before = cv::imread(image_file, cv::IMREAD_UNCHANGED);
    if (isdist == 0)
    {
        k2 = k1;
        image_ = image_before;
    }
    if (isdist == 1)
    {
        cv::fisheye::undistortImage(image_before, image_, k1, c1, k2);
    }
    if (isdist == 2)
    {
        k2.at<float>(0, 0) = fx_;
        k2.at<float>(1, 1) = fy_;
        undistort(image_before, image_, k1, c1, k2);
    }
    fx_ = k2.at<float>(0,0);
    cx_ = k2.at<float>(0, 2);
    fy_ = k2.at<float>(1, 1);
    cy_ = k2.at<float>(1, 2);
    k1_ = 0;
    k2_ = 0;
    p1_ = 0;
    p2_ = 0;
    k3_ = 0;
    cv::imwrite("check/" + std::to_string(dataProcessingNum) + "_dist.png", image_);
    if (!image_.data)
    {
        std::string msg = "Can not load image from " + image_file;
        std::cout << msg << std::endl;
        exit(-1);
    }
    else
    {
        std::string msg = "Sucessfully load image!";
        std::cout << msg << std::endl;
        // cv::imshow("window",image_);
        // cv::waitKey(0);
    }
    width_ = image_.cols;
    height_ = image_.rows;
    // check rgb or gray
    // std::cout<< image_.type() <<std::endl;
    if (image_.type() == CV_8UC1)
    {
        grey_image_ = image_;
    }
    else if (image_.type() == CV_8UC3)
    {
        ConvertRGB2GRAY(image_, grey_image_);
        //cv::cvtColor(image_, grey_image_, cv::COLOR_BGR2GRAY);
    }
    else
    {
        std::string msg = "Unsupported image type, please use CV_8UC3 or CV_8UC1";
        exit(-1);
    }
    // if (image_.type() == CV_8UC1)
    // {
    //   grey_image_ = image_;
    // } 
    // else
    // {
    //   cv::cvtColor(image_, grey_image_, cv::COLOR_BGR2GRAY);
    // }


    // cv::imshow("window",grey_image_);
    // cv::waitKey(0);

    cv::Mat edge_image;

    edgeDetector(rgb_canny_threshold_, rgb_edge_minLen_, grey_image_, edge_image, rgb_egde_cloud_);
    //edit to show the edges in images

    std::string msg = "Sucessfully extract edge from image, edge size:" + std::to_string(rgb_egde_cloud_->size());
    std::cout << msg << std::endl;

    raw_lidar_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    if (!pcl::io::loadPCDFile(pcd_file, *raw_lidar_cloud_))
    {
        // down_sampling_voxel(*raw_lidar_cloud_, 0.02);
        std::string msg = "Sucessfully load pcd, pointcloud size: " + std::to_string(raw_lidar_cloud_->size());
        std::cout << msg << std::endl;
    }
    else
    {
        std::string msg = "Unable to load " + pcd_file;
        std::cout << msg << std::endl;
        exit(-1);
    }

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudA;

    // visualization
    // pcl::visualization::PCLVisualizer visualizer;
    // visualizer.addPointCloud<pcl::PointXYZI> (raw_lidar_cloud_, "source cloud transformed");
    // while (!visualizer.wasStopped ()) 
    // {
    //     visualizer.spinOnce ();
    //     pcl_sleep(0.01);
    // };

    // Eigen::Vector3d lwh(50, 50, 30);//what are those used for ?
    // Eigen::Vector3d origin(0, -25, -10);
    // std::vector<VoxelGrid> voxel_list;
    std::unordered_map<VOXEL_LOC, Voxel*> voxel_map;
    initVoxel(raw_lidar_cloud_, voxel_size_, voxel_map);//voxel_size_=1.0m
    //这里希望保存不同voxel不同颜色的点云
    LiDAREdgeExtraction(voxel_map, ransac_dis_threshold_, plane_size_threshold_, plane_line_cloud_);
    //Ransac.dis_threshold: 0.015; Plane.max_size: 5
    plane_line_cloud_->height = 1;
    plane_line_cloud_->width = plane_line_cloud_->size();

    pcl::io::savePCDFileASCII("check/" + std::to_string(dataProcessingNum) + "_laserEdges.pcd", *plane_line_cloud_);//(“plane_line_cloud.pcd”, plane_line_cloud_);
    dataProcessingNum++;
};

bool Calibration::loadCameraConfig(const std::string& camera_file)
{
    //cv::FileStorage cameraSettings(camera_file, cv::FileStorage::READ);
    /*if (!cameraSettings.isOpened())
    {
        std::cerr << "Failed to open camera settings file at " << camera_file
            << std::endl;
        exit(-1);
    }*/
   /* cameraSettings["CameraMat"] >> camera_matrix_;
    cameraSettings["DistCoeffs"] >> dist_coeffs_;*/
    camera_matrix_ = parseMatrixFromYAML(camera_file, "CameraMat");
    cout << camera_matrix_ << endl;
    dist_coeffs_ = parseMatrixFromYAML(camera_file, "DistCoeffs");
    fx_ = camera_matrix_.at<double>(0, 0);
    cx_ = camera_matrix_.at<double>(0, 2);
    fy_ = camera_matrix_.at<double>(1, 1);
    cy_ = camera_matrix_.at<double>(1, 2);
    k1_ = dist_coeffs_.at<double>(0, 0);
    k2_ = dist_coeffs_.at<double>(0, 1);
    p1_ = dist_coeffs_.at<double>(0, 2);
    p2_ = dist_coeffs_.at<double>(0, 3);
    k3_ = dist_coeffs_.at<double>(0, 4);
    std::cout << "Camera Matrix: " << std::endl << camera_matrix_ << std::endl;
    std::cout << "Distortion Coeffs: " << std::endl << dist_coeffs_ << std::endl;
    return true;
};

bool Calibration::loadCalibConfig(const std::string& config_file)
{
    /*cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
    if (!fSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << config_file
            << std::endl;
        exit(-1);
    }
    else {}*/
    init_extrinsic_ = parseMatrixFromYAML(config_file, "ExtrinsicMat");
    init_rotation_matrix_ << init_extrinsic_.at<double>(0, 0),
        init_extrinsic_.at<double>(0, 1), init_extrinsic_.at<double>(0, 2),
        init_extrinsic_.at<double>(1, 0), init_extrinsic_.at<double>(1, 1),
        init_extrinsic_.at<double>(1, 2), init_extrinsic_.at<double>(2, 0),
        init_extrinsic_.at<double>(2, 1), init_extrinsic_.at<double>(2, 2);
    init_translation_vector_ << init_extrinsic_.at<double>(0, 3),
        init_extrinsic_.at<double>(1, 3), init_extrinsic_.at<double>(2, 3);
    rgb_canny_threshold_ = parseDoubleFromYAML(config_file, "Canny.gray_threshold");       //canny param 10
    rgb_edge_minLen_ = parseDoubleFromYAML(config_file, "Canny.len_threshold");            //edge in image minlength 200
    voxel_size_ = parseDoubleFromYAML(config_file, "Voxel.size");                          //voxel size 0.5
    // down_sample_size_ = fSettings["Voxel.down_sample_size"];        //downsample to 0.02, may be not necessary
    plane_size_threshold_ = parseDoubleFromYAML(config_file, "Plane.min_points_size");     //min points of a plane 30
    plane_max_size_ = parseDoubleFromYAML(config_file, "Plane.max_size");                  //??????????
    ransac_dis_threshold_ = parseDoubleFromYAML(config_file, "Ransac.dis_threshold");      //ransac distance threshold 0.02//small and strange
    // min_line_dis_threshold_ = fSettings["Edge.min_dis_threshold"];  //edge in point cloud minlength 0.03
    // max_line_dis_threshold_ = fSettings["Edge.max_dis_threshold"];  //edge in point cloud maxlength 0.06
    SearchForPointDis_ = parseDoubleFromYAML(config_file, "Edge.SearchForPointDis");
    theta_min_ = parseDoubleFromYAML(config_file, "Plane.normal_theta_min");               //may be the intersaction detection---min
    theta_max_ = parseDoubleFromYAML(config_file, "Plane.normal_theta_max");               //may be the intersaction detection---max
    theta_min_ = cos(DEG2RAD(theta_min_));//useful!!
    theta_max_ = cos(DEG2RAD(theta_max_));
    direction_theta_min_ = cos(DEG2RAD(30.0));
    direction_theta_max_ = cos(DEG2RAD(150.0));
    color_intensity_threshold_ = parseDoubleFromYAML(config_file, "Color.intensity_threshold");
    return true;
};

// Color the point cloud by rgb image using given extrinsic
void Calibration::colorCloud(
    const Vector6d& extrinsic_params, const int density,
    const cv::Mat& input_image,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color_cloud)
{
    cv::Mat rgb_img;
    if (input_image.type() == CV_8UC1)
    {
        cv::cvtColor(input_image, rgb_img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        rgb_img = input_image;
    }
    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < lidar_cloud->size(); i += density)
    {
        pcl::PointXYZI point = lidar_cloud->points[i];
        // float depth = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
        // // if (depth > 2 && depth < 50 && point.intensity >= color_intensity_threshold_)
        // //Color.intensity_threshold: 10//why 2 < depth < 50
        // if (depth > 2 && depth < 50)
        // {
        //   pts_3d.emplace_back(cv::Point3f(point.x, point.y, point.z));//类似push_back
        // }
        pts_3d.emplace_back(cv::Point3f(point.x, point.y, point.z));//类似push_back
    }
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    //三个欧拉角转轴角，然后相乘，似乎是转成旋转向量(李代数)

    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
    cv::Mat r_vec =
        (cv::Mat_<double>(3, 1) <<
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    //奇怪的数学表达//是适用于opencv的Rodrigues rotation vector//李代数与这个什么关系呢？

    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3], extrinsic_params[4], extrinsic_params[5]);
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff, pts_2d);
    //Projects 3D points to an image plane. 
    //pts_2d: Output array of image points, 1xN/Nx1 2-channel, or vector<Point2f>
    int image_rows = rgb_img.rows;
    int image_cols = rgb_img.cols;
    color_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < pts_2d.size(); i++)
    {
        if (pts_2d[i].x >= 0 && pts_2d[i].x < image_cols && pts_2d[i].y >= 0 && pts_2d[i].y < image_rows)
        {
            cv::Scalar color = rgb_img.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x);
            if (color[0] == 0 && color[1] == 0 && color[2] == 0)
            {
                continue;
            }
            // if (pts_3d[i].x > 100)// this is no any need, since depth < 50
            // {
            //   continue;
            // }
            pcl::PointXYZRGB p;
            p.x = pts_3d[i].x;
            p.y = pts_3d[i].y;
            p.z = pts_3d[i].z;
            // p.a = 255;
            p.b = color[0];
            p.g = color[1];
            p.r = color[2];
            color_cloud->points.push_back(p);
        }
    }
    color_cloud->width = color_cloud->points.size();
    color_cloud->height = 1;
}
void myUndistortPoints(const std::vector<cv::Point2d>& src, std::vector<cv::Point2d>& dst,
    const cv::Mat& cameraMatrix, const cv::Mat& distortionCoeff)
{

    dst.clear();
    double fx = cameraMatrix.at<float>(0, 0);
    double fy = cameraMatrix.at<float>(1, 1);
    double ux = cameraMatrix.at<float>(0, 2);
    double uy = cameraMatrix.at<float>(1, 2);

    double k1 = distortionCoeff.at<float>(0, 0);
    double k2 = distortionCoeff.at<float>(1, 0);
    double p1 = distortionCoeff.at<float>(2, 0);
    double p2 = distortionCoeff.at<float>(3, 0);
    double k3 = distortionCoeff.at<float>(4, 0);
    double k4=0;
    double k5=0;
    double k6=0;

    for (unsigned int i = 0; i < src.size(); i++)
    {
        const cv::Point2d& p = src[i];

        //首先进行坐标转换；
        double xDistortion = (p.x - ux) / fx;
        double yDistortion = (p.y - uy) / fy;

        double xCorrected, yCorrected;

        double x0 = xDistortion;
        double y0 = yDistortion;

        //这里使用迭代的方式进行求解，因为根据2中的公式直接求解是困难的，所以通过设定初值进行迭代，这也是OpenCV的求解策略；
        for (int j = 0; j < 50; j++)
        {
            double r2 = xDistortion * xDistortion + yDistortion * yDistortion;

            double distRadialA = 1 / (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
            double distRadialB = 1. + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;

            double deltaX = 2. * p1 * xDistortion * yDistortion + p2 * (r2 + 2. * xDistortion * xDistortion);
            double deltaY = p1 * (r2 + 2. * yDistortion * yDistortion) + 2. * p2 * xDistortion * yDistortion;

            xCorrected = (x0 - deltaX) * distRadialA * distRadialB;
            yCorrected = (y0 - deltaY) * distRadialA * distRadialB;

            xDistortion = xCorrected;
            yDistortion = yCorrected;
        }

        //进行坐标变换；
        xCorrected = xCorrected * fx + ux;
        yCorrected = yCorrected * fy + uy;

        dst.push_back(cv::Point2d(xCorrected, yCorrected));
    }

}

// Detect edge by canny, and filter by edge length
void Calibration::edgeDetector(const int& canny_threshold, const int& edge_threshold, const cv::Mat& src_img, cv::Mat& edge_img, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_cloud)
{
    
    //c1.at<float>(0) = 0; c1.at<float>(1) =0; c1.at<float>(2) =0; c1.at<float>(3) =0; c1.at<float>(4) = 0;
    int gaussian_size = 5;
    cv::imwrite("check/" + std::to_string(dataProcessingNum) + "_gray.png", src_img);
    cv::GaussianBlur(src_img, src_img, cv::Size(gaussian_size, gaussian_size), 0, 0);//gaussian_size=5
    //Gaussian blur is a usual image denoising technic, use the Gaussian fuction to convolute the image to denoise and decrease the details
    cv::imwrite("check/" + std::to_string(dataProcessingNum) + "_GaussianBlur.png", src_img);
    cv::Mat canny_result = cv::Mat::zeros(height_, width_, CV_8UC1);
    cv::Canny(src_img, canny_result, canny_threshold, canny_threshold * 3, 3, true);//canny_threshold=20, this is a threshold for pixel gradient from Sobel
    //https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
    cv::imwrite("check/" + std::to_string(dataProcessingNum) + "_canny.png", canny_result);
    edge_img = cv::Mat::zeros(height_, width_, CV_8UC3);

    edge_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < height_; i++)
    {
        for (size_t j = 0; j < width_; j++)
        {
            if (canny_result.at<uchar>(i, j) == 255)
            {
                cv::Vec3b RGBColor = randomRGBColor();
                edge_img.at<cv::Vec3b>(i, j) = RGBColor;
            }

        }
    }

    // cv::imshow("window",edge_img);
    // cv::imwrite("effective_edge.png",edge_img);
    // cv::waitKey(0);

    for (int x = 0; x < edge_img.cols; x++)
    {
        for (int y = 0; y < edge_img.rows; y++)
        {
            if ((edge_img.at<cv::Vec3b>(y, x)[0] != 0) || (edge_img.at<cv::Vec3b>(y, x)[1] != 0) || (edge_img.at<cv::Vec3b>(y, x)[2] != 0))
            {
                pcl::PointXYZRGB p;
                /*vector<cv::Point2d> p1;
                vector<cv::Point2d> p2;
                p1.push_back(cv::Point2d(x, y));
                myUndistortPoints(p1, p2, k1, c1);
                double x_d = p2[0].x;
                double y_d = p2[0].y;
                if (x_d<1 || x_d> edge_img.cols - 1 || y_d<1 || y_d> edge_img.rows - 1)
                {
                    continue;
                }*/

                p.x = x;
                p.y = -y;
                p.z = 0;
                p.b = edge_img.at<cv::Vec3b>(y, x)[0];
                p.g = edge_img.at<cv::Vec3b>(y, x)[1];
                p.r = edge_img.at<cv::Vec3b>(y, x)[2];
                edge_cloud->points.push_back(p);
            }
        }
    }

    edge_cloud->width = edge_cloud->points.size();
    edge_cloud->height = 1;
    saveRGBPointCloud("check/" + std::to_string(dataProcessingNum) + "_ImgEdges.pcd", edge_cloud);
    // cv::imshow("canny result", canny_result);////////////////////////////////////////////////////////////////////////////
    // cv::imshow("edge result", edge_img);
    // cv::waitKey();
}
// void Calibration::edgeDetector(const int &canny_threshold, const int &edge_threshold, const cv::Mat &src_img, cv::Mat &edge_img, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &edge_cloud) 
// {
//   int gaussian_size = 5;

//   // cv::imshow("window",src_img);
//   cv::imwrite("check/gray.png",src_img);
//   // cv::waitKey(0);

//   cv::GaussianBlur(src_img, src_img, cv::Size(gaussian_size, gaussian_size), 0, 0);//gaussian_size=5
//   //Gaussian blur is a usual image denoising technic, use the Gaussian fuction to convolute the image to denoise and decrease the details

//   // cv::imshow("window",src_img);
//   cv::imwrite("check/GaussianBlur.png",src_img);
//   // cv::waitKey(0);

//   cv::Mat canny_result = cv::Mat::zeros(height_, width_, CV_8UC1);
//   cv::Canny(src_img, canny_result, canny_threshold, canny_threshold * 3, 3, true);//canny_threshold=20, this is a threshold for pixel gradient from Sobel
//   //https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html

//   // cv::imshow("window",canny_result);
//   cv::imwrite("check/canny.png",canny_result);
//   // cv::waitKey(0);

//   std::vector<std::vector<cv::Point>> contours;//轮廓
//   std::vector<cv::Vec4i> hierarchy;//层次结构
//   cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
//   //contours -- Detected contours. Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >)
//   //hierarchy -- Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology
//   edge_img = cv::Mat::zeros(height_, width_, CV_8UC3);

//   edge_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

//   for (size_t i = 0; i < contours.size(); i++)
//   {
//     if (contours[i].size() > edge_threshold)//200
//     {
//       cv::Mat debug_img = cv::Mat::zeros(height_, width_, CV_8UC3);

//       cv::Vec3b RGBColor = randomRGBColor();


//       // std::cout<<RGBColor<<endl;

//       for(size_t j = 0; j < contours[i].size(); j++)
//       {
//         edge_img.at<cv::Vec3b>(contours[i][j].y, contours[i][j].x) = RGBColor;//255
//         //多通道图像: img.at<Vec3b>(i,j)[c]
//         //单通道图像: img.at<uchar>(i,j)
//       }
//     }
//   }

//   // cv::imshow("window",edge_img);
//   // cv::imwrite("effective_edge.png",edge_img);
//   // cv::waitKey(0);

//   for (int x = 0; x < edge_img.cols; x++)
//   {
//     for (int y = 0; y < edge_img.rows; y++)
//     {
//       if ((edge_img.at<cv::Vec3b>(y, x)[0] != 0)||(edge_img.at<cv::Vec3b>(y, x)[1] != 0)||(edge_img.at<cv::Vec3b>(y, x)[2] != 0))
//       {
//         pcl::PointXYZRGB p;
//         p.x = x;
//         p.y = -y;
//         p.z = 0;
//         p.b = edge_img.at<cv::Vec3b>(y, x)[0];
//         p.g = edge_img.at<cv::Vec3b>(y, x)[1];
//         p.r = edge_img.at<cv::Vec3b>(y, x)[2];
//         edge_cloud->points.push_back(p);
//       }
//     }
//   }

//   edge_cloud->width = edge_cloud->points.size();
//   edge_cloud->height = 1;
//   saveRGBPointCloud("check/edge_cloud.pcd", edge_cloud);
//   // cv::imshow("canny result", canny_result);////////////////////////////////////////////////////////////////////////////
//   // cv::imshow("edge result", edge_img);
//   // cv::waitKey();
// }

//将点云投影为深度图
void Calibration::projection(
    const Vector6d& extrinsic_params,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
    const ProjectionType projection_type, const bool is_fill_img,
    cv::Mat& projection_img)
{
    std::vector<cv::Point3f> pts_3d;
    std::vector<float> intensity_list;
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());

    for (size_t i = 0; i < lidar_cloud->size(); i++)
    {
        pcl::PointXYZI point_3d = lidar_cloud->points[i];
        float depth = sqrt(pow(point_3d.x, 2) + pow(point_3d.y, 2) + pow(point_3d.z, 2));
        if (depth > min_depth_ && depth < max_depth_)//min_depth_ = 2.5; max_depth_ = 50;
        {
            pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
            intensity_list.emplace_back(lidar_cloud->points[i].intensity);
        }
    }
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
    cv::Mat r_vec =
        (cv::Mat_<double>(3, 1)
            <<
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
        extrinsic_params[4], extrinsic_params[5]);
    // project 3d-points into image view
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff, pts_2d);
    cv::Mat image_project = cv::Mat::zeros(height_, width_, CV_16UC1);
    cv::Mat rgb_image_project = cv::Mat::zeros(height_, width_, CV_8UC3);
    for (size_t i = 0; i < pts_2d.size(); ++i)
    {
        cv::Point2f point_2d = pts_2d[i];
        if (point_2d.x <= 5 || point_2d.x >= width_ - 5 || point_2d.y <= 5 || point_2d.y >= height_ - 5)
        {
            continue;
        }
        else
        {
            // test depth and intensity both
            if (projection_type == DEPTH)//projection_type = INTENSITY
            {

            }
            else
            {
                float intensity = intensity_list[i];
                if (intensity > 100)//这里的参数似乎是经验参数
                {
                    intensity = 65535;
                }
                else
                {
                    intensity = (intensity / 150.0) * 65535;
                }
                cv::circle(image_project, cv::Point2f(point_2d.x, point_2d.y), 7, intensity, -1);
                /*image_project.at<ushort>(point_2d.y, point_2d.x) = intensity;
                image_project.at<ushort>(point_2d.y - 1, point_2d.x) = intensity;
                image_project.at<ushort>(point_2d.y, point_2d.x - 1) = intensity;
                image_project.at<ushort>(point_2d.y - 1, point_2d.x - 1) = intensity;
                image_project.at<ushort>(point_2d.y + 1, point_2d.x) = intensity;
                image_project.at<ushort>(point_2d.y, point_2d.x + 1) = intensity;
                image_project.at<ushort>(point_2d.y + 1, point_2d.x + 1) = intensity;
                image_project.at<ushort>(point_2d.y + 1, point_2d.x - 1) = intensity;
                image_project.at<ushort>(point_2d.y - 1, point_2d.x + 1) = intensity;*/
            }
        }
    }
    cv::Mat grey_image_projection;
    cv::cvtColor(rgb_image_project, grey_image_projection, cv::COLOR_BGR2GRAY);

    image_project.convertTo(image_project, CV_8UC1, 1 / 256.0);

    projection_img = image_project.clone();
}
// 填补雷达深度图像！！！！
cv::Mat Calibration::fillImg(const cv::Mat& input_img,
    const Direction first_direct,
    const Direction second_direct) {
    cv::Mat fill_img = input_img.clone();
    for (int y = 2; y < input_img.rows - 2; y++) {
        for (int x = 2; x < input_img.cols - 2; x++) {
            if (input_img.at<uchar>(y, x) == 0) {
                if (input_img.at<uchar>(y - 1, x) != 0) {
                    fill_img.at<uchar>(y, x) = input_img.at<uchar>(y - 1, x);
                }
                else {
                    if ((input_img.at<uchar>(y, x - 1)) != 0) {
                        fill_img.at<uchar>(y, x) = input_img.at<uchar>(y, x - 1);
                    }
                }
            }
            else {
                int left_depth = input_img.at<uchar>(y, x - 1);
                int right_depth = input_img.at<uchar>(y, x + 1);
                int up_depth = input_img.at<uchar>(y + 1, x);
                int down_depth = input_img.at<uchar>(y - 1, x);
                int current_depth = input_img.at<uchar>(y, x);
                if ((current_depth - left_depth) > 5 &&
                    (current_depth - right_depth) > 5 && left_depth != 0 &&
                    right_depth != 0) {
                    fill_img.at<uchar>(y, x) = (left_depth + right_depth) / 2;
                }
                else if ((current_depth - up_depth) > 5 &&
                    (current_depth - down_depth) > 5 && up_depth != 0 &&
                    down_depth != 0) {
                    fill_img.at<uchar>(y, x) = (up_depth + down_depth) / 2;
                }
            }
        }
    }
    return fill_img;
}
//生成线特征联接图
cv::Mat Calibration::getConnectImg(
    const int dis_threshold,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_edge_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& depth_edge_cloud)
{
    cv::Mat connect_img = cv::Mat::zeros(height_, width_, CV_8UC3);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr search_cloud =
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tree_cloud =
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    kdtree->setInputCloud(rgb_edge_cloud);
    tree_cloud = rgb_edge_cloud;
    for (size_t i = 0; i < depth_edge_cloud->points.size(); i++)
    {
        cv::Point2d p2(depth_edge_cloud->points[i].x, -depth_edge_cloud->points[i].y);
        if (checkFov(p2))
        {
            pcl::PointXYZRGB p = depth_edge_cloud->points[i];
            search_cloud->points.push_back(p);
        }
    }
    float sumDistance = 0;
    int sumNum = 0;

    int line_count = 0;
    // 指定近邻个数
    int K = 1;
    // 创建两个向量，分别存放近邻的索引值、近邻的中心距
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (size_t i = 0; i < search_cloud->points.size(); i++)
    {
        pcl::PointXYZRGB searchPoint = search_cloud->points[i];
        if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            for (int j = 0; j < K; j++) {
                float distance = sqrt(
                    pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
                    pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
                if (distance < dis_threshold)
                {
                    if (distance > 1)
                        sumDistance = sumDistance + distance - 1;
                    sumNum++;
                    cv::Scalar color = cv::Scalar(0, 255, 0);
                    line_count++;
                    if ((line_count % 3) == 0) {
                        cv::line(connect_img,
                            cv::Point(search_cloud->points[i].x,
                                -search_cloud->points[i].y),
                            cv::Point(tree_cloud->points[pointIdxNKNSearch[j]].x,
                                -tree_cloud->points[pointIdxNKNSearch[j]].y),
                            color, 1);
                    }
                }
            }
        }
    }
    std::cout << "reprojection distance sum " << sumDistance << " reprojection num sum " << sumNum << " Average Pixel Error" << sumDistance / sumNum << std::endl;
    for (size_t i = 0; i < rgb_edge_cloud->size(); i++)
    {
        connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
            rgb_edge_cloud->points[i].x)[0] = 255;
        connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
            rgb_edge_cloud->points[i].x)[1] = 0;
        connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
            rgb_edge_cloud->points[i].x)[2] = 0;
    }
    for (size_t i = 0; i < search_cloud->size(); i++)
    {
        connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
            search_cloud->points[i].x)[0] = 0;
        connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
            search_cloud->points[i].x)[1] = 0;
        connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
            search_cloud->points[i].x)[2] = 255;
    }
    int expand_size = 2;
    cv::Mat expand_edge_img;
    expand_edge_img = connect_img.clone();
    for (int x = expand_size; x < connect_img.cols - expand_size; x++)
    {
        for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
            if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
                for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
                    for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
                    }
                }
            }
            else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
                for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
                    for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
                    }
                }
            }
        }
    }
    return connect_img;
}

bool Calibration::checkFov(const cv::Point2d& p)
{
    if (p.x > 0 && p.x < width_ && p.y > 0 && p.y < height_) {
        return true;
    }
    else {
        return false;
    }
}

// void Calibration::initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
//     const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map) 
// {
//   // for voxel test
//   srand((unsigned)time(NULL));
//   pcl::PointCloud<pcl::PointXYZRGB> test_cloud;

//   for (size_t i = 0; i < input_cloud->size(); i++)
//   {
//     const pcl::PointXYZI &p_c = input_cloud->points[i];
//     float loc_xyz[3];
//     for (int j = 0; j < 3; j++)
//     {
//       loc_xyz[j] = p_c.data[j] / voxel_size;
//       if (loc_xyz[j] < 0)
//       {
//         loc_xyz[j] -= 1.0;//maybe no +0 and -0
//       }
//     }
//     VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
//     //这里是int型，自动把坐标分成了不同的voxel
//     auto iter = voxel_map.find(position);
//     if (iter != voxel_map.end())
//     {
//       voxel_map[position]->cloud->push_back(p_c);
//       pcl::PointXYZRGB p_rgb;
//       p_rgb.x = p_c.x;
//       p_rgb.y = p_c.y;
//       p_rgb.z = p_c.z;
//       p_rgb.r = voxel_map[position]->voxel_color(0);
//       p_rgb.g = voxel_map[position]->voxel_color(1);
//       p_rgb.b = voxel_map[position]->voxel_color(2);
//       test_cloud.push_back(p_rgb);
//     } 
//     else //如果没有voxel那么创建一个，但是查询点会被抛弃
//     {
//       Voxel *voxel = new Voxel(voxel_size);
//       voxel_map[position] = voxel;
//       voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
//       voxel_map[position]->voxel_origin[1] = position.y * voxel_size;
//       voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
//       voxel_map[position]->cloud->push_back(p_c);
//       int r = rand() % 256;
//       int g = rand() % 256;
//       int b = rand() % 256;
//       voxel_map[position]->voxel_color << r, g, b;
//     }
//   }

//   pcl::io::savePCDFileASCII("VoxelCloud.pcd", test_cloud);

//   // sensor_msgs::PointCloud2 pub_cloud;
//   // pcl::toROSMsg(test_cloud, pub_cloud);
//   // pub_cloud.header.frame_id = "livox";
//   // rgb_cloud_pub_.publish(pub_cloud);
//   for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
//   {
//     if (iter->second->cloud->size() > 20)
//     {
//       //down_sampling_voxel(*(iter->second->cloud), 0.02);
//       //体素滤波，小于0.01不处理
//       //我认为低密度点云可以不滤波反而更好
//     }
//   }
// }

void Calibration::initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map)
{
    // for voxel test
    srand((unsigned)time(NULL));
    // pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr VoxeledPts(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < input_cloud->size(); i++)
    {
        const pcl::PointXYZI& p_c = input_cloud->points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = (p_c.data[j] + voxel_size / 2) / voxel_size;
            // if (loc_xyz[j] < 0)
            // {
            //   loc_xyz[j] = loc_xyz[j] - 1.0;//maybe no +0 and -0
            // }
        }
        // cout<<voxel_size<<"voxel_size"<<endl;
        //点云在voxel中移动, +为沿轴反向, -为沿轴正向
        VOXEL_LOC position((int64_t)ceil(loc_xyz[0]), (int64_t)ceil(loc_xyz[1]), (int64_t)ceil(loc_xyz[2]));
        //这里是int型，自动把坐标分成了不同的voxel
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end())
        {
            voxel_map[position]->cloud->push_back(p_c);
            pcl::PointXYZRGB p_rgb;
            p_rgb.x = p_c.x;
            p_rgb.y = p_c.y;
            p_rgb.z = p_c.z;
            p_rgb.r = voxel_map[position]->voxel_color(0);
            p_rgb.g = voxel_map[position]->voxel_color(1);
            p_rgb.b = voxel_map[position]->voxel_color(2);
            VoxeledPts->push_back(p_rgb);
        }
        else //如果没有voxel那么创建一个，但是查询点会被抛弃
        {
            Voxel* voxel = new Voxel(voxel_size);
            voxel_map[position] = voxel;
            voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
            voxel_map[position]->voxel_origin[1] = (position.y - 0.5) * voxel_size;
            voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
            voxel_map[position]->cloud->push_back(p_c);
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;
            voxel_map[position]->voxel_color << r, g, b;
        }
    }

    pcl::io::savePCDFileASCII("check/" + std::to_string(dataProcessingNum) + "_VoxelCloud.pcd", *VoxeledPts);
    // sensor_msgs::PointCloud2 pub_cloud;
    // pcl::toROSMsg(test_cloud, pub_cloud);
    // pub_cloud.header.frame_id = "livox";
    // rgb_cloud_pub_.publish(pub_cloud);
    // for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
    // {
    //   if (iter->second->cloud->size() > 20)
    //   {
    //     //down_sampling_voxel(*(iter->second->cloud), 0.02);
    //     //体素滤波，小于0.01不处理
    //     //我认为低密度点云可以不滤波反而更好
    //   }
    // }
}

void Calibration::LiDAREdgeExtraction(
    const std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map,
    const float ransac_dis_thre, const int plane_size_threshold,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d)
{

    lidar_line_cloud_3d = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
    {
        if (iter->second->cloud->size() > 30)
        {
            std::vector<Plane> plane_list;
            // 创建一个体素滤波器
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
            //创建一个模型参数对象，用于记录结果
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            // inliers表示误差能容忍的点，记录点云序号
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            //创建一个分割器
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            // Optional,设置结果平面展示的点是分割掉的点还是分割剩下的点
            seg.setOptimizeCoefficients(true);
            // Mandatory-设置目标几何形状
            seg.setModelType(pcl::SACMODEL_PLANE);
            //分割方法：随机采样法
            seg.setMethodType(pcl::SAC_RANSAC);
            //设置误差容忍范围，也就是阈值
            // if (iter->second->voxel_origin[0] < 10) 
            // {
            //   seg.setDistanceThreshold(ransac_dis_thre);
            // }
            // else 
            // {
            //   seg.setDistanceThreshold(ransac_dis_thre);
            // }
            seg.setDistanceThreshold(ransac_dis_thre);//0.01

            // pcl::PointCloud<pcl::PointXYZRGB> color_planner_cloud;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_planner_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr mergedPts(new pcl::PointCloud<pcl::PointXYZRGB>());
            int plane_index = 0;
            while (cloud_filter->points.size() > 10)
            {
                pcl::PointCloud<pcl::PointXYZI> planner_cloud;
                pcl::ExtractIndices<pcl::PointXYZI> extract;
                //输入点云
                seg.setInputCloud(cloud_filter);
                seg.setMaxIterations(500);
                //分割点云
                seg.segment(*inliers, *coefficients);

                if (inliers->indices.size() == 0)
                {
                    break;
                }
                extract.setIndices(inliers);
                extract.setInputCloud(cloud_filter);
                extract.filter(planner_cloud);
                if (planner_cloud.size() > plane_size_threshold)
                {
                    pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
                    std::vector<unsigned int> colors;
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    pcl::PointXYZRGB p_center;
                    p_center.x = 0;
                    p_center.y =0;
                    p_center.z = 0;
                    for (size_t i = 0; i < planner_cloud.points.size(); i++)
                    {
                        pcl::PointXYZRGB p;
                        p.x = planner_cloud.points[i].x;
                        p.y = planner_cloud.points[i].y;
                        p.z = planner_cloud.points[i].z;
                        p_center.x += p.x;
                        p_center.y += p.y;
                        p_center.z += p.z;
                        p.r = colors[0];
                        p.g = colors[1];
                        p.b = colors[2];
                        color_cloud.push_back(p);
                        color_planner_cloud->push_back(p);
                    }
                    p_center.x = p_center.x / planner_cloud.size();
                    p_center.y = p_center.y / planner_cloud.size();
                    p_center.z = p_center.z / planner_cloud.size();
                    Plane single_plane;
                    single_plane.cloud = planner_cloud;
                    single_plane.p_center = p_center;
                    single_plane.normal << coefficients->values[0],
                        coefficients->values[1], coefficients->values[2];
                    single_plane.index = plane_index;
                    plane_list.push_back(single_plane);
                    plane_index++;
                }
                extract.setNegative(true);
                pcl::PointCloud<pcl::PointXYZI> cloud_f;
                extract.filter(cloud_f);
                *cloud_filter = cloud_f;
            }
            if (plane_list.size() >= 2)
            {

            }

            mergePlanes(plane_list);

            savePlanes(plane_list, mergedPts);
            pcl::io::savePCDFileASCII("check/" + std::to_string(dataProcessingNum) + "_LaserPlanes.pcd", *color_planner_cloud);
            pcl::io::savePCDFileASCII("check/" + std::to_string(dataProcessingNum) + "_LaserMergedPlanes.pcd", *mergedPts);

            std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
            // calcLine(plane_list, voxel_size_, iter->second->voxel_origin, line_cloud_list);
            calcLineLin(plane_list, line_cloud_list);
            // ouster 5,normal 3
            // std::cout<<line_cloud_list.size()<<std::endl;
            // std::cout<<plane_list.size()<<std::endl;
            if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 8)
            {
                for (size_t cloud_index = 0; cloud_index < line_cloud_list.size(); cloud_index++)
                {
                    for (size_t i = 0; i < line_cloud_list[cloud_index].size(); i++)
                    {
                        pcl::PointXYZI p = line_cloud_list[cloud_index].points[i];
                        plane_line_cloud_->points.push_back(p);
                        plane_line_number_.push_back(line_number_);
                    }
                    line_number_++;
                }
            }
        }
    }
}

// void Calibration::calcLineLin(std::vector<Plane> &plane_list, std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list)
// {
//   if (plane_list.size() >= 2 && plane_list.size() <= plane_max_size_)
//   {
//     // pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
//     for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1; plane_index1++)
//     {
//       for (size_t plane_index2 = plane_index1 + 1; plane_index2 < plane_list.size(); plane_index2++)
//       {
//         float a1 = plane_list[plane_index1].normal[0];
//         float b1 = plane_list[plane_index1].normal[1];
//         float c1 = plane_list[plane_index1].normal[2];
//         float x1 = plane_list[plane_index1].p_center.x;
//         float y1 = plane_list[plane_index1].p_center.y;
//         float z1 = plane_list[plane_index1].p_center.z;
//         float minusd1 = a1 * x1 + b1 * y1 + c1 * z1;
//         float a2 = plane_list[plane_index2].normal[0];
//         float b2 = plane_list[plane_index2].normal[1];
//         float c2 = plane_list[plane_index2].normal[2];
//         float x2 = plane_list[plane_index2].p_center.x;
//         float y2 = plane_list[plane_index2].p_center.y;
//         float z2 = plane_list[plane_index2].p_center.z;
//         float minusd2 = a2 * x2 + b2 * y2 + c2 * z2;
//         float matrix[4][5];
//         matrix[1][1] = a1;
//         matrix[1][2] = b1;
//         matrix[1][3] = c1;
//         matrix[1][4] = minusd1;
//         matrix[2][1] = a2;
//         matrix[2][2] = b2;
//         matrix[2][3] = c2;
//         matrix[2][4] = minusd2;

//         std::vector<Eigen::Vector3d> EndPointSet;
//         Eigen::Vector3d EndPoint;

//         float theta = a1 * a2 + b1 * b2 + c1 * c2;
//         //
//         float point_dis_threshold = 0.00;
//         if (theta > theta_max_ && theta < theta_min_) //cos(30) -- cos(150)
//         {
//           ////////////////////////两个点云合并
//           pcl::PointCloud<pcl::PointXYZ>::Ptr GlobalCloud(new pcl::PointCloud<pcl::PointXYZ>);
//           pcl::copyPointCloud(plane_list[plane_index1].cloud + plane_list[plane_index2].cloud, *GlobalCloud);
//           // pcl::PointCloud<pcl::PointXYZI> GlobalCloud;
//           // GlobalCloud = plane_list[plane_index1].cloud + plane_list[plane_index2].cloud;
//           ////////////////////////获取坐标范围，得到两个点
//           pcl::PointXYZ minXYZ, maxXYZ;
// 	        pcl::getMinMax3D(*GlobalCloud, minXYZ, maxXYZ);

//           ////////////////////////得到线与面的两个交点
//           matrix[3][1] = 1;
//           matrix[3][2] = 0;
//           matrix[3][3] = 0;
//           matrix[3][4] = minXYZ.x;
//           calc<float>(matrix, EndPoint);

//           if (EndPoint[0] >= minXYZ.x && EndPoint[0] <= maxXYZ.x &&
//               EndPoint[1] >= minXYZ.y && EndPoint[1] <= maxXYZ.y &&
//               EndPoint[2] >= minXYZ.z && EndPoint[2] <= maxXYZ.z )
//           {
//             EndPointSet.push_back(EndPoint);
//           }
//           matrix[3][4] = maxXYZ.x;
//           calc<float>(matrix, EndPoint);
//           if (EndPoint[0] >= minXYZ.x && EndPoint[0] <= maxXYZ.x &&
//               EndPoint[1] >= minXYZ.y && EndPoint[1] <= maxXYZ.y &&
//               EndPoint[2] >= minXYZ.z && EndPoint[2] <= maxXYZ.z )
//           {
//             EndPointSet.push_back(EndPoint);
//           }

//           matrix[3][1] = 0;
//           matrix[3][2] = 1;
//           matrix[3][3] = 0;
//           matrix[3][4] = minXYZ.y;
//           calc<float>(matrix, EndPoint);
//           if (EndPoint[0] >= minXYZ.x && EndPoint[0] <= maxXYZ.x &&
//               EndPoint[1] >= minXYZ.y && EndPoint[1] <= maxXYZ.y &&
//               EndPoint[2] >= minXYZ.z && EndPoint[2] <= maxXYZ.z )
//           {
//             EndPointSet.push_back(EndPoint);
//           }
//           matrix[3][4] = maxXYZ.y;
//           calc<float>(matrix, EndPoint);
//           if (EndPoint[0] >= minXYZ.x && EndPoint[0] <= maxXYZ.x &&
//               EndPoint[1] >= minXYZ.y && EndPoint[1] <= maxXYZ.y &&
//               EndPoint[2] >= minXYZ.z && EndPoint[2] <= maxXYZ.z )
//           {
//             EndPointSet.push_back(EndPoint);
//           }


//           matrix[3][1] = 0;
//           matrix[3][2] = 0;
//           matrix[3][3] = 1;
//           matrix[3][4] = minXYZ.z;
//           calc<float>(matrix, EndPoint);
//           if (EndPoint[0] >= minXYZ.x && EndPoint[0] <= maxXYZ.x &&
//               EndPoint[1] >= minXYZ.y && EndPoint[1] <= maxXYZ.y &&
//               EndPoint[2] >= minXYZ.z && EndPoint[2] <= maxXYZ.z )
//           {
//             EndPointSet.push_back(EndPoint);
//           }


//           matrix[3][4] = maxXYZ.z;
//           calc<float>(matrix, EndPoint);
//           if (EndPoint[0] >= minXYZ.x && EndPoint[0] <= maxXYZ.x &&
//               EndPoint[1] >= minXYZ.y && EndPoint[1] <= maxXYZ.y &&
//               EndPoint[2] >= minXYZ.z && EndPoint[2] <= maxXYZ.z )
//           {
//             EndPointSet.push_back(EndPoint);
//           }


//           ////////////////////////得到相交平面的两个端点？？？？为了处理稀疏点云，还是两个端点好,两端对进！
//           // pcl::PointCloud<pcl::PointXYZI> line_cloud;
//           // Eigen::Vector3d EndPointA = pointA;
//           // Eigen::Vector3d EndPointB = pointB;

//           Eigen::Vector3d EndPointA;
//           Eigen::Vector3d EndPointB;

//           // std::cout << "points size:" << EndPointSet.size() << std::endl;

//           if (EndPointSet.size() == 2)
//           {
//             pcl::PointCloud<pcl::PointXYZI> line_cloud;
//             pcl::PointXYZ p1(EndPointSet[0][0], EndPointSet[0][1], EndPointSet[0][2]);//let it A
//             pcl::PointXYZ p2(EndPointSet[1][0], EndPointSet[1][1], EndPointSet[1][2]);//let it B
//             float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +pow(p1.z - p2.z, 2));
//             cout<<"length"<<length<<endl;
//             // 指定近邻个数
//             int K = 1;
//             int EndPointFlag = 0;
//             // 创建两个向量，分别存放近邻的索引值、近邻的中心距
//             std::vector<int> pointIdxRSearch1(K);
//             std::vector<float> pointRSquaredDistance1(K);
//             std::vector<int> pointIdxRSearch2(K);
//             std::vector<float> pointRSquaredDistance2(K);
//             pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(new pcl::search::KdTree<pcl::PointXYZI>());
//             pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZI>());
//             kdtree1->setInputCloud(plane_list[plane_index1].cloud.makeShared());
//             kdtree2->setInputCloud(plane_list[plane_index2].cloud.makeShared());

//             for (float inc = 0; inc <= length; inc += 0.01)//From A to B searching
//             {
//               pcl::PointXYZI p;
//               p.x = p1.x + (p2.x - p1.x) * inc / length;
//               p.y = p1.y + (p2.y - p1.y) * inc / length;
//               p.z = p1.z + (p2.z - p1.z) * inc / length;
//               p.intensity = 100;
//               //number of neighbors found in radius 
//               if(kdtree1->radiusSearch(p, 0.01, pointIdxRSearch1, pointRSquaredDistance1)>0||
//                  kdtree2->radiusSearch(p, 0.01, pointIdxRSearch2, pointRSquaredDistance2)>0)
//               {
//                 // EndPointA = p.getVector3fMap();
//                 EndPointA[0] = p.x;
//                 EndPointA[1] = p.y;
//                 EndPointA[2] = p.z;
//                 EndPointFlag++;
//                 break;
//               }
//             }
//             for (float inc = 0; inc <= length; inc += 0.01)//From B to A searching
//             {
//               pcl::PointXYZI p;
//               p.x = p2.x + (p1.x - p2.x) * inc / length;
//               p.y = p2.y + (p1.y - p2.y) * inc / length;
//               p.z = p2.z + (p1.z - p2.z) * inc / length;
//               p.intensity = 100;
//               //number of neighbors found in radius 
//               if(kdtree1->radiusSearch(p, 0.01, pointIdxRSearch1, pointRSquaredDistance1)>0||
//                  kdtree2->radiusSearch(p, 0.01, pointIdxRSearch2, pointRSquaredDistance2)>0)
//               {
//                 // EndPointB = p.getVector3fMap();
//                 EndPointB[0] = p.x;
//                 EndPointB[1] = p.y;
//                 EndPointB[2] = p.z;
//                 EndPointFlag++;
//                 break;
//               }
//             }
//             // std::cout<<endl<<"ENDAB"<<endl;
//             // std::cout<<"/////////////////////////////////////"<<endl;
//             // std::cout<<EndPointA<<endl;
//             // std::cout<<"/////////////////////////////////////"<<endl;
//             // std::cout<<EndPointB<<endl;
//             // std::cout<<"/////////////////////////////////////"<<endl;
//             ////////////////////////利用两个端点得到合适密度的点云线段，现在的密度是0.01
//             if(EndPointFlag==2)
//             {
//               float SegmentLength = sqrt(pow(EndPointA[0] - EndPointB[0], 2) + pow(EndPointA[1] - EndPointB[1], 2) +pow(EndPointA[2] - EndPointB[2], 2));
//               for (float inc = 0; inc <= SegmentLength; inc += 0.01)
//               {
//                 pcl::PointXYZI p;
//                 p.x = EndPointA[0] + (EndPointB[0] - EndPointA[0]) * inc / SegmentLength;
//                 p.y = EndPointA[1] + (EndPointB[1] - EndPointA[1]) * inc / SegmentLength;
//                 p.z = EndPointA[2] + (EndPointB[2] - EndPointA[2]) * inc / SegmentLength;
//                 p.intensity = 100;
//                 line_cloud.push_back(p);
//               }
//             }

//             // cout<<"now we have"<<line_cloud.size()<<endl;
//             if (line_cloud.size() > 10)
//             {
//               line_cloud_list.push_back(line_cloud);
//             }

//           }
//         }
//       }
//     }
//   }
// }

void Calibration::calcLineLin(std::vector<Plane>& plane_list, std::vector<pcl::PointCloud<pcl::PointXYZI>>& line_cloud_list)
{
    if (plane_list.size() >= 2 && plane_list.size() <= plane_max_size_)
    {
        // pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
        for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1; plane_index1++)
        {
            for (size_t plane_index2 = plane_index1 + 1; plane_index2 < plane_list.size(); plane_index2++)
            {
                float a1 = plane_list[plane_index1].normal[0];
                float b1 = plane_list[plane_index1].normal[1];
                float c1 = plane_list[plane_index1].normal[2];
                float x1 = plane_list[plane_index1].p_center.x;
                float y1 = plane_list[plane_index1].p_center.y;
                float z1 = plane_list[plane_index1].p_center.z;
                float minusd1 = a1 * x1 + b1 * y1 + c1 * z1;
                float a2 = plane_list[plane_index2].normal[0];
                float b2 = plane_list[plane_index2].normal[1];
                float c2 = plane_list[plane_index2].normal[2];
                float x2 = plane_list[plane_index2].p_center.x;
                float y2 = plane_list[plane_index2].p_center.y;
                float z2 = plane_list[plane_index2].p_center.z;
                float minusd2 = a2 * x2 + b2 * y2 + c2 * z2;
                float matrix[4][5];
                matrix[1][1] = a1;
                matrix[1][2] = b1;
                matrix[1][3] = c1;
                matrix[1][4] = minusd1;
                matrix[2][1] = a2;
                matrix[2][2] = b2;
                matrix[2][3] = c2;
                matrix[2][4] = minusd2;

                std::vector<Eigen::Vector3d> points;
                Eigen::Vector3d pointA;
                Eigen::Vector3d pointB;

                float theta = a1 * a2 + b1 * b2 + c1 * c2;
                //
                float point_dis_threshold = 0.00;
                if (theta > theta_max_ && theta < theta_min_) //cos(30) -- cos(150)
                {
                    ////////////////////////两个点云合并
                    pcl::PointCloud<pcl::PointXYZ>::Ptr GlobalCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::copyPointCloud(plane_list[plane_index1].cloud + plane_list[plane_index2].cloud, *GlobalCloud);
                    // pcl::PointCloud<pcl::PointXYZI> GlobalCloud;
                    // GlobalCloud = plane_list[plane_index1].cloud + plane_list[plane_index2].cloud;
                    ////////////////////////获取坐标范围，得到两个点
                    pcl::PointXYZ minXYZ, maxXYZ;
                    pcl::getMinMax3D(*GlobalCloud, minXYZ, maxXYZ);

                    ////////////////////////叉乘获得线与面的交点所在的两个面
                    Eigen::Vector3d LineDirection = plane_list[plane_index1].normal.cross(plane_list[plane_index2].normal);
                    // std::cout<<endl<<"plane_list[plane_index1].normal"<<endl;
                    // std::cout<<plane_list[plane_index1].normal<<endl;
                    // std::cout<<plane_list[plane_index2].normal<<endl;
                    // std::cout<<LineDirection<<endl;
                    int IntscFlag = -1;//0yoz//1xoz//2xoy
                    if (abs(LineDirection[0]) > abs(LineDirection[1]) && abs(LineDirection[0]) > abs(LineDirection[2]))
                        IntscFlag = 0;
                    if (abs(LineDirection[1]) > abs(LineDirection[0]) && abs(LineDirection[1]) > abs(LineDirection[2]))
                        IntscFlag = 1;
                    if (abs(LineDirection[2]) > abs(LineDirection[0]) && abs(LineDirection[2]) > abs(LineDirection[1]))
                        IntscFlag = 2;
                    ////////////////////////得到线与面的两个交点
                    if (IntscFlag == -1)break;
                    if (IntscFlag == 0)
                    {
                        matrix[3][1] = 1;
                        matrix[3][2] = 0;
                        matrix[3][3] = 0;
                        matrix[3][4] = minXYZ.x;
                        calc<float>(matrix, pointA);
                        points.push_back(pointA);
                        matrix[3][4] = maxXYZ.x;
                        calc<float>(matrix, pointB);
                        points.push_back(pointB);
                    }
                    if (IntscFlag == 1)
                    {
                        matrix[3][1] = 0;
                        matrix[3][2] = 1;
                        matrix[3][3] = 0;
                        matrix[3][4] = minXYZ.y;
                        calc<float>(matrix, pointA);
                        points.push_back(pointA);
                        matrix[3][4] = maxXYZ.y;
                        calc<float>(matrix, pointB);
                        points.push_back(pointB);
                    }
                    if (IntscFlag == 2)
                    {
                        matrix[3][1] = 0;
                        matrix[3][2] = 0;
                        matrix[3][3] = 1;
                        matrix[3][4] = minXYZ.z;
                        calc<float>(matrix, pointA);
                        points.push_back(pointA);
                        matrix[3][4] = maxXYZ.z;
                        calc<float>(matrix, pointB);
                        points.push_back(pointB);
                    }
                    // std::cout<<endl<<"AB"<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<pointA<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<pointB<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    ////////////////////////得到相交平面的两个端点？？？？为了处理稀疏点云，还是两个端点好,两端对进！
                    // pcl::PointCloud<pcl::PointXYZI> line_cloud;
                    // Eigen::Vector3d EndPointA = pointA;
                    // Eigen::Vector3d EndPointB = pointB;

                    Eigen::Vector3d EndPointA;
                    Eigen::Vector3d EndPointB;

                    // std::cout << "points size:" << points.size() << std::endl;

                    pcl::PointCloud<pcl::PointXYZI> line_cloud;
                    pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);//let it A
                    pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);//let it B
                    float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                    // 指定近邻个数
                    int K = 1;
                    int EndPointFlag = 0;
                    // 创建两个向量，分别存放近邻的索引值、近邻的中心距
                    std::vector<int> pointIdxRSearch1(K);
                    std::vector<float> pointRSquaredDistance1(K);
                    std::vector<int> pointIdxRSearch2(K);
                    std::vector<float> pointRSquaredDistance2(K);
                    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(new pcl::search::KdTree<pcl::PointXYZI>());
                    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZI>());
                    kdtree1->setInputCloud(plane_list[plane_index1].cloud.makeShared());
                    kdtree2->setInputCloud(plane_list[plane_index2].cloud.makeShared());

                    for (float inc = 0; inc <= length; inc += 0.01)//From A to B searching
                    {
                        pcl::PointXYZI p;
                        p.x = p1.x + (p2.x - p1.x) * inc / length;
                        p.y = p1.y + (p2.y - p1.y) * inc / length;
                        p.z = p1.z + (p2.z - p1.z) * inc / length;
                        p.intensity = 100;
                        //number of neighbors found in radius 
                        if (kdtree1->radiusSearch(p, SearchForPointDis_, pointIdxRSearch1, pointRSquaredDistance1) > 0 ||
                            kdtree2->radiusSearch(p, SearchForPointDis_, pointIdxRSearch2, pointRSquaredDistance2) > 0)
                        {
                            // EndPointA = p.getVector3fMap();
                            EndPointA[0] = p.x;
                            EndPointA[1] = p.y;
                            EndPointA[2] = p.z;
                            EndPointFlag++;
                            break;
                        }
                    }
                    for (float inc = 0; inc <= length; inc += 0.01)//From B to A searching
                    {
                        pcl::PointXYZI p;
                        p.x = p2.x + (p1.x - p2.x) * inc / length;
                        p.y = p2.y + (p1.y - p2.y) * inc / length;
                        p.z = p2.z + (p1.z - p2.z) * inc / length;
                        p.intensity = 100;
                        //number of neighbors found in radius 
                        if (kdtree1->radiusSearch(p, SearchForPointDis_, pointIdxRSearch1, pointRSquaredDistance1) > 0 ||
                            kdtree2->radiusSearch(p, SearchForPointDis_, pointIdxRSearch2, pointRSquaredDistance2) > 0)
                        {
                            // EndPointB = p.getVector3fMap();
                            EndPointB[0] = p.x;
                            EndPointB[1] = p.y;
                            EndPointB[2] = p.z;
                            EndPointFlag++;
                            break;
                        }
                    }
                    // std::cout<<endl<<"ENDAB"<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<EndPointA<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<EndPointB<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    ////////////////////////利用两个端点得到合适密度的点云线段，现在的密度是0.01
                    if (EndPointFlag == 2)
                    {
                        float SegmentLength = sqrt(pow(EndPointA[0] - EndPointB[0], 2) + pow(EndPointA[1] - EndPointB[1], 2) + pow(EndPointA[2] - EndPointB[2], 2));

                        if (SegmentLength > 0.4)
                        {
                            Eigen::Vector3d NewA, NewB;
                            NewA = EndPointA + (EndPointB - EndPointA) * 0.04 / SegmentLength;
                            NewB = EndPointB + (EndPointA - EndPointB) * 0.04 / SegmentLength;
                            EndPointA = NewA;
                            EndPointB = NewB;
                        }

                        for (float inc = 0; inc <= SegmentLength; inc += 0.01)
                        {
                            pcl::PointXYZI p;
                            p.x = EndPointA[0] + (EndPointB[0] - EndPointA[0]) * inc / SegmentLength;
                            p.y = EndPointA[1] + (EndPointB[1] - EndPointA[1]) * inc / SegmentLength;
                            p.z = EndPointA[2] + (EndPointB[2] - EndPointA[2]) * inc / SegmentLength;
                            p.intensity = 100;
                            line_cloud.push_back(p);
                        }
                    }
                    // cout<<"now we have"<<line_cloud.size()<<endl;
                    if (line_cloud.size() > 10)
                    {
                        line_cloud_list.push_back(line_cloud);
                    }
                }
            }
        }
    }
}

// void Calibration::calcLine(
//     const std::vector<Plane> &plane_list, const double voxel_size,
//     const Eigen::Vector3d origin,
//     std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list) 
// {
//   if (plane_list.size() >= 2 && plane_list.size() <= plane_max_size_) 
//   {
//     pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
//     for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1; plane_index1++) 
//     {
//       for (size_t plane_index2 = plane_index1 + 1; plane_index2 < plane_list.size(); plane_index2++) 
//       {
//         float a1 = plane_list[plane_index1].normal[0];
//         float b1 = plane_list[plane_index1].normal[1];
//         float c1 = plane_list[plane_index1].normal[2];
//         float x1 = plane_list[plane_index1].p_center.x;
//         float y1 = plane_list[plane_index1].p_center.y;
//         float z1 = plane_list[plane_index1].p_center.z;
//         float a2 = plane_list[plane_index2].normal[0];
//         float b2 = plane_list[plane_index2].normal[1];
//         float c2 = plane_list[plane_index2].normal[2];
//         float x2 = plane_list[plane_index2].p_center.x;
//         float y2 = plane_list[plane_index2].p_center.y;
//         float z2 = plane_list[plane_index2].p_center.z;
//         float theta = a1 * a2 + b1 * b2 + c1 * c2;
//         //
//         float point_dis_threshold = 0.00;
//         if (theta > theta_max_ && theta < theta_min_) //cos(30) -- cos(150) 
//         {
//           // for (int i = 0; i < 6; i++) {
//           if (plane_list[plane_index1].cloud.size() > 0 && plane_list[plane_index2].cloud.size() > 0)
//           {
//             float matrix[4][5];
//             matrix[1][1] = a1;
//             matrix[1][2] = b1;
//             matrix[1][3] = c1;
//             matrix[1][4] = a1 * x1 + b1 * y1 + c1 * z1;
//             matrix[2][1] = a2;
//             matrix[2][2] = b2;
//             matrix[2][3] = c2;
//             matrix[2][4] = a2 * x2 + b2 * y2 + c2 * z2;
//             // six types
//             std::vector<Eigen::Vector3d> points;
//             Eigen::Vector3d point;
//             matrix[3][1] = 1;
//             matrix[3][2] = 0;
//             matrix[3][3] = 0;
//             matrix[3][4] = origin[0];
//             calc<float>(matrix, point);
//             if (point[0] >= origin[0] - point_dis_threshold &&
//                 point[0] <= origin[0] + voxel_size + point_dis_threshold &&
//                 point[1] >= origin[1] - point_dis_threshold &&
//                 point[1] <= origin[1] + voxel_size + point_dis_threshold &&
//                 point[2] >= origin[2] - point_dis_threshold &&
//                 point[2] <= origin[2] + voxel_size + point_dis_threshold)
//             {
//               points.push_back(point);
//             }
//             matrix[3][1] = 0;
//             matrix[3][2] = 1;
//             matrix[3][3] = 0;
//             matrix[3][4] = origin[1];
//             calc<float>(matrix, point);
//             if (point[0] >= origin[0] - point_dis_threshold &&
//                 point[0] <= origin[0] + voxel_size + point_dis_threshold &&
//                 point[1] >= origin[1] - point_dis_threshold &&
//                 point[1] <= origin[1] + voxel_size + point_dis_threshold &&
//                 point[2] >= origin[2] - point_dis_threshold &&
//                 point[2] <= origin[2] + voxel_size + point_dis_threshold)
//             {
//               points.push_back(point);
//             }
//             matrix[3][1] = 0;
//             matrix[3][2] = 0;
//             matrix[3][3] = 1;
//             matrix[3][4] = origin[2];
//             calc<float>(matrix, point);
//             if (point[0] >= origin[0] - point_dis_threshold &&
//                 point[0] <= origin[0] + voxel_size + point_dis_threshold &&
//                 point[1] >= origin[1] - point_dis_threshold &&
//                 point[1] <= origin[1] + voxel_size + point_dis_threshold &&
//                 point[2] >= origin[2] - point_dis_threshold &&
//                 point[2] <= origin[2] + voxel_size + point_dis_threshold)
//             {
//               points.push_back(point);
//             }
//             matrix[3][1] = 1;
//             matrix[3][2] = 0;
//             matrix[3][3] = 0;
//             matrix[3][4] = origin[0] + voxel_size;
//             calc<float>(matrix, point);
//             if (point[0] >= origin[0] - point_dis_threshold &&
//                 point[0] <= origin[0] + voxel_size + point_dis_threshold &&
//                 point[1] >= origin[1] - point_dis_threshold &&
//                 point[1] <= origin[1] + voxel_size + point_dis_threshold &&
//                 point[2] >= origin[2] - point_dis_threshold &&
//                 point[2] <= origin[2] + voxel_size + point_dis_threshold)
//             {
//               points.push_back(point);
//             }
//             matrix[3][1] = 0;
//             matrix[3][2] = 1;
//             matrix[3][3] = 0;
//             matrix[3][4] = origin[1] + voxel_size;
//             calc<float>(matrix, point);
//             if (point[0] >= origin[0] - point_dis_threshold &&
//                 point[0] <= origin[0] + voxel_size + point_dis_threshold &&
//                 point[1] >= origin[1] - point_dis_threshold &&
//                 point[1] <= origin[1] + voxel_size + point_dis_threshold &&
//                 point[2] >= origin[2] - point_dis_threshold &&
//                 point[2] <= origin[2] + voxel_size + point_dis_threshold)
//             {
//               points.push_back(point);
//             }
//             matrix[3][1] = 0;
//             matrix[3][2] = 0;
//             matrix[3][3] = 1;
//             matrix[3][4] = origin[2] + voxel_size;
//             calc<float>(matrix, point);
//             if (point[0] >= origin[0] - point_dis_threshold &&
//                 point[0] <= origin[0] + voxel_size + point_dis_threshold &&
//                 point[1] >= origin[1] - point_dis_threshold &&
//                 point[1] <= origin[1] + voxel_size + point_dis_threshold &&
//                 point[2] >= origin[2] - point_dis_threshold &&
//                 point[2] <= origin[2] + voxel_size + point_dis_threshold)
//             {
//               points.push_back(point);
//             }
//             // std::cout << "points size:" << points.size() << std::endl;
//             if (points.size() == 2)
//             {
//               pcl::PointCloud<pcl::PointXYZI> line_cloud;
//               pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);
//               pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);
//               float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +pow(p1.z - p2.z, 2));
//               // 指定近邻个数
//               int K = 1;
//               // 创建两个向量，分别存放近邻的索引值、近邻的中心距
//               std::vector<int> pointIdxNKNSearch1(K);
//               std::vector<float> pointNKNSquaredDistance1(K);
//               std::vector<int> pointIdxNKNSearch2(K);
//               std::vector<float> pointNKNSquaredDistance2(K);
//               pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(new pcl::search::KdTree<pcl::PointXYZI>());
//               pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(
//                   new pcl::search::KdTree<pcl::PointXYZI>());
//               kdtree1->setInputCloud(
//                   plane_list[plane_index1].cloud.makeShared());
//               kdtree2->setInputCloud(
//                   plane_list[plane_index2].cloud.makeShared());
//               for (float inc = 0; inc <= length; inc += 0.01) {
//                 pcl::PointXYZI p;
//                 p.x = p1.x + (p2.x - p1.x) * inc / length;
//                 p.y = p1.y + (p2.y - p1.y) * inc / length;
//                 p.z = p1.z + (p2.z - p1.z) * inc / length;
//                 p.intensity = 100;
//                 if ((kdtree1->nearestKSearch(p, K, pointIdxNKNSearch1, pointNKNSquaredDistance1) > 0) &&
//                     (kdtree2->nearestKSearch(p, K, pointIdxNKNSearch2, pointNKNSquaredDistance2) > 0))
//                 {
//                   float dis1 =
//                       pow(p.x - plane_list[plane_index1]
//                                     .cloud.points[pointIdxNKNSearch1[0]]
//                                     .x,
//                           2) +
//                       pow(p.y - plane_list[plane_index1]
//                                     .cloud.points[pointIdxNKNSearch1[0]]
//                                     .y,
//                           2) +
//                       pow(p.z - plane_list[plane_index1]
//                                     .cloud.points[pointIdxNKNSearch1[0]]
//                                     .z,
//                           2);
//                   float dis2 =
//                       pow(p.x - plane_list[plane_index2]
//                                     .cloud.points[pointIdxNKNSearch2[0]]
//                                     .x,
//                           2) +
//                       pow(p.y - plane_list[plane_index2]
//                                     .cloud.points[pointIdxNKNSearch2[0]]
//                                     .y,
//                           2) +
//                       pow(p.z - plane_list[plane_index2]
//                                     .cloud.points[pointIdxNKNSearch2[0]]
//                                     .z,
//                           2);
//                   if ((dis1 < min_line_dis_threshold_ * min_line_dis_threshold_ &&
//                        dis2 < max_line_dis_threshold_ * max_line_dis_threshold_) ||
//                       ((dis1 < max_line_dis_threshold_ * max_line_dis_threshold_ &&
//                         dis2 < min_line_dis_threshold_ * min_line_dis_threshold_))) 
//                   {
//                     line_cloud.push_back(p);
//                   }
//                 }
//               }
//               if (line_cloud.size() > 10) {
//                 line_cloud_list.push_back(line_cloud);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

void Calibration::buildVPnp(
    const Vector6d& extrinsic_params, const int dis_threshold,
    const bool show_residual,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cam_edge_cloud_2d,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
    std::vector<VPnPData>& pnp_list)
{
    pnp_list.clear();
    std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
    for (int y = 0; y < height_; y++) {
        std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
        for (int x = 0; x < width_; x++) {
            std::vector<pcl::PointXYZI> col_pts_container;
            row_pts_container.push_back(col_pts_container);
        }
        img_pts_container.push_back(row_pts_container);
    }
    std::vector<cv::Point3d> pts_3d;
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());

    for (size_t i = 0; i < lidar_line_cloud_3d->size(); i++) {
        pcl::PointXYZI point_3d = lidar_line_cloud_3d->points[i];
        pts_3d.emplace_back(cv::Point3d(point_3d.x, point_3d.y, point_3d.z));
    }
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
    /*cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);*/
    cv::Mat r_vec =
        (cv::Mat_<double>(3, 1)
            << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
        extrinsic_params[4], extrinsic_params[5]);
    // project 3d-points into image view
    std::vector<cv::Point2d> pts_2d;
    // debug
    // std::cout << "camera_matrix:" << camera_matrix << std::endl;
    // std::cout << "distortion_coeff:" << distortion_coeff << std::endl;
    // std::cout << "r_vec:" << r_vec << std::endl;
    // std::cout << "t_vec:" << t_vec << std::endl;
    // std::cout << "pts 3d size:" << pts_3d.size() << std::endl;
    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff, pts_2d);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr line_edge_cloud_2d(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<int> line_edge_cloud_2d_number;
    for (size_t i = 0; i < pts_2d.size(); i++) {
        pcl::PointXYZRGB p;
        p.x = pts_2d[i].x;
        p.y = -pts_2d[i].y;
        p.z = 0;
        pcl::PointXYZI pi_3d;
        pi_3d.x = pts_3d[i].x;
        pi_3d.y = pts_3d[i].y;
        pi_3d.z = pts_3d[i].z;
        pi_3d.intensity = 1;
        if (p.x > 0 && p.x < width_ && pts_2d[i].y > 0 && pts_2d[i].y < height_) {
            if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) {
                line_edge_cloud_2d->points.push_back(p);
                line_edge_cloud_2d_number.push_back(plane_line_number_[i]);
                img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
            }
            else
            {
                img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
            }
        }
    }
    if (show_residual)
    {
        cv::Mat residual_img = getConnectImg(dis_threshold, cam_edge_cloud_2d, line_edge_cloud_2d);
        std::string FileName = std::string("./residual/") + std::to_string(IterNum) + std::string("residual.png");
        IterNum++;
        cv::imwrite(FileName, residual_img);
        // cv::imshow("residual", residual_img);
        // cv::waitKey(0);
    }
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(
        new pcl::search::KdTree<pcl::PointXYZRGB>());
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree_lidar(
        new pcl::search::KdTree<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr search_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tree_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tree_cloud_lidar = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    kdtree->setInputCloud(cam_edge_cloud_2d);
    kdtree_lidar->setInputCloud(line_edge_cloud_2d);
    tree_cloud = cam_edge_cloud_2d;
    tree_cloud_lidar = line_edge_cloud_2d;
    search_cloud = line_edge_cloud_2d;
    // 指定近邻个数
    int K = 5;
    // 创建两个向量，分别存放近邻的索引值、近邻的中心距
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    std::vector<int> pointIdxNKNSearchLidar(K);
    std::vector<float> pointNKNSquaredDistanceLidar(K);
    int match_count = 0;
    double mean_distance;
    int line_count = 0;
    std::vector<cv::Point2d> lidar_2d_list;
    std::vector<cv::Point2d> img_2d_list;
    std::vector<Eigen::Vector2d> camera_direction_list;
    std::vector<Eigen::Vector2d> lidar_direction_list;
    std::vector<int> lidar_2d_number;
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
        pcl::PointXYZRGB searchPoint = search_cloud->points[i];
        if ((kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
            pointNKNSquaredDistance) > 0) &&
            (kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar,
                pointNKNSquaredDistanceLidar) > 0)) {
            bool dis_check = true;
            for (int j = 0; j < K; j++) {
                float distance = sqrt(
                    pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
                    pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
                if (distance > dis_threshold) {
                    dis_check = false;
                }
            }
            if (dis_check) {
                cv::Point p_l_2d(search_cloud->points[i].x, -search_cloud->points[i].y);
                cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[0]].x,
                    -tree_cloud->points[pointIdxNKNSearch[0]].y);
                Eigen::Vector2d direction_cam(0, 0);
                std::vector<Eigen::Vector2d> points_cam;
                for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
                    Eigen::Vector2d p(tree_cloud->points[pointIdxNKNSearch[i]].x,
                        -tree_cloud->points[pointIdxNKNSearch[i]].y);
                    points_cam.push_back(p);
                }
                calcDirection(points_cam, direction_cam);
                Eigen::Vector2d direction_lidar(0, 0);
                std::vector<Eigen::Vector2d> points_lidar;
                for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
                    Eigen::Vector2d p(
                        tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x,
                        -tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
                    points_lidar.push_back(p);
                }
                calcDirection(points_lidar, direction_lidar);
                // direction.normalize();
                if (checkFov(p_l_2d)) {
                    lidar_2d_list.push_back(p_l_2d);
                    img_2d_list.push_back(p_c_2d);
                    camera_direction_list.push_back(direction_cam);
                    lidar_direction_list.push_back(direction_lidar);
                    lidar_2d_number.push_back(line_edge_cloud_2d_number[i]);
                }
            }
        }
    }
    for (size_t i = 0; i < lidar_2d_list.size(); i++)
    {
        int y = lidar_2d_list[i].y;
        int x = lidar_2d_list[i].x;
        int pixel_points_size = img_pts_container[y][x].size();
        if (pixel_points_size > 0) {
            VPnPData pnp;
            pnp.x = 0;
            pnp.y = 0;
            pnp.z = 0;
            pnp.u = img_2d_list[i].x;
            pnp.v = img_2d_list[i].y;
            for (size_t j = 0; j < pixel_points_size; j++) {
                pnp.x += img_pts_container[y][x][j].x;
                pnp.y += img_pts_container[y][x][j].y;
                pnp.z += img_pts_container[y][x][j].z;
            }
            pnp.x = pnp.x / pixel_points_size;
            pnp.y = pnp.y / pixel_points_size;
            pnp.z = pnp.z / pixel_points_size;
            pnp.direction = camera_direction_list[i];
            pnp.direction_lidar = lidar_direction_list[i];
            pnp.number = lidar_2d_number[i];
            float theta = pnp.direction.dot(pnp.direction_lidar);
            if (theta > direction_theta_min_ || theta < direction_theta_max_) {
                pnp_list.push_back(pnp);
            }
        }
    }
}

void Calibration::calcDirection(const std::vector<Eigen::Vector2d>& points, Eigen::Vector2d& direction)
{
    Eigen::Vector2d mean_point(0, 0);
    for (size_t i = 0; i < points.size(); i++) {
        mean_point(0) += points[i](0);
        mean_point(1) += points[i](1);
    }
    mean_point(0) = mean_point(0) / points.size();
    mean_point(1) = mean_point(1) / points.size();
    Eigen::Matrix2d S;
    S << 0, 0, 0, 0;
    for (size_t i = 0; i < points.size(); i++) {
        Eigen::Matrix2d s =
            (points[i] - mean_point) * (points[i] - mean_point).transpose();
        S += s;
    }
    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
    Eigen::MatrixXcd evecs = es.eigenvectors();
    Eigen::MatrixXcd evals = es.eigenvalues();
    Eigen::MatrixXd evalsReal;
    evalsReal = evals.real();
    Eigen::MatrixXf::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax); //得到最大特征值的位置
    direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
}

cv::Mat Calibration::getProjectionImg(const Vector6d& extrinsic_params)
{
    cv::Mat depth_projection_img;//似乎应该叫做intensity projection image
    projection(extrinsic_params, raw_lidar_cloud_, INTENSITY, false, depth_projection_img);

    cv::Mat merge_img = image_.clone();
    for (int x = 0; x < merge_img.cols; x++)
    {
        for (int y = 0; y < merge_img.rows; y++)
        {
            uint8_t r, g, b;
            float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
            if (norm > 0)
            {
                mapJet(norm, 0, 1, r, g, b);//利用intensity生成伪彩色图像
                merge_img.at<cv::Vec3b>(y, x)[0] = b;
                merge_img.at<cv::Vec3b>(y, x)[1] = g;
                merge_img.at<cv::Vec3b>(y, x)[2] = r;
            }
        }
    }
    return merge_img;
}

// cv::Mat Calibration::getProjectionImg(const Vector6d &extrinsic_params)
// {
//   cv::Mat depth_projection_img;//似乎应该叫做intensity projection image
//   projection(extrinsic_params, raw_lidar_cloud_, INTENSITY, false, depth_projection_img);
//   cv::Mat map_img = cv::Mat::zeros(height_, width_, CV_8UC3);
//   for (int x = 0; x < map_img.cols; x++)
//   {
//     for (int y = 0; y < map_img.rows; y++)
//     {
//       uint8_t r, g, b;
//       float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
//       mapJet(norm, 0, 1, r, g, b);//利用intensity生成伪彩色图像
//       ///////////////////////////////////////////
//       cv::Scalar color = cv::Scalar(b, g, r);
//       int thickness = 2;
//       int radius = 2;
//       cv::Point center(y, x);
//       cv::circle(map_img, center, radius, color, thickness);
//       //////////////////////////////////////////
//       // map_img.at<cv::Vec3b>(y, x)[0] = b;
//       // map_img.at<cv::Vec3b>(y, x)[1] = g;
//       // map_img.at<cv::Vec3b>(y, x)[2] = r;
//     }
//   }
//   cv::Mat merge_img;
//   if (image_.type() == CV_8UC3)
//   {
//     merge_img = 0.5 * map_img + 0.8 * image_;//图像融合非常impressive
//   }
//   else
//   {
//     cv::Mat src_rgb;
//     cv::cvtColor(image_, src_rgb, cv::COLOR_GRAY2BGR);
//     merge_img = 0.5 * map_img + 0.8 * src_rgb;
//   }
//   return merge_img;
// }

void Calibration::calcResidual(const Vector6d& extrinsic_params,
    const std::vector<VPnPData> vpnp_list,
    std::vector<float>& residual_list)
{
    residual_list.clear();
    Eigen::Vector3d euler_angle(extrinsic_params[0], extrinsic_params[1],
        extrinsic_params[2]);
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    Eigen::Vector3d transation(extrinsic_params[3], extrinsic_params[4],
        extrinsic_params[5]);
    for (size_t i = 0; i < vpnp_list.size(); i++) {
        Eigen::Vector2d residual;
        Eigen::Matrix2f var;
        calcCovarance(extrinsic_params, vpnp_list[i], 1, 0.02, 0.05, var);
        VPnPData vpnp_point = vpnp_list[i];
        float fx = fx_;
        float cx = cx_;
        float fy = fy_;
        float cy = cy_;
        Eigen::Matrix3d inner;
        inner << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        Eigen::Vector4d distor;
        distor << k1_, k2_, p1_, p2_;
        Eigen::Vector3d p_l(vpnp_point.x, vpnp_point.y, vpnp_point.z);
        Eigen::Vector3d p_c = rotation_matrix * p_l + transation;
        Eigen::Vector3d p_2 = inner * p_c;
        float uo = p_2[0] / p_2[2];
        float vo = p_2[1] / p_2[2];
        float xo = (uo - cx) / fx;
        float yo = (vo - cy) / fy;
        float r2 = xo * xo + yo * yo;
        float r4 = r2 * r2;
        float distortion = 1.0 + distor[0] * r2 + distor[1] * r4;
        float xd = xo * distortion + (distor[2] * xo * yo + distor[2] * xo * yo) +
            distor[3] * (r2 + xo * xo + xo * xo);
        float yd = yo * distortion + distor[2] * xo * yo + distor[2] * xo * yo +
            distor[2] * (r2 + yo * yo + yo * yo);
        float ud = fx * xd + cx;
        float vd = fy * yd + cy;
        residual[0] = ud - vpnp_point.u;
        residual[1] = vd - vpnp_point.v;
        // if (vpnp_point.direction(0) == 0 && vpnp_point.direction(1) == 0) {
        //   residual[0] = ud - vpnp_point.u;
        //   residual[1] = vd - vpnp_point.v;
        // } else {
        //   residual[0] = ud - vpnp_point.u;
        //   residual[1] = vd - vpnp_point.v;
        //   Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        //   Eigen::Vector2d n = vpnp_point.direction;
        //   Eigen::Matrix2d V = n * n.transpose();
        //   Eigen::Matrix2d R;
        //   R << residual[0], 0, 0, residual[1];
        //   V = I - V;
        //   R = V * R * V.transpose();
        //   residual[0] = R(0, 0);
        //   residual[1] = R(1, 1);
        // }
        // Eigen::Vector2d v(-vpnp_point.direction(1), vpnp_point.direction(0));
        // if (v(0) < 0) {
        //   v = -v;
        // }
        float cost = sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
        residual_list.push_back(cost);
    }
}

void Calibration::calcCovarance(const Vector6d& extrinsic_params,
    const VPnPData& vpnp_point,
    const float pixel_inc, const float range_inc,
    const float degree_inc,
    Eigen::Matrix2f& covarance) {
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    Eigen::Vector3d transation(extrinsic_params[3], extrinsic_params[4],
        extrinsic_params[5]);
    float fx = fx_;
    float cx = cx_;
    float fy = fy_;
    float cy = cy_;
    Eigen::Vector3f p_l(vpnp_point.x, vpnp_point.y, vpnp_point.z);
    Eigen::Vector3f p_c = rotation.cast<float>() * p_l + transation.cast<float>();

    Eigen::Matrix2f var_camera_pixel;
    var_camera_pixel << pow(pixel_inc, 2), 0, 0, pow(pixel_inc, 2);

    Eigen::Matrix2f var_lidar_pixel;
    float range = sqrt(vpnp_point.x * vpnp_point.x + vpnp_point.y * vpnp_point.y +
        vpnp_point.z * vpnp_point.z);
    Eigen::Vector3f direction(vpnp_point.x, vpnp_point.y, vpnp_point.z);
    direction.normalize();
    Eigen::Matrix3f direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0,
        -direction(0), -direction(1), direction(0), 0;
    float range_var = range_inc * range_inc;
    Eigen::Matrix2f direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
        pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3f base_vector1(1, 1,
        -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3f base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<float, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
        base_vector1(2), base_vector2(2);
    Eigen::Matrix<float, 3, 2> A = range * direction_hat * N;
    Eigen::Matrix3f lidar_position_var =
        direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
    Eigen::Matrix3f lidar_position_var_camera =
        rotation.cast<float>() * lidar_position_var *
        rotation.transpose().cast<float>();
    Eigen::Matrix2f lidar_pixel_var_2d;
    Eigen::Matrix<float, 2, 3> B;
    B << fx / p_c(2), 0, fx* p_c(0) / pow(p_c(2), 2), 0, fy / p_c(2),
        fy* p_c(1) / pow(p_c(2), 2);
    var_lidar_pixel = B * lidar_position_var * B.transpose();
    covarance = var_camera_pixel + var_lidar_pixel;
}

