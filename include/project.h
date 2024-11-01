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
////////////////////////////////////////////////
#include <pcl/point_types.h>
#include <pcl/common/common.h>
using namespace std;
enum ProjectionType { DEPTH, INTENSITY, BOTH };

double min_depth_ = 2.5;
double max_depth_ = 50;
double fx_;
double fy_;
double cx_;
double cy_;

double k1_;
double k2_;
double p1_;
double p2_;
double k3_;


void project(
    const Vector6d& extrinsic_params,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
    const ProjectionType projection_type, const bool is_fill_img,
    cv::Mat& projection_img, cv::Mat image_)
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
    cv::Mat image_project = cv::Mat::zeros(image_.rows, image_.cols, CV_16UC1);
    cv::Mat rgb_image_project = cv::Mat::zeros(image_.rows, image_.cols, CV_8UC3);
    for (size_t i = 0; i < pts_2d.size(); ++i)
    {
        cv::Point2f point_2d = pts_2d[i];
        if (point_2d.x <= 1 || point_2d.x >= image_.cols - 1 || point_2d.y <= 1 || point_2d.y >= image_.rows + 1)
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
cv::Mat lidar2image(const pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud_, cv::Mat image_, const Vector6d& extrinsic_params)
{
    cv::Mat depth_projection_img;//似乎应该叫做intensity projection image
    project(extrinsic_params, raw_lidar_cloud_, INTENSITY, false, depth_projection_img,image_);

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
