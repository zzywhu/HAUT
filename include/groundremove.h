#include <iostream>
#include <vector>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

// 定义点类型
struct PointXYZIR {
    float x, y, z, intensity;
    uint16_t ring;
};

using VPoint = PointXYZIR;
using PointCloud = pcl::PointCloud<VPoint>;

int calculate_ring(const PointXYZIR& point, float v_fov_min, float v_fov_max, int num_rings) {
    // 计算点的垂直角度（单位：弧度）
    float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    float theta = std::asin(point.z / distance); // 计算垂直角度

    // 将角度范围限制在 [v_fov_min, v_fov_max] 内
    float theta_deg = theta * 180.0 / M_PI; // 转为角度制
    
    // 将角度映射到 [0, num_rings - 1]
    int ring = static_cast<int>(((theta_deg - v_fov_min) / (v_fov_max - v_fov_min)) * num_rings);
    if (ring <0) {
        return 0; // 超出垂直视场范围，返回无效 ring
    }
    if (ring >15) {
        return 15; // 超出垂直视场范围，返回无效 ring
    }

    return ring;
}

class GroundPlaneFit {
public:
    GroundPlaneFit(int num_lpr = 20, double th_seeds = 0.2, double th_dist = 0.3, int num_iter = 10);

    void process(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_pc_in, 
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_pc_out, 
                 std::vector<unsigned int>& labels);

private:
    void extract_initial_seeds_(const PointCloud& sorted_cloud, PointCloud::Ptr& seeds_cloud);
    void estimate_plane_(const PointCloud::Ptr& ground_cloud);

    int num_lpr_;
    double th_seeds_;
    double th_dist_;
    int num_iter_;

    Eigen::Vector3f normal_;
    float d_;
    float th_dist_d_;
};

// 构造函数
GroundPlaneFit::GroundPlaneFit(int num_lpr, double th_seeds, double th_dist, int num_iter)
    : num_lpr_(num_lpr), th_seeds_(th_seeds), th_dist_(th_dist), num_iter_(num_iter) {}

// 提取初始地面种子点
void GroundPlaneFit::extract_initial_seeds_(const PointCloud& sorted_cloud, PointCloud::Ptr& seeds_cloud) {
    double sum = 0;
    int cnt = 0;

    for (int i = 0; i < sorted_cloud.points.size() && cnt < num_lpr_; ++i) {
        sum += sorted_cloud.points[i].z;
        ++cnt;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0;

    seeds_cloud->clear();
    for (const auto& point : sorted_cloud.points) {
        if (point.z < lpr_height + th_seeds_) {
            seeds_cloud->points.push_back(point);
        }
    }
}

// 估计地面平面
void GroundPlaneFit::estimate_plane_(const PointCloud::Ptr& ground_cloud) {
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    pcl::computeMeanAndCovarianceMatrix(*ground_cloud, cov, pc_mean);

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU);
    normal_ = svd.matrixU().col(2);
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();
    d_ = -(normal_.transpose() * seeds_mean)(0, 0);
    th_dist_d_ = th_dist_ - d_;
}

// 处理点云，分类地面点和非地面点
void GroundPlaneFit::process(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_pc_in, 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_pc_out, 
    std::vector<unsigned int>& labels) 
{
   // 转换输入点云类型
PointCloud::Ptr input_cloud(new PointCloud);
std::vector<size_t> original_indices(pcl_pc_in->points.size()); // 记录原始索引
input_cloud->points.reserve(pcl_pc_in->points.size());
for (size_t i = 0; i < pcl_pc_in->points.size(); ++i) {
    const auto& pt = pcl_pc_in->points[i];
    VPoint new_pt;
    new_pt.x = pt.x;
    new_pt.y = pt.y;
    new_pt.z = pt.z;
    new_pt.intensity = pt.intensity;
    int ring = calculate_ring(new_pt, -15, 15, 16);
    new_pt.ring = ring;  // 初始化 ring
    input_cloud->points.push_back(new_pt);
    original_indices[i] = i; // 记录原始索引
}

// 创建临时存储
PointCloud::Ptr sorted_cloud(new PointCloud(*input_cloud));
std::vector<size_t> sorted_indices(original_indices); // 复制索引

// 排序点云，同时维护索引
std::sort(sorted_indices.begin(), sorted_indices.end(), [&input_cloud](size_t a, size_t b) {
    return input_cloud->points[a].z < input_cloud->points[b].z;
});

// 更新 sorted_cloud 按照排序后的顺序
for (size_t i = 0; i < sorted_indices.size(); ++i) {
    sorted_cloud->points[i] = input_cloud->points[sorted_indices[i]];
}

// 初始化种子点
PointCloud::Ptr seeds_cloud(new PointCloud);
extract_initial_seeds_(*sorted_cloud, seeds_cloud);

PointCloud::Ptr ground_cloud = seeds_cloud;
PointCloud::Ptr non_ground_cloud(new PointCloud);

// 将标签大小初始化为原始点云大小
labels.resize(input_cloud->points.size(), 0);

for (int i = 0; i < num_iter_; ++i) {
    estimate_plane_(ground_cloud);

    ground_cloud->clear();
    non_ground_cloud->clear();

    for (size_t j = 0; j < sorted_cloud->points.size(); ++j) {
        const auto& pt = sorted_cloud->points[j];
        Eigen::Vector3f point(pt.x, pt.y, pt.z);
        float dist = normal_.dot(point) + d_;

        if (dist < th_dist_) {
            ground_cloud->points.push_back(pt);
            //labels[sorted_indices[j]] = 1u;  // 使用排序前的索引设置标签
            labels[sorted_indices[j]] = 0;  // 使用排序前的索引设置标签
        } else {
            non_ground_cloud->points.push_back(pt);
            labels[sorted_indices[j]] = 0;  // 使用排序前的索引设置标签
        }
    }
}

// 创建颜色化点云
pcl_pc_out->clear();
pcl_pc_out->points.reserve(input_cloud->points.size());

for (size_t i = 0; i < input_cloud->points.size(); ++i) {
    pcl::PointXYZRGB color_pt;
    const auto& pt = input_cloud->points[i];

    color_pt.x = pt.x;
    color_pt.y = pt.y;
    color_pt.z = pt.z;

    if (labels[i] == 1) {
        color_pt.r = 0;  // 地面点：绿色
        color_pt.g = 255;
        color_pt.b = 0;
    } else {
        color_pt.r = 255;  // 非地面点：红色
        color_pt.g = 0;
        color_pt.b = 0;
    }

    pcl_pc_out->points.push_back(color_pt);
}

pcl_pc_out->width = pcl_pc_out->points.size();
pcl_pc_out->height = 1;
pcl_pc_out->is_dense = true;

}

