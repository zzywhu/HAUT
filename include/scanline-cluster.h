#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <cmath>
#include <vector>
#include <forward_list>
#include <iostream>
#include <Eigen/Core>
#include<groundremove.h>
#include <opencv2/opencv.hpp>

// Custom Point Structure for holding clustered points
namespace scan_line_run
{
    /** Euclidean Velodyne coordinate, including intensity and ring number, and label. */
    struct PointXYZIRL
    {
        float x, y, z;        // Coordinates
        float intensity;      ///< Laser intensity reading
        uint16_t ring;        ///< Laser ring number
        uint16_t label;       ///< Point label
    };
};

#define SLRPointXYZIRL scan_line_run::PointXYZIRL
#define RUN std::vector<SLRPointXYZIRL>

void saveIntensityAsPng(const std::vector<std::vector<SLRPointXYZIRL>>& laser_frame, const std::string& filename) {
    int height = 16;    // sensor_model_
    int width = 4000; // Assume uniform width (2251)

    // Create a matrix to store intensity data
    cv::Mat intensity_image(height, width, CV_8UC1, cv::Scalar(0));

    // Find min and max intensity values
    float min_intensity = std::numeric_limits<float>::max();
    float max_intensity = std::numeric_limits<float>::lowest();

    for (const auto& row : laser_frame) {
        for (const auto& point : row) {
            min_intensity = std::min(min_intensity, point.intensity);
            max_intensity = std::max(max_intensity, point.intensity);
        }
    }

    // Normalize intensity to [0, 255] and populate the matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float normalized_intensity = (laser_frame[i][j].intensity - min_intensity) / 
                                         (max_intensity - min_intensity);
            intensity_image.at<uchar>(i, j) = static_cast<uchar>(normalized_intensity * 255);
        }
    }
    //cv::resize(intensity_image,intensity_image,cv::Size(2500,128),0,0,cv::INTER_CUBIC);

    // Save the intensity image as a PNG file
    cv::imwrite(filename, intensity_image);
}

int calculate_ring(const SLRPointXYZIRL& point, float v_fov_min, float v_fov_max, int num_rings) {
    // 计算点的垂直角度（单位：弧度）
    float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    float theta = std::asin(point.z / distance); // 计算垂直角度

    // 将角度范围限制在 [v_fov_min, v_fov_max] 内
    float theta_deg = theta * 180.0 / M_PI; // 转为角度制
    
    // 将角度映射到 [0, num_rings - 1]
    int ring = static_cast<int>(std::round(((theta_deg - v_fov_min) / (v_fov_max - v_fov_min)) * (num_rings-1)));
    if (ring <0) {
        return 0; // 超出垂直视场范围，返回无效 ring
    }
    if (ring >15) {
        return 15; // 超出垂直视场范围，返回无效 ring
    }

    return ring;
}

// Helper function for distance computation
double dist(const SLRPointXYZIRL& a, const SLRPointXYZIRL& b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)+(a.z - b.z)*(a.z - b.z));
}
void sortNgIdx(std::vector<std::vector<int>>& ng_idx_) {
    // 遍历每个子向量并进行排序
    for (auto& sub_vector : ng_idx_) {
        std::sort(sub_vector.begin(), sub_vector.end());
    }
}

class ScanLineRun
{
public:
    ScanLineRun();
    void process_pointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_pc_in, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>>& clusters);

private:
    int sensor_model_;
    double th_run_;      // Threshold of distance for points to belong to the same run
    double th_merge_;    // Threshold of distance for runs to be merged

    // For organizing points
    std::vector<std::vector<SLRPointXYZIRL>> laser_frame_;
    std::vector<std::list<SLRPointXYZIRL*>> runs_;   // Holds all runs
    uint16_t max_label_;  // Max run labels, for distinguishing different runs
    std::vector<std::vector<int>> ng_idx_;  // Non-ground point indices
    void check_zero_labels(int scan_line);
    // For finding runs on a scanline
    void find_runs_(int scan_line);
    // For merging runs
    void merge_runs_(uint16_t cur_label, uint16_t target_label);
    // For updating points cluster label after merge action
    void update_labels_(int scan_line);
};

// Constructor: Initializes the basic parameters and structures
ScanLineRun::ScanLineRun()
    : sensor_model_(16), th_run_(0.20), th_merge_(0.30), max_label_(1)
{
    laser_frame_.resize(sensor_model_, std::vector<SLRPointXYZIRL>(4000));
    runs_.resize(2); // Add dummy runs for non-ground and ground points
}

void ScanLineRun::check_zero_labels(int scan_line)
{
    int point_size = laser_frame_[scan_line].size();
    
    // Iterate through each point in the scan line to check for label 0
    for (int i = 0; i < point_size; ++i)
    {
        auto& p = laser_frame_[scan_line][i];

        // Check if the label of the current point is 0
        if (p.label == 0)
        {
            // Print out information about the point with label 0
            cout<<"find 0"<<endl;
            break;
            // Optional: Add any other handling code, such as adding to a list or flagging for review
        }
    }
}

// Process a point cloud and output clusters in pcl::PointCloud<pcl::PointXYZRGB>::Ptr format
void ScanLineRun::process_pointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_pc_in, 
                                     std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>>& clusters)
{
    std::vector<unsigned int> labels;

ng_idx_.clear();
ng_idx_.resize(sensor_model_);  // Resize the non-ground point index array based on the sensor model

GroundPlaneFit gpf;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZRGB>);
gpf.process(pcl_pc_in, pcl_pc_out, labels);
cout << "Labels size: " << labels.size() << endl;
//pcl::io::savePCDFileASCII("check/" + std::to_string(dataProcessingNum) + "_groundremove.pcd", *pcl_pc_out);

// Dummy filling for demonstration purposes
double range = 0;
int row = 0;
for (size_t i = 0; i < pcl_pc_in->points.size(); ++i) {
    const auto& pt = pcl_pc_in->points[i];
    SLRPointXYZIRL new_pt;
    new_pt.x = pt.x;
    new_pt.y = pt.y;
    new_pt.z = pt.z;
    // new_pt.x = 2.964816;
    // new_pt.y = -0.482851;
    // new_pt.z = 0.157427;
    new_pt.intensity = pt.intensity;
    new_pt.label = labels[i]; // Initialize label to 0 (non-ground)

    // Calculate the ring (scan line) for the point
    int ring = calculate_ring(new_pt, -15, 15, sensor_model_);
    new_pt.ring = ring;

    // Compute and angle. 
            // @Note: In this case, `x` points right and `y` points forward.
            range = sqrt(new_pt.x*new_pt.x + new_pt.y*new_pt.y + new_pt.z*new_pt.z);
            if(new_pt.y<=0){
                row = int(std::round(2000 - asin(new_pt.x/range)/0.00157079633));
            }else if(new_pt.y>0&&new_pt.x>0){
                row = int(std::round(0 + asin(new_pt.x/range)/0.00157079633));
            }else if(new_pt.y>0&&new_pt.x<0){
                row = int(std::round(3999+asin(new_pt.x/range)/0.00157079633));
            }
            

    // if (row > 2250 || row < 0) {
    //     //ROS_ERROR("Row: %d is out of index.", row);
    //     return;
    // } 
    if(laser_frame_[new_pt.ring][row].x!=0)
    {
        continue;
    }
    else {
        laser_frame_[new_pt.ring][row] = new_pt; // Insert into laser_frame at the computed row
    }

    

    // Process non-ground points
    if (new_pt.label != 1u) { // Assuming label 1 is ground
        ng_idx_[new_pt.ring].push_back(row); // Store non-ground points by their ring
    } else {
        runs_[1].push_back(&laser_frame_[new_pt.ring][row]); // Add ground points to the ground run
    }
}
    sortNgIdx(ng_idx_);
    //check
    saveIntensityAsPng(laser_frame_, "check/" + std::to_string(dataProcessingNum) + "_intensityimg.png");

    // Process each scanline
    for (int scan_line = 0; scan_line < sensor_model_; ++scan_line) {
        find_runs_(scan_line);
        //check_zero_labels(scan_line);
        update_labels_(scan_line);
    }
    for (int scan_line = 0; scan_line < sensor_model_; ++scan_line) {
        //find_runs_(scan_line);
        //check_zero_labels(scan_line);
        //update_labels_(scan_line);
    }

    for (const auto& run : runs_) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    cluster->height = 1;
    cluster->width = run.size(); // Set the width directly from the number of points in the run

    // Generate a random color for the entire cluster
    uint8_t r = rand() % 256;
    uint8_t g = rand() % 256;
    uint8_t b = rand() % 256;

    for (auto* pt : run) {
        pcl::PointXYZRGB pcl_pt;
        pcl_pt.x = pt->x;
        pcl_pt.y = pt->y;
        pcl_pt.z = pt->z;

        // Set the same color for all points in the cluster
        pcl_pt.r = r;
        pcl_pt.g = g;
        pcl_pt.b = b;

        cluster->points.push_back(pcl_pt);
    }
    clusters.push_back(cluster);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    merged_cloud->height=1;
    merged_cloud->width=0;
    for (int i=1;i<clusters.size();i++) {
        // Append all points from the cluster to the merged cloud
        for(int j=0;j<clusters[i]->points.size();j++)
        {
          merged_cloud->points.push_back(clusters[i]->points[j]);  
        }
        merged_cloud->width+=clusters[i]->width;
    }


    /*** Output the clusters with different colors ***/
        //std::string filename = "cluster_" + std::to_string(i) + ".pcd";
        //pcl::io::savePCDFileASCII(filename, *clusters[i]);
    pcl::io::savePCDFileASCII("check/" + std::to_string(dataProcessingNum) + "_cluster.pcd", *merged_cloud);
    cout<<"cluster finished"<<endl;
}

// Find runs for a particular scan line
void ScanLineRun::find_runs_(int scan_line)
{
    int point_size = ng_idx_[scan_line].size();
    if (point_size <= 0)
        return;

    int non_g_pt_idx = ng_idx_[scan_line][0]; // First non-ground point
    int non_g_pt_idx_l = ng_idx_[scan_line][point_size - 1]; // Last non-ground point

    for (int i_idx = 0; i_idx < point_size - 1; ++i_idx)
    {
        int i = ng_idx_[scan_line][i_idx];
        int i1 = ng_idx_[scan_line][i_idx + 1];

        // First point in the run
        if (i_idx == 0)
        {
            auto& p_0 = laser_frame_[scan_line][i];
            max_label_ += 1;
            laser_frame_[scan_line][i].label = max_label_;
            runs_.push_back({});
            runs_[p_0.label].push_back(&laser_frame_[scan_line][i]);
        }

        auto& p_i = laser_frame_[scan_line][i];
        auto& p_i1 = laser_frame_[scan_line][i1];

        // Skip if next point is ground
        if (p_i1.label == 1u) {
            runs_[p_i1.label].push_back(&laser_frame_[scan_line][i1]);
            continue;
        }

        // If points are within threshold, belong to the same run
        if (dist(p_i, p_i1) < th_run_) {
            p_i1.label = p_i.label;
        }
        else {
            max_label_ += 1;
            laser_frame_[scan_line][i1].label = max_label_;
            runs_.push_back({});
        }

        runs_[p_i1.label].push_back(&laser_frame_[scan_line][i1]);
    }
    //check_zero_labels(scan_line);

    if (point_size > 1)
    {
        auto& p_0 = laser_frame_[scan_line][non_g_pt_idx];
        auto& p_l = laser_frame_[scan_line][non_g_pt_idx_l];

        if (p_0.label == 1u || p_l.label == 1u)
        {
            return;
        }
        else if (dist(p_0, p_l) < th_run_)
        {
            merge_runs_(p_l.label, p_0.label);
        }
    }
    else if (point_size == 1)
    {
        auto& p_0 = laser_frame_[scan_line][non_g_pt_idx];
        max_label_ += 1;
        //p_0.label = max_label_;
        runs_.push_back({});
        laser_frame_[scan_line][non_g_pt_idx].label = max_label_;
        runs_[p_0.label].push_back(&laser_frame_[scan_line][non_g_pt_idx]);
    }
    //check_zero_labels(scan_line);
    for(int i=2;i<runs_.size();i++)
    {
        if(runs_[i].size()<10)
        {
            runs_[i].clear();
            for(int j=0;j<laser_frame_[scan_line].size();j++)
            {
                if( laser_frame_[scan_line][j].label==i)
                {
                    laser_frame_[scan_line][j].label=0;
                }
            }
        }
    }
}

// Merge two runs
void ScanLineRun::merge_runs_(uint16_t cur_label, uint16_t target_label)
{
    if (cur_label == 0 || target_label == 0)
    {
        std::cerr << "Error merging runs: " << cur_label << " " << target_label << std::endl;
    }

    for (auto& p : runs_[cur_label]) {
        p->label = target_label;
    }

    runs_[target_label].insert(runs_[target_label].end(), runs_[cur_label].begin(), runs_[cur_label].end());
    runs_[cur_label].clear();
}

// Update point labels based on nearby points
void ScanLineRun::update_labels_(int scan_line)
{
    int point_size_j_idx = ng_idx_[scan_line].size();
    // Current scan line is emtpy, do nothing.
    if(point_size_j_idx==0) return;

    // Iterate each point of this scan line to update the labels.
    for(int j_idx=5;j_idx<point_size_j_idx-5;j_idx++){
        int j = ng_idx_[scan_line][j_idx];

        auto &p_j = laser_frame_[scan_line][j];

        // Runs above from scan line 0 to scan_line
        for(int l=scan_line-1;l>=0;l--){
            if(ng_idx_[l].size()==0)
                continue;

            // Smart index for the near enough point, after re-organized these points.
            int nn_idx = j;

            if(laser_frame_[l][nn_idx].intensity ==-1 || laser_frame_[l][nn_idx].label == 1u){
                continue;
            }

            // Nearest neighbour point
            SLRPointXYZIRL p_nn = laser_frame_[l][nn_idx];
            // Skip, if these two points already belong to the same run.
            if(p_j.label == p_nn.label||p_nn.label==0){
                continue;
            }
            double dist_min=9999;
            for(int z=-5;z<5;z++)
            {
                p_nn = laser_frame_[l][nn_idx+z];
                if(dist(p_j, p_nn)<dist_min)
                {
                    dist_min=dist(p_j, p_nn);
                }

            }
            p_nn = laser_frame_[l][nn_idx];
            double dist_min_max = dist(p_j, p_nn);
            double dist_check=sqrt(p_j.x*p_j.x+p_j.y*p_j.y+p_j.z*p_j.z);
            
            double delta_th=dist_check/10;
            double max_th=0.5;
            th_merge_=0.25+delta_th*0.1;
            max_th= 0.5+delta_th*0.1;
            /* Otherwise,
            If the distance of the `nearest point` is within `th_merge_`, 
            then merge to the smaller run.
            */

            if(dist_min < th_merge_&&dist_min_max<max_th){
                uint16_t  cur_label = 0, target_label = 0;

                if(p_j.label ==0 || p_nn.label==0){
                    //ROS_ERROR("p_j.label:%u, p_nn.label:%u", p_j.label, p_nn.label);
                    continue;
                }
                // Merge to a smaller label cluster
                if(p_j.label > p_nn.label){
                    cur_label = p_j.label;
                    target_label = p_nn.label;
                }else{
                    cur_label = p_nn.label;
                    target_label = p_j.label;
                }

                // Merge these two runs.
                merge_runs_(cur_label, target_label);
            }
        }
    }
}

