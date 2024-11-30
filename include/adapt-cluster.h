// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
//#include <pcl/visualization/cloud_viewer.h>

//#define LOG
extern int dataProcessingNum;

int leaf_=1;
float z_axis_min_=-0.8;
float z_axis_max_=2.0;
int cluster_size_min_=3;
int cluster_size_max_=2200000;

const int region_max_ = 10; // Change this value to match how far you want to detect.
int regions_[100];

int frames; clock_t start_time; bool reset = true; //fps

void pointCloudcluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_pc_in,    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> &clusters) {
    /*** Downsampling + ground & ceiling removal ***/
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
    pcl::IndicesPtr pc_indices(new std::vector<int>);
    for (int i = 0; i < pcl_pc_in->size(); ++i) {
        if (i % leaf_ == 0) {
            if (pcl_pc_in->points[i].z >= z_axis_min_ && pcl_pc_in->points[i].z <= z_axis_max_) {
                pc_indices->push_back(i);
            }
        }
    }

    /*** Divide the point cloud into nested circular regions ***/
    boost::array<std::vector<int>, region_max_> indices_array;
    for (int i = 0; i < pc_indices->size(); i++) {
        float range = 0.0;
        for (int j = 0; j < region_max_; j++) {
            float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
                       pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
                       pcl_pc_in->points[(*pc_indices)[i]].z * pcl_pc_in->points[(*pc_indices)[i]].z;
            if (d2 > range * range && d2 <= (range + regions_[j]) * (range + regions_[j])) {
                indices_array[j].push_back((*pc_indices)[i]);
                break;
            }
            range += regions_[j];
        }
    }

    /*** Euclidean clustering ***/
    float tolerance = 0.0;


    for (int i = 0; i < region_max_; i++) {
        tolerance += 0.1;
        if (indices_array[i].size() > cluster_size_min_) {
            boost::shared_ptr<std::vector<int>> indices_array_ptr(new std::vector<int>(indices_array[i]));
            pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
            tree->setInputCloud(pcl_pc_in, indices_array_ptr);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
            ec.setClusterTolerance(tolerance);
            ec.setMinClusterSize(cluster_size_min_);
            ec.setMaxClusterSize(cluster_size_max_);
            ec.setSearchMethod(tree);
            ec.setInputCloud(pcl_pc_in);
            ec.setIndices(indices_array_ptr);
            ec.extract(cluster_indices);

            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
                
                // Set a unique color for each cluster
                uint8_t r = rand() % 256;
                uint8_t g = rand() % 256;
                uint8_t b = rand() % 256;

                for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
                    pcl::PointXYZRGB point;
                    point.x = pcl_pc_in->points[*pit].x;
                    point.y = pcl_pc_in->points[*pit].y;
                    point.z = pcl_pc_in->points[*pit].z;
                    point.r = r;
                    point.g = g;
                    point.b = b;
                    cluster->points.push_back(point);
                }
                cluster->width = cluster->size();
                cluster->height = 1;
                cluster->is_dense = true;
                clusters.push_back(cluster);
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    merged_cloud->height=1;
    merged_cloud->width=0;
    for (int i=0;i<clusters.size();i++) {
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

