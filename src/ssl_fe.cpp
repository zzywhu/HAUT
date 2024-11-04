#include<iostream>
#include<queue>
#include<pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include "ceres/ceres.h"
#include"common.h"
#include"calib.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include "loadconfig.h"
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include"project.h"
#include "stdio.h"
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
////////////////////////////////////////////global parameters
// multi
// Data path
string image_path;
string pcd_path;
string result_path;
std::vector<cv::Point2d> cluster2d;
std::vector<cv::Point3d> cluster3d;
int data_num;
//// Camera config
//vector<double> camera_matrix;
//vector<double> dist_coeffs;
double width;
double height;

// Calib config
bool use_rough_calib;
string calib_config_file;
// instrins matrix
Eigen::Matrix3d inner;
// Distortion coefficient
Eigen::Vector4d distor;
Eigen::Vector4d quaternion;
Eigen::Vector3d transation;

/////////////////////////////////////////////////
int IterNum = 0;
int dataProcessingNum = 1;

// pnp calib with direction vector
class vpnp_calib 
{
public:
  vpnp_calib(VPnPData p) { pd = p; }
  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) 
    {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    } else 
    {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
      Eigen::Matrix<T, 2, 2> I =
          Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;
      Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 0) = residuals[1];
      R = V * R;
      // Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<float, 2,
      // 2>::Zero().cast<T>(); R.coeffRef(0, 0) = residuals[0];
      // R.coeffRef(1, 1) = residuals[1]; R = V * R * V.transpose();
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 0);
    }
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p)
  {
    return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(new vpnp_calib(p)));
  }

private:
  VPnPData pd;
};
/////////////////////////////////////////////////////////////////////////////////////////////


int main()
{
  // string image_file("/home/linzp/data/calib/erbcalib/1/gen.jpg");
  // string pcd_file("/home/linzp/data/calib/erbcalib/1/cutuse2.pcd");
  // string result_file("/home/linzp/data/calib/erbcalib/1/extrinsic.txt");
  string calib_config_file("param/config_multi.yaml");
  //cv::FileStorage fSettings(calib_config_file, cv::FileStorage::READ);
  string configfile = "param/config_multi.yaml";
  data_num = parseIntFromYAML(configfile, "data_num");
  image_path = parseStringFromYAML(configfile, "image_path");
  pcd_path = parseStringFromYAML(configfile, "pcd_path");
  cout << image_path << endl;
  /*image_path = "E:\\slam20_c_data\\lidar-camera\\lidar-camera-m\\data";
  pcd_path = "E:\\slam20_c_data\\lidar-camera\\lidar-camera-m\\data";*/
  result_path = "result/extrinsic.txt";
  vector<double> camera_matrix({ 1155.12539531, 0.0,1035.8484391 , 0.0, 1154.54311767 ,757.55707028 , 0.0, 0.0, 1.0});
  vector<double>dist_coeffs({ -0.03154265,	-0.01279613, 0.0138742,	-0.00618993, 0});
  bool use_rough_calib = false;
 
  
  //第一步就应该做去畸变！！！
 
  std::vector<Calibration> calibs;
  // Calibration calibra(image_file, pcd_file, calib_config_file);

  for (size_t i = 0; i < data_num; i++)
  {
    string image_file, pcd_file;
    // string image_file, pcd_file, EdgeFile = "";
    image_file = image_path + "/" + std::to_string(i+1) + ".bmp";
    pcd_file = pcd_path + "/" + std::to_string(i+1) + ".pcd";
    cout << image_file << endl;
    Calibration single_calib(image_file, pcd_file, calib_config_file, camera_matrix, dist_coeffs);
    // EdgeFile = pcd_path + "plane_line_cloud" + std::to_string(i+1) + ".pcd";
    // pcl::io::loadPCDFile("check/plane_line_cloud.pcd", *single_calib.plane_line_cloud_);
    // pcl::io::loadPCDFile(EdgeFile, *single_calib.plane_line_cloud_);
    
    /*single_calib.fx_ = camera_matrix[0];
    single_calib.cx_ = camera_matrix[2];
    single_calib.fy_ = camera_matrix[4];
    single_calib.cy_ = camera_matrix[5];
    single_calib.k1_ = dist_coeffs[0];
    single_calib.k2_ = dist_coeffs[1];
    single_calib.p1_ = dist_coeffs[2];
    single_calib.p2_ = dist_coeffs[3];
    single_calib.k3_ = dist_coeffs[4];*/

    calibs.push_back(single_calib);
  }
  if(!keyExistsInYAML(configfile, "initial_guess_result"))
  {calibs[0].calinitialguess(calib_config_file);}
  Eigen::Vector3d init_euler_angle =
      calibs[0].init_rotation_matrix_.eulerAngles(2, 1, 0);
  Eigen::Vector3d init_transation = calibs[0].init_translation_vector_;

  Vector6d calib_params;
  calib_params << init_euler_angle(0), init_euler_angle(1), init_euler_angle(2),
      init_transation(0), init_transation(1), init_transation(2);

    std::vector<PnPData> pnp_list;
    std::vector<VPnPData> vpnp_list;//只留一个

    std::cout<<"Finish prepare!"<<std::endl;
  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  inner << calibs[0].fx_, 0.0, calibs[0].cx_, 0.0, calibs[0].fy_, calibs[0].cy_,
      0.0, 0.0, 1.0;
  distor << calibs[0].k1_, calibs[0].k2_, calibs[0].p1_, calibs[0].p2_;
  R = calibs[0].init_rotation_matrix_;
  T = calibs[0].init_translation_vector_;
  std::cout << "Initial rotation matrix:" << std::endl
            << calibs[0].init_rotation_matrix_ << std::endl;
  std::cout << "Initial translation:"
            << calibs[0].init_translation_vector_.transpose() << std::endl;
    Eigen::Vector3d euler = R.eulerAngles(2, 1, 0);
    //这里似乎是欧拉角？？？
    //bool use_vpnp = true;
    calib_params[0] = euler[0];
    calib_params[1] = euler[1];
    calib_params[2] = euler[2];
    calib_params[3] = T[0];
    calib_params[4] = T[1];
    calib_params[5] = T[2];
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // calibra.colorCloud(calib_params, 1, calibra.image_, calibra.raw_lidar_cloud_, rgb_cloud);
    // saveRGBPointCloud("init_colored_pts.pcd", rgb_cloud);
    //density = 5;
    cv::Mat init_img = calibs[0].getProjectionImg(calib_params);
    //点云强度投影后与图像融合
    cv::imwrite("initcalib.png", init_img);
    // cv::imshow("Initial extrinsic", init_img);
    // cv::waitKey(0);

    int iter = 0;
    // Maximum match distance threshold: 15 pixels
    // If initial extrinsic lead to error over 15 pixels, the algorithm will not
    // work
    int dis_threshold = 20;
    bool opt_flag = true;


    // Iteratively reduce the matching distance threshold
    // for (dis_threshold = 20; dis_threshold > 8; dis_threshold -= 1)
    for (dis_threshold = 30; dis_threshold > 8; dis_threshold -= 1)
    {
        // For each distance, do twice optimization
        for (int cnt = 0; cnt < 2; cnt++)
        {
          std::vector<std::vector<VPnPData>> vpnp_list_vect;
          int vpnp_size = 0;
          for (size_t i = 0; i < data_num; i++)
          {
            std::vector<VPnPData> vpnp_list;
            calibs[i].buildVPnp(calib_params, dis_threshold, true,calibs[i].rgb_egde_cloud_,
                            calibs[i].plane_line_cloud_, vpnp_list);
            vpnp_list_vect.push_back(vpnp_list);
            vpnp_size += vpnp_list.size();
          }

          std::cout << "Iteration:" << iter++ << " Dis:" << dis_threshold
                << " pnp size: " << vpnp_size << std::endl;

          for(int i=0;i<data_num;i++)
          {
            cv::Mat projection_img = calibs[i].getProjectionImg(calib_params);
            std::string FileName = std::string("./projection/") + std::to_string(IterNum-data_num+i+1) + std::string("iter.png");
            cv::imwrite(FileName, projection_img);
          }

          // IterNum++;

          // cv::imshow("Optimization", projection_img);
          // cv::waitKey(0);
          
          Eigen::Vector3d euler_angle(calib_params[0], calib_params[1], calib_params[2]);
          Eigen::Matrix3d opt_init_R;
          opt_init_R = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
                       Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());
            Eigen::Quaterniond q(opt_init_R);
            Eigen::Vector3d ori_t = T;
            double ext[7];
            ext[0] = q.x();
            ext[1] = q.y();
            ext[2] = q.z();
            ext[3] = q.w();
            ext[4] = T[0];
            ext[5] = T[1];
            ext[6] = T[2];
            Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
            Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem problem;

            problem.AddParameterBlock(ext, 4, q_parameterization);
            problem.AddParameterBlock(ext + 4, 3);
            for (size_t i = 0; i < data_num; i++)
            {
              for (auto val : vpnp_list_vect[i])
              {
                ceres::CostFunction *cost_function;
                cost_function = vpnp_calib::Create(val);
                problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
              }
            }
            ceres::Solver::Options options;
            options.preconditioner_type = ceres::JACOBI;
            options.linear_solver_type = ceres::SPARSE_SCHUR;//DENSE_QR
            options.minimizer_progress_to_stdout = true;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            Eigen::Matrix3d rot = m_q.toRotationMatrix();
            euler_angle = rot.eulerAngles(2, 1, 0);
            // std::cout << rot << std::endl;
            // std::cout << m_t << std::endl;
            calib_params[0] = euler_angle[0];
            calib_params[1] = euler_angle[1];
            calib_params[2] = euler_angle[2];
            calib_params[3] = m_t(0);
            calib_params[4] = m_t(1);
            calib_params[5] = m_t(2);
            R = rot;
            T[0] = m_t(0);
            T[1] = m_t(1);
            T[2] = m_t(2);
            Eigen::Quaterniond opt_q(R);
            std::cout << "q_dis:" << RAD2DEG(opt_q.angularDistance(q))
                        << " ,t_dis:" << (T - ori_t).norm() << std::endl;
        // getchar();
        // if (opt_q.angularDistance(q) < DEG2RAD(0.01) &&
        //     (T - ori_t).norm() < 0.005) {
        //   opt_flag = false;
        // }
        // if (!opt_flag) {
        //   break;
        // }
        }
        if (!opt_flag) 
        {
            break;
        }
    }

    R = Eigen::AngleAxisd(calib_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(calib_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(calib_params[2], Eigen::Vector3d::UnitX());
    std::ofstream outfile(result_path);
    for (int i = 0; i < 3; i++) 
    {
        outfile << R(i, 0) << " " << R(i, 1) << " " << R(i, 2) << " " << T[i] << std::endl;
    }
    outfile << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
    cv::Mat opt_img = calibs[0].getProjectionImg(calib_params);
    // cv::imshow("Optimization result", opt_img);
    cv::imwrite("final.png", opt_img);
    Eigen::Matrix3d init_rotation;
    init_rotation << 0, -1.0, 0, 0, 0, -1.0, 1, 0, 0;
    Eigen::Matrix3d adjust_rotation;
    adjust_rotation = init_rotation.inverse() * R;
    Eigen::Vector3d adjust_euler = adjust_rotation.eulerAngles(2, 1, 0);

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud_final(new pcl::PointCloud<pcl::PointXYZRGB>);
    // calibra.colorCloud(calib_params, 1, calibra.image_,calibra.raw_lidar_cloud_, rgb_cloud_final);

    // pcl::io::savePCDFileASCII("finalraw.pcd", *calibra.raw_lidar_cloud_);
    // pcl::io::savePCDFileASCII("final.pcd", *rgb_cloud_final);

    cout<<"calibration finished!!"<<endl;
    
    return 0;
}
// int main(int argc, char** argv)
// {
//     //定义指针cloud
//     //pcl::PointCloud<PType>::Ptr cloud(new pcl::PointCloud<PType>);
// 	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>());
//     if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/linzp/newWS/calib_ws/src/livox_camera_calib/data/pcd/0.pcd", *cloudA) == -1)
// 	{
//         PCL_ERROR("Couldn't read file pcd\n");
//         return(-1);
//     }
//     // //visualization
//     pcl::visualization::PCLVisualizer visualizer;
//     visualizer.addPointCloud<pcl::PointXYZ> (cloudA, "source cloud transformed");
  
//     while (!visualizer.wasStopped ()) 
//     {
//         visualizer.spinOnce ();
//         pcl_sleep(0.01);
//     }

//     return 0;
// }
