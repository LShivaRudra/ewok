/**
* This file is part of Ewok.
*
* Copyright 2017 Vladyslav Usenko, Technical University of Munich.
* Developed by Vladyslav Usenko <vlad dot usenko at tum dot de>,
* for more information see <http://vision.in.tum.de/research/robotvision/replanning>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* Ewok is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Ewok is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with Ewok. If not, see <http://www.gnu.org/licenses/>.
*/



#include <thread>
#include <chrono>
#include <map>

#include <Eigen/Core>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <std_srvs/Empty.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <visualization_msgs/MarkerArray.h>

#include <ewok/polynomial_3d_optimization.h>
#include <ewok/uniform_bspline_3d_optimization.h>
#include <chrono>
using namespace std::chrono;

const int POW = 7;

double dt ;
int num_opt_points;

bool initialized = false;

ros::Publisher occ_marker_pub, free_marker_pub, dist_marker_pub, trajectory_pub, current_traj_pub;
tf::TransformListener * listener;
ewok::EuclideanDistanceRingBuffer<POW>::Ptr edrb;

void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    auto start = high_resolution_clock::now();
    // ROS_INFO("recieved depth image");
    
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    const float fx = 554.254691191187;
    const float fy = 554.254691191187;
    const float cx = 320.5;
    const float cy = 240.5;

    tf::StampedTransform transform;


    try{

        listener->lookupTransform("map", msg->header.frame_id,
                                  msg->header.stamp, transform);
    }
    catch (tf::TransformException &ex) {
        ROS_INFO("Couldn't get transform");
        ROS_WARN("%s",ex.what());
        return;
    }


    Eigen::Affine3d dT_w_c;
    tf::transformTFToEigen(transform, dT_w_c);

    Eigen::Affine3f T_w_c = dT_w_c.cast<float>();

    float * data = (float *) cv_ptr->image.data;


    auto t1 = std::chrono::high_resolution_clock::now();

    ewok::EuclideanDistanceRingBuffer<POW>::PointCloud cloud1;

    for(int u=0; u < cv_ptr->image.cols; u+=4) {
        // ROS_INFO("Hi");
        for(int v=0; v < cv_ptr->image.rows; v+=4) {
            float val = data[v*cv_ptr->image.cols + u];

            //ROS_INFO_STREAM(val);

            if(std::isfinite(val)) {
                Eigen::Vector4f p;
                p[0] = val*(u - cx)/fx;
                p[1] = val*(v - cy)/fy;
                p[2] = val;
                p[3] = 1;

                p = T_w_c * p;

                cloud1.push_back(p);
            }
        }
    }
    // std::cout<<"The size of cloud is: "<<cloud1.size()<<std::endl;

    Eigen::Vector3f origin = (T_w_c * Eigen::Vector4f(0,0,0,1)).head<3>();

    auto t2 = std::chrono::high_resolution_clock::now();

    if(!initialized) {
        Eigen::Vector3i idx;
        edrb->getIdx(origin, idx);

        ROS_INFO_STREAM("Origin: " << origin.transpose() << " idx " << idx.transpose());

        edrb->setOffset(idx);

        initialized = true;
    } else {
        // ROS_INFO("Hello");
        Eigen::Vector3i origin_idx, offset, diff;
        edrb->getIdx(origin, origin_idx);

        offset = edrb->getVolumeCenter();
        diff = origin_idx - offset;

        while(diff.array().any()) {
            //ROS_INFO("Moving Volume");
            edrb->moveVolume(diff);

            offset = edrb->getVolumeCenter();
            diff = origin_idx - offset;
        }


    }

    //ROS_INFO_STREAM("cloud1 size: " << cloud1.size());


    auto t3 = std::chrono::high_resolution_clock::now();

    edrb->insertPointCloud(cloud1, origin);


    // visualization_msgs::Marker m_occ, m_free;
    // edrb->getMarkerOccupied(m_occ);
    // edrb->getMarkerFree(m_free);


    // occ_marker_pub.publish(m_occ);
    // free_marker_pub.publish(m_free);

    auto end = high_resolution_clock::now();
    auto secs = (end - start);
    auto duration = duration_cast<microseconds>(end-start);
    std::cout<<"Time: "<<duration.count()<<std::endl;
}


int main(int argc, char** argv){
    ros::init(argc, argv, "trajectory_replanning_example");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string path = ros::package::getPath("ewok_simulation") + "/benchmarking/";

    ROS_INFO_STREAM("path: " << path);

    listener = new tf::TransformListener;

    occ_marker_pub = nh.advertise<visualization_msgs::Marker>("ring_buffer/occupied", 5);
    free_marker_pub = nh.advertise<visualization_msgs::Marker>("ring_buffer/free", 5);
    dist_marker_pub = nh.advertise<visualization_msgs::Marker>("ring_buffer/distance", 5);

    message_filters::Subscriber<sensor_msgs::Image> depth_image_sub_ ;
    depth_image_sub_.subscribe(nh, "/iris/camera/depth/image_raw", 5);

    tf::MessageFilter<sensor_msgs::Image> tf_filter_(depth_image_sub_, *listener, "map", 5);
    tf_filter_.registerCallback(depthImageCallback);


    double my_resolution;
    double truncation_distance;
    pnh.param("my_resolution", my_resolution, 0.5);
    pnh.param("truncationdistance", truncation_distance, 1.0);

    edrb.reset(new ewok::EuclideanDistanceRingBuffer<POW>(my_resolution, truncation_distance));

    double distance_threshold;
    pnh.param("distance_threshold", distance_threshold, 0.5);



   
    // std_srvs::Empty srv;
    // bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
    // unsigned int i = 0;

    // // Trying to unpause Gazebo for 10 seconds.
    // while (i <= 10 && !unpaused) {
    //     ROS_INFO("Wait for 1 second before trying to unpause Gazebo again.");
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    //     unpaused = ros::service::call("/gazebo/unpause_physics", srv);
    //     ++i;
    // }

    // if (!unpaused) {
    //     ROS_FATAL("Could not wake up Gazebo.");
    //     return -1;
    // }
    // else {
    //     ROS_INFO("Unpaused the Gazebo simulation.");
    // }

    // Wait for 5 seconds to let the Gazebo GUI show up.
    ros::Duration(5.0).sleep();

    ros::Duration(5.0).sleep();

    ros::spin();

 
}
