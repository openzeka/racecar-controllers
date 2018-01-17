#include <iostream>
#include <ros/ros.h>
#include <string>
#include <sensor_msgs/LaserScan.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

float bangbang(float desiredDistance, float wallDistance);

using namespace std;
ros::Publisher vescPub;
ackermann_msgs::AckermannDriveStamped msg;
void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
    float leftWallDistance = scan->range_max;

    int rangesCount = scan->ranges.size();  // 360
    int middle = rangesCount / 2;           // 180

    int angle = 0;
    const vector<float> ranges = scan->ranges;
    for(int i = middle + 20; i < rangesCount - 80; ++i) {
        if(!isinf(ranges[i]) && ranges[i] < leftWallDistance) {
            leftWallDistance = ranges[i];
            angle = i;
        }
    }
    printf("left wall distance = %f     angle = %i\n", leftWallDistance, angle);
    printf("bangbang: %f\n", bangbang(0.5, leftWallDistance));

    msg.header.stamp = scan->header.stamp;
    msg.drive.speed = 1;
    msg.drive.steering_angle = bangbang(0.5, leftWallDistance);
    vescPub.publish(msg);
}

float bangbang(float desiredDistance, float wallDistance) {
    float e = wallDistance - desiredDistance;
    if (e < 0) {
        return -2;
    } else {
        return 2;
    }
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "bangbang");
    ros::NodeHandle nh;

    ros::Subscriber scan_sub_ = nh.subscribe("/scan", 10, &scanCallback);

    vescPub = nh.advertise<ackermann_msgs::AckermannDriveStamped>("/ackermann_cmd_mux/input/navigation",10);

    ros::spin(); // keep alive

    return 0;
}
