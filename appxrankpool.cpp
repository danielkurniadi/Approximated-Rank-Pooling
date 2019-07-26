/* Approximate Rank Pooling (APR) algorithm 
**
** Paper: Dynamic Image Networks for Action Recognition - Section 3.1, Equation (2)
** Author: Daniel Kurniadi et al
**
** Usage: 
** >  ./appxrankpool.exe <YOUR_SRC_IMGS_DIR> <YOUR_DEST_IMGS_DIR>
**
*/

#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <glob.h>

using namespace std;

/************************************************************************************/

void cvApproxRankPooling_DIN(vector<cv::Mat> &images, cv::Mat &arp_img)
{
    size_t T = images.size();

    float h = 0.0;
    float weight;

    cv::Mat sum_img = cv::Mat::zeros(images[0].size(), CV_32FC3);
    
    vector<float> H(T+1);
    vector<cv::Mat> W(T);

    for (size_t t=0; t < T+1; t++) // generate harmonics
    {
        H[t] = h;
        h += 1.0/(t+1);
    }

    cv::Mat w_image, temp;
    for(size_t t=1; t < T+1; t++) // calculate weighted images, index t=1 to T (as seen in paper)
    {
        weight = 2 * (T-t+1) - (T+1) * (H[T] - H[t-1]);  // scalar
        cout << weight << " ";
        temp = images[t-1];
        temp.convertTo(w_image, CV_32FC3);

        w_image = w_image * weight;
        sum_img = sum_img + w_image;
    }

    cv::normalize(sum_img, arp_img, 0, 225.0, cv::NORM_MINMAX, CV_32F);

}

/************************************************************************************/

int main(int argc, char**argv)
{
    // check cli arguments
    if (argc < 3)
    {
        cout << "Argument(s) for src or dest images' folder are missing." << endl;
        cout << "Usage: " << argv[0] << " <SRC_DIR> <DEST_DIR>" << endl;
       return 1;
    }
    
    string src = argv[1];
    string dest = argv[2];
    string fpat = src + "/img*.jpg";  //rgb images
    string outpath;

    stringstream ss;

    vector<cv::String> fnames;
    vector<cv::Mat> images;
    cv::Mat arp_image;  // output of APR operation

    cout <<"Approximate Rank Pooling executables:" << endl;

    glob(fpat, fnames, false);  // recursive false
    cout << "- Found " << fnames.size() << " files in " << src << endl;
    cout << "- Reading files in " << src << "..." << endl;
    
    size_t count = fnames.size();
    for (size_t i=0; i < count; i++)
        images.push_back(imread(fnames[i]));  // append image

    cvApproxRankPooling_DIN(images, arp_image);
    
    ss << std::setw(5) << std::setfill('0') << 0;  // format suffix number

    outpath = dest + "rprgb_" + ss.str() + ".jpg";
    cout << "- Writing Appx Rank Pooling output to: " << outpath << endl;
    
    cv::imwrite(outpath, arp_image);

    return 0;
}
