#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void vec_to_mat( Mat& mat_matrix, vector <vector <double>>& vec_matrix )
{
  for( int i = 0; i < vec_matrix.size(); i++ )
  {
    mat_matrix.at<double>(i,0) = vec_matrix[i][0];
    mat_matrix.at<double>(i,1) = vec_matrix[i][1];
    mat_matrix.at<double>(i,2) = vec_matrix[i][2];
    mat_matrix.at<double>(i,3) = vec_matrix[i][3];
  }
}

void calc_mean_vector( Mat& input_dataset, Mat& mean_matrix )
{
  int input_dataset_size = input_dataset.cols;

  for( int i = 0; i < input_dataset.cols; i++ )
  {
    mean_matrix.at<double>(0,0) += input_dataset.at<double>(0,i);
    mean_matrix.at<double>(1,0) += input_dataset.at<double>(1,i);
    mean_matrix.at<double>(2,0) += input_dataset.at<double>(2,i);
    mean_matrix.at<double>(3,0) += input_dataset.at<double>(3,i);
  }
  
  mean_matrix.at<double>(0,0) /= input_dataset_size;
  mean_matrix.at<double>(0,0) /= input_dataset_size;
  mean_matrix.at<double>(0,0) /= input_dataset_size;
  mean_matrix.at<double>(0,0) /= input_dataset_size;
}

double mat_element_covariation( Mat& input_matrix, int a_vec_num, int b_vec_num, double& a_vec_mean, double& b_vec_mean )
{
  double temp_sum = 0;

  for( int i = 0; i < input_matrix.cols; i++ )
  {
    temp_sum += ( input_matrix.at<double>(a_vec_num,i) - a_vec_mean ) * ( input_matrix.at<double>(b_vec_num,i) - b_vec_mean );
  }

  temp_sum /= input_matrix.cols;

  return( temp_sum );
}

void cov_matrix_calc( Mat& input_dataset, Mat& mean_matrix, Mat& cov_matrix )
{
  for( int i = 0; i < 4; i++ )
  {
    for( int j = 0; j < 4; j++ )
    {
      cov_matrix.at<double>(i,j) = mat_element_covariation( input_dataset, i, j, mean_matrix.at<double>(0,i), mean_matrix.at<double>(0,j) );
    }
  }
}

vector <double>  find_claster( Mat& test_dataset, Mat& setosa_mean_mat, Mat& versicolor_mean_mat, Mat& virginica_mean_mat, Mat& setosa_cov_mat, Mat& versicolor_cov_mat, Mat& virginica_cov_mat )
{
  vector <double> claster_dist_vector;
  
  Mat vec_dif       = ( test_dataset - setosa_mean_mat );
  Mat vec_dif_trans = vec_dif.t( );
  Mat tmp_mat       = vec_dif * setosa_cov_mat * vec_dif_trans;
  
  claster_dist_vector.push_back( tmp_mat.at<double>(0,0) );

  vec_dif       = ( test_dataset - versicolor_mean_mat );
  vec_dif_trans = vec_dif.t( );
  tmp_mat       = vec_dif * versicolor_cov_mat * vec_dif_trans;
  
  claster_dist_vector.push_back( tmp_mat.at<double>(0,0) );

  vec_dif       = ( test_dataset - virginica_mean_mat );
  vec_dif_trans = vec_dif.t( );
  tmp_mat       = vec_dif * virginica_cov_mat * vec_dif_trans;
  
  claster_dist_vector.push_back( tmp_mat.at<double>(0,0) );

  return( claster_dist_vector );
}

int main(int argc, char *argv[])
{

  QCoreApplication a(argc, argv);

  setlocale( LC_NUMERIC, "C" );

  vector <vector <double>> iris_setosa;
  vector <vector <double>> iris_versicolor;
  vector <vector <double>> iris_virginica;
  
  vector <vector <double>> test_dataset;
  vector <int>             test_dataset_source_class;

  ifstream iris_dataset( "/home/skr/qt_projects/lab_2_ida/lab_2/iris.txt" );

// read data from file and separate into 3 classes
  while( iris_dataset )
  {
    string cur_str_line;
    getline( iris_dataset, cur_str_line );
    const char *cur_char_line = cur_str_line.c_str();

    vector <double> flower;

    char sp_length  [3];
    char sp_width   [3];
    char pt_length  [3];
    char pt_width   [3];
    char iris_class [15] = { };

    sscanf( cur_char_line, "%[^,] %*[,] %[^,] %*[,] %[^,] %*[,] %[^,] %*[,] %s", sp_length, sp_width, pt_length, pt_width, iris_class );

    flower.push_back( strtod( sp_length, NULL ) );
    flower.push_back( strtod( sp_width,  NULL ) );
    flower.push_back( strtod( pt_length, NULL ) );
    flower.push_back( strtod( pt_width,  NULL ) );

    string str_iris_class = iris_class;

    if( str_iris_class == "Iris-setosa" )
      iris_setosa.push_back( flower );
    else if( str_iris_class == "Iris-versicolor" )
      iris_versicolor.push_back( flower );
    else if( str_iris_class == "Iris-virginica" )
      iris_virginica.push_back( flower );
  }

// extract test dataset from all classes
  for( int i = 0; i < 3; i++ )
  {
    int max_rand_index = 50;
    for( int j = 0; j < 5; j++ )
    {
      int index = rand() % max_rand_index;
      if( i == 0 )
      {
        test_dataset.push_back( iris_setosa[index] );
        iris_setosa.erase( iris_setosa.begin() + index ); 
      }
      else if( i == 1 )
      {
        test_dataset.push_back( iris_versicolor[index] );
        iris_versicolor.erase( iris_versicolor.begin() + index ); 
      }
      else if( i == 2 )
      { 
        test_dataset.push_back( iris_virginica[index] );
        iris_virginica.erase( iris_virginica.begin() + index ); 
      }
      test_dataset_source_class.push_back( i );
      max_rand_index--;
    }
  }

// transform vectors into cv::Mat for futher calculations
  Mat iris_setosa_dataset     = Mat::zeros( 4, iris_setosa.size(),     CV_64FC1 );
  Mat iris_versicolor_dataset = Mat::zeros( 4, iris_versicolor.size(), CV_64FC1 );
  Mat iris_virginica_dataset  = Mat::zeros( 4, iris_virginica.size(),  CV_64FC1 );
  Mat iris_test_dataset       = Mat::zeros( 4, test_dataset.size(),    CV_64FC1 );

  vec_to_mat( iris_setosa_dataset,     iris_setosa     );
  vec_to_mat( iris_versicolor_dataset, iris_versicolor );
  vec_to_mat( iris_virginica_dataset,  iris_virginica  );
  vec_to_mat( iris_test_dataset,       test_dataset    );

  Mat iris_setosa_mean_matrix     = Mat::zeros( 4, 1, CV_64FC1 );
  Mat iris_versicolor_mean_matrix = Mat::zeros( 4, 1, CV_64FC1 );
  Mat iris_virginica_mean_matrix  = Mat::zeros( 4, 1, CV_64FC1 );

  calc_mean_vector( iris_setosa_dataset,     iris_setosa_mean_matrix     );
  calc_mean_vector( iris_versicolor_dataset, iris_versicolor_mean_matrix );
  calc_mean_vector( iris_virginica_dataset,  iris_virginica_mean_matrix  );

  Mat iris_setosa_cov_matrix     = Mat::zeros( 4, 4, CV_64FC1 );
  Mat iris_versicolor_cov_matrix = Mat::zeros( 4, 4, CV_64FC1 );
  Mat iris_virginica_cov_matrix  = Mat::zeros( 4, 4, CV_64FC1 );

  cov_matrix_calc( iris_setosa_dataset,     iris_setosa_mean_matrix,     iris_setosa_cov_matrix     );
  cov_matrix_calc( iris_versicolor_dataset, iris_versicolor_mean_matrix, iris_versicolor_cov_matrix );
  cov_matrix_calc( iris_virginica_dataset,  iris_virginica_mean_matrix,  iris_virginica_cov_matrix  );

  invert( iris_setosa_cov_matrix,     iris_setosa_cov_matrix     );
  invert( iris_versicolor_cov_matrix, iris_versicolor_cov_matrix );
  invert( iris_virginica_cov_matrix,  iris_virginica_cov_matrix  );

  Scalar setosa_s_det     = determinant( iris_setosa_cov_matrix     );
  Scalar versicolor_s_det = determinant( iris_versicolor_cov_matrix );
  Scalar virginica_s_det  = determinant( iris_virginica_cov_matrix  );

  double iris_setosa_ln_det     = log( setosa_s_det[0]     );
  double iris_versicolor_ln_det = log( versicolor_s_det[0] );
  double iris_virginica_ln_det  = log( virginica_s_det[0]  );
  
  vector <double> result = find_claster( iris_test_dataset.col(0), iris_setosa_mean_matrix, iris_versicolor_mean_matrix, iris_virginica_mean_matrix, iris_setosa_cov_matrix, iris_versicolor_cov_matrix, iris_virginica_cov_matrix );

  return a.exec();
}
