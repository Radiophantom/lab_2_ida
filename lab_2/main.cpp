#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv4/opencv2/core.hpp>

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

  for( int i = 0; i < input_dataset.rows; i++ )
  {
    mean_matrix += input_dataset.row(i);
  }
  
  mean_matrix /= input_dataset.rows;
}

double cov_matrix_element( Mat& a_col, Mat& b_col, double& a_col_mean, double& b_col_mean )
{
  double temp_var = 0;

  for( int i = 0; i < a_col.rows; i++ )
  {
    temp_var += ( a_col.at<double>(0,i) - a_col_mean ) * ( b_col.at<double>(0,i) - b_col_mean );
  }

  return( temp_var/a_col.rows );
}

void cov_matrix_calc( Mat& input_dataset, Mat& mean_matrix, Mat& cov_matrix )
{
  Mat a_col;
  Mat b_col;

  for( int i = 0; i < cov_matrix.rows; i++ )
  {
    for( int j = 0; j < cov_matrix.cols; j++ )
    {
      input_dataset.col(i).copyTo(a_col);
      input_dataset.col(j).copyTo(b_col);
      cov_matrix.at<double>(i,j) = cov_matrix_element( a_col, b_col, mean_matrix.at<double>(0,i), mean_matrix.at<double>(0,j) );
    }
  }
}

int find_min_index( vector <double>& dist_vec )
{
  int min_index = 0;
  double min_value = dist_vec[0];

  for( int i = 1; i < dist_vec.size(); i++ )
  {
    if( dist_vec[i] < min_value )
    {
      min_index = i;
      min_value = dist_vec[i];
    }
  }
  
  return( min_index );
}

int find_cluster( Mat& test_row, Mat& setosa_mean_mat, Mat& versicolor_mean_mat, Mat& virginica_mean_mat, Mat& setosa_cov_mat, Mat& versicolor_cov_mat, Mat& virginica_cov_mat )
{
  vector <double> cluster_dist_vector;

  Scalar setosa_s_det     = determinant( setosa_cov_mat     );
  Scalar versicolor_s_det = determinant( versicolor_cov_mat );
  Scalar virginica_s_det  = determinant( virginica_cov_mat  );

  Mat vec_dif       = ( test_row - setosa_mean_mat );
  Mat vec_dif_trans = vec_dif.t( );
  Mat tmp_val       = vec_dif * setosa_cov_mat * vec_dif_trans;
  double fk_dist    = tmp_val.at<double>(0,0) + log( setosa_s_det[0] );
  
  cluster_dist_vector.push_back( fk_dist );

  vec_dif       = ( test_row - versicolor_mean_mat );
  vec_dif_trans = vec_dif.t( );
  tmp_val       = vec_dif * versicolor_cov_mat * vec_dif_trans;
  fk_dist    = tmp_val.at<double>(0,0) + log( versicolor_s_det[0] );
  
  cluster_dist_vector.push_back( fk_dist );

  vec_dif       = ( test_row - virginica_mean_mat );
  vec_dif_trans = vec_dif.t( );
  tmp_val       = vec_dif * virginica_cov_mat * vec_dif_trans;
  fk_dist    = tmp_val.at<double>(0,0) + log( virginica_s_det[0] );
  
  cluster_dist_vector.push_back( fk_dist );

  return( find_min_index( cluster_dist_vector ) );
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
  Mat iris_setosa_dataset     = Mat::zeros( iris_setosa.size(),     4, CV_64FC1 );
  Mat iris_versicolor_dataset = Mat::zeros( iris_versicolor.size(), 4, CV_64FC1 );
  Mat iris_virginica_dataset  = Mat::zeros( iris_virginica.size(),  4, CV_64FC1 );
  Mat iris_test_dataset       = Mat::zeros( test_dataset.size(),    4, CV_64FC1 );

  vec_to_mat( iris_setosa_dataset,     iris_setosa     );
  vec_to_mat( iris_versicolor_dataset, iris_versicolor );
  vec_to_mat( iris_virginica_dataset,  iris_virginica  );
  vec_to_mat( iris_test_dataset,       test_dataset    );

  Mat iris_setosa_mean_matrix     = Mat::zeros( 1, 4, CV_64FC1 );
  Mat iris_versicolor_mean_matrix = Mat::zeros( 1, 4, CV_64FC1 );
  Mat iris_virginica_mean_matrix  = Mat::zeros( 1, 4, CV_64FC1 );

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

  vector <int> result_class_vec;

  for( int i = 0; i < iris_test_dataset.rows; i++ )
  {
    Mat test_row;
    int cluster_num;

    iris_test_dataset.row(i).copyTo( test_row );
    cluster_num = find_cluster( test_row, iris_setosa_mean_matrix, iris_versicolor_mean_matrix, iris_virginica_mean_matrix, iris_setosa_cov_matrix, iris_versicolor_cov_matrix, iris_virginica_cov_matrix );
    result_class_vec.push_back( cluster_num );
  }

  double right_desicions = 0;
  double wrong_desicions = 0;

  for( int i = 0; i < result_class_vec.size(); i++ )
  {
    if( i < 5 )
    {
      if( result_class_vec[i] == 0 ) { right_desicions++; }
      else                           { wrong_desicions++; }
    }
    else if( i < 10 )
    {
      if( result_class_vec[i] == 1 ) { right_desicions++; }
      else                           { wrong_desicions++; }
    }
    else if( i < 15 )
    {
      if( result_class_vec[i] == 2 ) { right_desicions++; }
      else                           { wrong_desicions++; }
    }
  }

  double test_result = ( right_desicions / result_class_vec.size() )*100;
  //printf( "right desicions percent = %.2f\n", test_result );
  cout << "right_desicions_percent = " << test_result << endl;

  return a.exec();
}
