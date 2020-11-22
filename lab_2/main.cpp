#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

struct iris_flower
{
  double sp_length;
  double sp_width;
  double pt_length;
  double pt_width;
};

iris_flower calc_mean_vector( vector <iris_flower> input_vector )
{
  iris_flower mean_vector = { 0, 0, 0, 0 };
  iris_flower temp_vector = { 0, 0, 0, 0 };
  int input_vec_size = input_vector.size();

  while( !input_vector.empty() )
  {
    temp_vector = input_vector.back();
    input_vector.pop_back();
    mean_vector.sp_length += temp_vector.sp_length;
    mean_vector.sp_width  += temp_vector.sp_width;
    mean_vector.pt_length += temp_vector.pt_length;
    mean_vector.pt_width  += temp_vector.pt_width;
  }
  
  mean_vector.sp_length /= input_vec_size;
  mean_vector.sp_width  /= input_vec_size;
  mean_vector.pt_length /= input_vec_size;
  mean_vector.pt_width  /= input_vec_size;

  return( mean_vector );
}

int main(int argc, char *argv[])
{

  QCoreApplication a(argc, argv);

  setlocale( LC_NUMERIC, "C" );

  vector <iris_flower> iris_setosa;
  vector <iris_flower> iris_versicolor;
  vector <iris_flower> iris_virginica;
  
  vector <iris_flower> test_dataset;

  ifstream iris_dataset( "/home/skr/qt_projects/lab_2/lab_2/iris.txt" );

// read data from file and separate into 3 classes
  while( iris_dataset )
  {
    string cur_str_line;
    getline( iris_dataset, cur_str_line );
    const char *cur_char_line = cur_str_line.c_str();

    iris_flower flower;

    char sp_length  [3];
    char sp_width   [3];
    char pt_length  [3];
    char pt_width   [3];
    char iris_class [15] = { };

    sscanf( cur_char_line, "%[^,] %*[,] %[^,] %*[,] %[^,] %*[,] %[^,] %*[,] %s", sp_length, sp_width, pt_length, pt_width, iris_class );

    flower.sp_length  = strtod( sp_length, NULL );
    flower.sp_width   = strtod( sp_width,  NULL );
    flower.pt_length  = strtod( pt_length, NULL );
    flower.pt_width   = strtod( pt_width,  NULL );

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
      max_rand_index--;
    }
  }

// 
  
  return a.exec();
}
