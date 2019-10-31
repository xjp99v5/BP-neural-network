// BP00.cpp -- implenting the BP class
// version 0.0

#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<fstream>

#include "BP00.h"

using namespace std; 
//---------------------------------------------------------------------


ofstream Out_W_File("All_W.txt",ios::out) ;
ofstream Out_Error("Error.txt",ios::out) ;


//构造函数,用来初始化权系数，输入，期望输出和学习速度

BP::BP()
{
	srand(time(NULL));//播种，以便产生随即数
	for(int i=1 ; i<Layer_Max ; i++)
	{
		for(int j=0 ; j<Layer_number[i] ; j++)
		{
			for(int k=0 ; k<Layer_number[i-1]+1 ; k++)
			{
				W[i][j][k] = RANDOM;//随机初始化权系数

			}
			//     Q[i][j] = RANDOM ;//初始化各神经元的阀值
		}
	}
	//输入归和输出归一化
	for(int l=0 ; l<InMax ; l++)
	{
		Input_Net[0][l] = l * 0.05 ;//把0～1分成20等分,表示x1
		Input_Net[1][l] = 1 - l * 0.05 ;//表示x2
	}
	for(int i=0 ; i<InMax ; i++)
	{
		for(int j=0 ; j<InMax ; j++)
		{
			Out_Exp[i][j] = Y(Input_Net[0][i],Input_Net[1][j]) ;//期望输出
			Out_Exp[i][j] = Out_Exp[i][j]/3.000000;//期望输出归一化
		}
	}

	Study_Speed=0.5;//初始化学习速度

	e=0.0001;//误差精度


}//end
//激发函数F()
double BP::F(double x)
{
	return(1.0/(1+exp(-x)));
}//end

//要逼近的函数Y()
//输入：两个浮点数
//输出：一个浮点数
double BP::Y(double x1,double x2)
{
	double temp;
	temp = pow(x1,4) + 2 * pow(x2-1,2);
	return temp;
}//end
//--------------------------------------------------------
//代价函数
double BP::Cost(double Out,double Exp)
{
	return(pow(Out-Exp,2));
}//end

//网络输出函数
//输入为：第input个样本
double BP::NetWorkOut(int x1 , int x2)
{
	int i,j,k;
	double N_node[Layer_Max][Neural_Max];
	//约定N_node[i][j]表示网络第i层的第j个神经元的总输入
	//第0层的神经元为输入，不用权系数和阀值，即输进什么即输出什么
	N_node[0][0] = Input_Net[0][x1] ;
	Layer_Node[0][0] = Input_Net[0][x1] ;
	N_node[0][1] = Input_Net[1][x2] ;
	Layer_Node[0][1] = Input_Net[1][x2] ;

	for(i=1 ; i<Layer_Max ; i++)//神经网络的第i层
	{
		for(j=0 ; j<Layer_number[i] ; j++)//Layer_number[i]为第i层的
		{                             //神经元个数
			N_node[i][j] = 0.0;
			for(k=0 ; k<Layer_number[i-1] ; k++)//Layer_number[i-1]
			{            //表示与第i层第j个神经元连接的上一层的
				//神经元个数

				//求上一层神经元对第i层第j个神经元的输入之和
				N_node[i][j]+=Layer_Node[i-1][k] * W[i][j][k];

			}
			N_node[i][j] = N_node[i][j]-W[i][j][k];//减去阀值

			//求Layer_Node[i][j]，即第i层第j个神经元的输出
			Layer_Node[i][j] = F(N_node[i][j]);
		}
	}
	return Layer_Node[Layer_Max-1][0];//最后一层的输出
}//end

//求所有神经元的输出误差微分函数
//输入为：第input个样本
//计算误差微分并保存在D[][]数组中
void BP::AllLayer_D(int x1 , int x2)
{
	int i,j,k;
	double temp;
	D[Layer_Max-1][0] = Layer_Node[Layer_Max-1][0] *
		(1-Layer_Node[Layer_Max-1][0])*
		(Layer_Node[Layer_Max-1][0]-Out_Exp[x1][x2]);
	for(i=Layer_Max-1 ; i>0 ; i--)
	{
		for(j=0 ; j<Layer_number[i-1] ; j++)
		{
			temp = 0 ;
			for(k=0 ; k<Layer_number[i] ; k++)
			{
				temp = temp+W[i][k][j]*D[i][k] ;
			}
			D[i-1][j] = Layer_Node[i-1][j] * (1-Layer_Node[i-1][j])
				*temp ;
		}
	}
}//end
//修改权系数和阀值
void BP::Change_W()
{
	int i,j,k;
	for(i=1 ; i<Layer_Max ; i++)
	{
		for(j=0;j<Layer_number[i];j++)
		{
			for(k=0;k<Layer_number[i-1];k++)
			{
				//修改权系数
				W[i][j][k]=W[i][j][k]-Study_Speed*
					D[i][j]*Layer_Node[i-1][k];

			}
			W[i][j][k]=W[i][j][k]+Study_Speed*D[i][j];//修改阀值
		}
	}
}//end
//训练函数
void BP::Train()
{
	int i,j;
	int ok=0;
	double Out;
	long int count=0;
	double err;
	ofstream Out_count("Out_count.txt",ios::out) ;
	//把其中的5个权系数的变化保存到文件里
	ofstream outWFile1("W[2][0][0].txt",ios::out) ;
	ofstream outWFile2("W[2][1][1].txt",ios::out) ;
	ofstream outWFile3("W[1][0][0].txt",ios::out) ;
	ofstream outWFile4("W[1][1][0].txt",ios::out) ;
	ofstream outWFile5("W[3][0][1].txt",ios::out) ;

	while(ok<441)
	{
		count++;
		//20个样本输入
		for(i=0,ok=0 ; i<InMax ; i++)
		{
			for(j=0 ; j<InMax ; j++)
			{
				Out = NetWorkOut(i,j);

				AllLayer_D(i,j);

				err = Cost(Out,Out_Exp[i][j]);//计算误差

				if(err<e) ok++;  //是否满足误差精度

				else Change_W();//否修改权系数和阀值
			}

		}
		if((count%1000)==0)//每1000次，保存权系数
		{
			cout<<count<<"     "<<err<<endl;
			Out_count<<count<<"," ;
			Out_Error<<err<<"," ;
			outWFile1<<W[2][0][0]<<"," ;
			outWFile2<<W[2][1][1]<<"," ;
			outWFile3<<W[1][0][0]<<"," ;
			outWFile4<<W[1][1][0]<<"," ;
			outWFile5<<W[3][0][1]<<"," ;
			for(int p=1 ; p<Layer_Max ; p++)
			{
				for(int j=0 ; j<Layer_number[p] ; j++)
				{
					for(int k=0 ; k<Layer_number[p-1]+1 ; k++)
					{
						Out_W_File<<'W'<<'['<<p<<']'
							<<'['<<j<<']'
							<<'['<<k<<']'
							<<'='<<W[p][j][k]<<' '<<' ';
					}
				}
			}
			Out_W_File<<'\n'<<'\n' ;
		}

	}
	cout<<err<<endl;
}//end

//打印权系数
void BP::BP_Print()
{
	//打印权系数
	cout<<"训练后的权系数"<<endl;
	for(int i=1 ; i<Layer_Max ; i++)
	{
		for(int j=0 ; j<Layer_number[i] ; j++)
		{
			for(int k=0 ; k<Layer_number[i-1]+1 ; k++)
			{
				cout<<W[i][j][k]<<"         ";
			}
			cout<<endl;
		}
	}
	cout<<endl<<endl;
}//end

//把结果保存到文件
void BP::After_Train_Out()
{
	int i,j ;
	ofstream Out_x1("Out_x1.txt",ios::out) ;

	ofstream Out_x2("Out_x2.txt",ios::out) ;

	ofstream Out_Net("Out_Net.txt",ios::out) ;

	ofstream Out_Exp("Out_Exp.txt",ios::out) ;

	ofstream W_End("W_End.txt",ios::out) ;

	ofstream Q_End("Q_End.txt",ios::out) ;

	ofstream Array("Array.txt",ios::out) ;

	ofstream Out_x11("x1.txt",ios::out) ;

	ofstream Out_x22("x2.txt",ios::out) ;

	ofstream Result1("result1.txt",ios::out) ;

	ofstream Out_x111("x11.txt",ios::out) ;

	ofstream Out_x222("x22.txt",ios::out) ;

	ofstream Result2("result2.txt",ios::out) ;


	for( i=0 ; i<InMax ; i++)
	{
		for(j=0 ; j<InMax ; j++)
		{
			Out_x11<<Input_Net[0][i]<<',';
			Out_x22<<Input_Net[1][j]<<"," ;
			Result1<<3*NetWorkOut(i,j)<<"," ;
			Out_x1<<Input_Net[0][i]<<"," ;

			Array<<Input_Net[0][i]<<"        " ;

			Out_x2<<Input_Net[1][j]<<"," ;

			Array<<Input_Net[1][j]<<"        " ;

			Out_Net<<3*NetWorkOut(i,j)<<"," ;

			Array<<Y(Input_Net[0][i],Input_Net[1][j])<<"        " ;

			Out_Exp<<Y(Input_Net[0][i],Input_Net[1][j])<<"," ;

			Array<<3*NetWorkOut(i,j)<<"        " ;

			Array<<'\n' ;
		}
		Out_x1<<'\n' ;
		Out_x2<<'\n' ;
		Out_x11<<'\n';
		Out_x22<<'\n';
		Result1<<'\n' ;

	}
	for(j=0 ; j<InMax ; j++)
	{
		for(i=0 ; i<InMax ; i++)
		{
			Out_x111<<Input_Net[0][i]<<',';
			Out_x222<<Input_Net[1][j]<<"," ;
			Result2<<3*NetWorkOut(i,j)<<"," ;
		}
		Out_x111<<'\n';
		Out_x222<<'\n' ;
		Result2<<'\n' ;
	}


	//把经过训练后的权系数和阀值保存到文件里
	for(i=1 ; i<Layer_Max ; i++)
	{
		for(int j=0 ; j<Layer_number[i] ; j++)
		{
			for(int k=0 ; k<Layer_number[i-1]+1 ; k++)
			{

				W_End<<W[i][j][k]<<"," ;//保存权系数
			}
		}
	}//end for

}//end
