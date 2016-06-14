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


//���캯��,������ʼ��Ȩϵ�������룬���������ѧϰ�ٶ�

BP::BP()
{
	srand(time(NULL));//���֣��Ա�����漴��
	for(int i=1 ; i<Layer_Max ; i++)
	{
		for(int j=0 ; j<Layer_number[i] ; j++)
		{
			for(int k=0 ; k<Layer_number[i-1]+1 ; k++)
			{
				W[i][j][k] = RANDOM;//�����ʼ��Ȩϵ��

			}
			//     Q[i][j] = RANDOM ;//��ʼ������Ԫ�ķ�ֵ
		}
	}
	//�����������һ��
	for(int l=0 ; l<InMax ; l++)
	{
		Input_Net[0][l] = l * 0.05 ;//��0��1�ֳ�20�ȷ�,��ʾx1
		Input_Net[1][l] = 1 - l * 0.05 ;//��ʾx2
	}
	for(int i=0 ; i<InMax ; i++)
	{
		for(int j=0 ; j<InMax ; j++)
		{
			Out_Exp[i][j] = Y(Input_Net[0][i],Input_Net[1][j]) ;//�������
			Out_Exp[i][j] = Out_Exp[i][j]/3.000000;//���������һ��
		}
	}

	Study_Speed=0.5;//��ʼ��ѧϰ�ٶ�

	e=0.0001;//����


}//end
//��������F()
double BP::F(double x)
{
	return(1.0/(1+exp(-x)));
}//end

//Ҫ�ƽ��ĺ���Y()
//���룺����������
//�����һ��������
double BP::Y(double x1,double x2)
{
	double temp;
	temp = pow(x1-1,4) + 2 * pow(x2,2);
	return temp;
}//end
//--------------------------------------------------------
//���ۺ���
double BP::Cost(double Out,double Exp)
{
	return(pow(Out-Exp,2));
}//end

//�����������
//����Ϊ����input������
double BP::NetWorkOut(int x1 , int x2)
{
	int i,j,k;
	double N_node[Layer_Max][Neural_Max];
	//Լ��N_node[i][j]��ʾ�����i��ĵ�j����Ԫ��������
	//��0�����ԪΪ���룬����Ȩϵ���ͷ�ֵ�������ʲô�����ʲô
	N_node[0][0] = Input_Net[0][x1] ;
	Layer_Node[0][0] = Input_Net[0][x1] ;
	N_node[0][1] = Input_Net[1][x2] ;
	Layer_Node[0][1] = Input_Net[1][x2] ;

	for(i=1 ; i<Layer_Max ; i++)//������ĵ�i��
	{
		for(j=0 ; j<Layer_number[i] ; j++)//Layer_number[i]Ϊ��i���
		{                             //��Ԫ����
			N_node[i][j] = 0.0;
			for(k=0 ; k<Layer_number[i-1] ; k++)//Layer_number[i-1]
			{            //��ʾ���i���j����Ԫ���ӵ���һ���
				//��Ԫ����

				//����һ����Ԫ�Ե�i���j����Ԫ������֮��
				N_node[i][j]+=Layer_Node[i-1][k] * W[i][j][k];

			}
			N_node[i][j] = N_node[i][j]-W[i][j][k];//��ȥ��ֵ

			//��Layer_Node[i][j]������i���j����Ԫ�����
			Layer_Node[i][j] = F(N_node[i][j]);
		}
	}
	return Layer_Node[Layer_Max-1][0];//���һ������
}//end

//��������Ԫ��������΢�ֺ���
//����Ϊ����input������
//�������΢�ֲ�������D[][]������
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
//�޸�Ȩϵ���ͷ�ֵ
void BP::Change_W()
{
	int i,j,k;
	for(i=1 ; i<Layer_Max ; i++)
	{
		for(j=0;j<Layer_number[i];j++)
		{
			for(k=0;k<Layer_number[i-1];k++)
			{
				//�޸�Ȩϵ��
				W[i][j][k]=W[i][j][k]-Study_Speed*
					D[i][j]*Layer_Node[i-1][k];

			}
			W[i][j][k]=W[i][j][k]+Study_Speed*D[i][j];//�޸ķ�ֵ
		}
	}
}//end
//ѵ������
void BP::Train()
{
	int i,j;
	int ok=0;
	double Out;
	long int count=0;
	double err;
	ofstream Out_count("Out_count.txt",ios::out) ;
	//�����е�5��Ȩϵ���ı仯���浽�ļ���
	ofstream outWFile1("W[2][0][0].txt",ios::out) ;
	ofstream outWFile2("W[2][1][1].txt",ios::out) ;
	ofstream outWFile3("W[1][0][0].txt",ios::out) ;
	ofstream outWFile4("W[1][1][0].txt",ios::out) ;
	ofstream outWFile5("W[3][0][1].txt",ios::out) ;

	while(ok<441)
	{
		count++;
		//20����������
		for(i=0,ok=0 ; i<InMax ; i++)
		{
			for(j=0 ; j<InMax ; j++)
			{
				Out = NetWorkOut(i,j);

				AllLayer_D(i,j);

				err = Cost(Out,Out_Exp[i][j]);//�������

				if(err<e) ok++;  //�Ƿ���������

				else Change_W();//���޸�Ȩϵ���ͷ�ֵ
			}

		}
		if((count%1000)==0)//ÿ1000�Σ�����Ȩϵ��
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

//��ӡȨϵ��
void BP::BP_Print()
{
	//��ӡȨϵ��
	cout<<"ѵ�����Ȩϵ��"<<endl;
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

//�ѽ�����浽�ļ�
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


	//�Ѿ���ѵ�����Ȩϵ���ͷ�ֵ���浽�ļ���
	for(i=1 ; i<Layer_Max ; i++)
	{
		for(int j=0 ; j<Layer_number[i] ; j++)
		{
			for(int k=0 ; k<Layer_number[i-1]+1 ; k++)
			{

				W_End<<W[i][j][k]<<"," ;//����Ȩϵ��
			}
		}
	}//end for

}//end