//---------------------------------------------------------------------------------------//
// BP算法例子：用一个五层的神经网络去逼近函数            
// f(x1,x2)=pow(x1-1,4)+2*pow(x2,2)                   
// Author：Jiaping Xiao                                            
// Date: 2014 运行于 VS2015                     
//--------------------------------------------------------------------------------------//

#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<fstream>

#include"BP00.h"


void main(void)
{   
	BP B;//生成一个BP类对象B
	B.Train();//开始训练
	B.BP_Print();//把结果打印出来
	B.After_Train_Out();//把结果保存到文件

}//end
