#显示程序运行状态
print("Connecting Wind API")
print()
# 导入 Wind Python 模块
from WindPy import *
#导入 dataframe 所需模块
import numpy as np
import pandas as pd
#导入 math 模块
import math
#导入最小二乘法相关模块
import numpy as np
from scipy.optimize import leastsq
#导入 ordered dict 模块（用于创建有顺序的dict）
from collections import OrderedDict
#导入 Excel 输出模块
from pandas import ExcelWriter
#以下是 最小二乘法 所需function
#定义想要拟合的函数
def fun(p, x):
     k, b = p  #从参数p获得拟合的参数
     return k*x+b
#定义误差函数
def err(p, x, y):
     i=fun(p,x)
     ii=i-y
     return ii

# 启动 Wind 
w.start();
#检查 Python 程序是否已经与 Wind 连接
if w.isconnected():
     print('\n Wind is connected successfully')
print()

###############################################################################
#以下是创建 raw data 的 dataframe 的过程
###############################################################################
#得到今天的日期
today=datetime.today().strftime('%Y%m%d')

#从 Wind 导入数据
r_data=w.edb("M1004696,M1001099,M1001100,M1001101,M1001102,M1001103,M1001104,M1001105,M1001106,M1001107,M1001108,M1001109,M1004697,M1001110,M1001111", "2006-03-01", today,"Fill=Previous")
#r_data 是一个万得的 class 文件。具体的数据构成参见万得python接口文档。本程序所需数据主要是其提供的的时间（Times）和利率（Data）数据


#设置我们需要的所有的月份数据（都是以月为单位）
timeslot=[1,3,6,9]+list(range(12,121,12))+[180]

r_time=r_data.Times #提取 万得提供的 raw data 中的 date 数据
r_rate=r_data.Data  #提取 万得提供的 raw data 中的 具体利率 数据

#将 data 数据转换成 str 格式，方便组合dataframe 【格式：20180616】
#因为 万得的 class 中关于时间的数据类型不是 string，为了方便构建dataframe，我们将它转换成 string，因此本程序中所由日期作为竖向 index 的 dataframe 都是 string 类型。
r_timestr=[]#将所有转化成 string 的时间放在一个 list 中，方便构建 dataframe
for i in range(len(r_time)):
     r_timestr.append(r_time[i].strftime("20%y%m%d"))

#为 dataframe 创建 date 数据的 column 做准备
time= ('Time',r_timestr)

#创建由不同 time slot 利率组成的 columns 数据的 nested list 集合 
r_rate_time=[]
for i in range(len(r_rate)):
     r_rate_time+=[(timeslot[i],r_rate[i])]

#将 date 数据和 利率 数据组合在一起，其中 date 数据在最后一个 column
r_rate_time.append(time)
    
#创建最终的 dataframe，但是目前会报 warning，正在排查优化中
r_df=pd.DataFrame.from_items(r_rate_time)

#将表示 date 的 column 作为最左边的 index 
r_df.index=r_df['Time'].copy()

#显示程序运行状态
print()
print('Raw data imported')
###############################################################################
# raw data 的 dataframe 创建完成
###############################################################################


###############################################################################
#以下是通过最小二乘法估计 Wind 中未提供的即期利率数据的方法【本部分需要大改优化以提升准确率】
###############################################################################

#补全数据第一部分：通过已知数据，找出对应的斜率

r_df_row_list=r_df.values.tolist() #将 row 的数据提取出来，作为求斜率的已知数据

les2eq=OrderedDict()#创建有顺序的 dict，用以保存求出的 k，b 值。

for i in range(len(r_df_row_list)): #将每一个 row 单独拿出来求，用最小二乘法求 k，b 值
     temp_date=r_df_row_list[i][len(timeslot)]
     r_df_row_list[i].pop()
     x_temp = np.array(timeslot)
     y_temp = np.array(r_df_row_list[i])
     p0 = [1,1]
     para_temp = leastsq(err, p0, args=(x_temp,y_temp))
     les2eq[temp_date]=(para_temp[0][0],para_temp[0][1])
     
# les2eq 是集成了时间和包含所对应 最小二乘法 公式k、b的 dict，第一个是 k，第二个是b
     
##########################################################
# 补全数据第二部分：通过 k，b值，找出对应的斜率

#以下是各个月份计算远期利率时所需的远期月份
tp3=[3,6,9,12,15,21,27,39,63,87,123]

tp6=[6,9,12,15,18,24,30,42,66,90,126]

tp9=[9,12,15,18,21,27,33,45,69,93,129]

tp12=[12,15,18,21,24,30,36,48,72,96,132]

tp24=[24,27,30,33,36,42,48,60,84,108,144]

tp=tp3+tp6+tp9+tp12+tp24 #加和，求出所有月份
tp=set(tp) # 设置成 set 自动删掉重复月份
tp=list(tp) 
tp.sort() #再转换成 list 并且排序


##########################################################

month_required=tp #所需要计算的月份

col_name=r_df.columns #提取现有的月份

for_pop=[] #找出所需要的和现有的重复额月份
for i in month_required:
     if i in col_name:
          for_pop.append(i)
          
temp_list = [x for x in month_required if x not in for_pop]# 去掉重复的
month_required=temp_list
r_df = r_df.drop('Time', 1)#删掉Time column
col_name=r_df.columns
for i in month_required:#计算出 missing data 并且插入到 raw data 里面
     temp_rate=[]
     for j in les2eq:
          temp_rate.append(les2eq[j][0]*i+les2eq[j][1])
     idx=0
     for k in range(len(col_name)):
          if i> int(col_name[k]) and i< int(col_name[k+1]):
               idx=k+1
               break
          elif i> int(col_name[len(col_name)-1]):
               idx=k+1
               break  
          else:
               k+=1
     r_df.insert(loc=idx, column=i, value=temp_rate)
     
r_df=r_df.sort_index(axis=1)

#显示程序进程
print()
print('Misssing data calculated')            
##########################################################
#使用最小二乘法模拟数据创建完成
##########################################################

##########################################################
#开始构建不同时间区间的 foward rate
##########################################################
tp5_list=[3,6,9,12,24]#找出5个月
tp5_list_list=[tp3,tp6,tp9,tp12,tp24]
forward_df_list=[]
for k in range(len(tp5_list)): 
     time_start=tp5_list[k]     
     new_df=OrderedDict()     
     new_df_month=tp5_list_list[k]    
     for i in new_df_month:
          new_df[i]=[]
     
          if i==time_start:
               new_df[i]=r_df[i].copy()
          else :
               for j in range(len(r_df[time_start])): #根据远期利率公式计算远期利率
                    new_df[i].append((math.pow((math.pow((1+((r_df[i][j]/100)/12)),i)/math.pow((1+((r_df[time_start][j]/100)/12)),time_start)),(1/(i/12-time_start/12)))-1)*100)
                    
     forward_df=pd.DataFrame.from_dict(new_df)     
     new_col_name=[]
     for i in range(len(forward_df.columns)):
          mon=forward_df.columns[i]     
          new_col_name.append((time_start,mon))     
     forward_df.columns=new_col_name 
     forward_df_list.append(forward_df)
#当前程序进程
print()
print('Forward rate calculated') 
##########################################################
#不同时间区间的 foward rate 构建完成
##########################################################

##########################################################
#开始构建不同时间区间的  break even rate
##########################################################

be_df_list=[]

for i in range(len(tp5_list)):

     be_time_start=tp5_list[i]
     forward_df=forward_df_list[i]

     
     be_df=forward_df.copy() # 创建 break even 专属的dataframe
     
     be_df=be_df.sub(r_df[be_time_start],axis=0) # 用 forward 的数据减去 即期利率 得到 break even
     
     be_df = be_df.drop((be_time_start,be_time_start), 1) #删掉无意义的（be_time_start，be_time_start） column
     
     be_df_list.append(be_df)

print()
print('Break even rate calculated')

##########################################################
#break even rate 计算完成
##########################################################

##########################################################
# 接下来计算持有期收益率
##########################################################
time_expected=int(input("请输入日期(格式: 20180518)(输出文件需保持关闭状态)(输入日期必须是工作日，并在今天之前): "))

#time_expected=20180621 #注意，输入的的日期是结算上一个的，不包括这个

bp_tp3_list=list(range(150,9,-10))
bp_tp6_list=list(range(130,-11,-10))
bp_tp9_list=list(range(150,9,-10))
bp_tp12_list=list(range(140,-1,-10))
bp_tp24_list=list(range(180,39,-10))
bp_tp_list=[bp_tp3_list,bp_tp6_list,bp_tp9_list,bp_tp12_list,bp_tp24_list]

tp_dict=dict()

for i in range(len(tp5_list)):
     temp_tp_be_value=[]
     for j in be_df_list[i].columns:
          temp_be=be_df_list[i].at[str(time_expected), j]
          temp_be_value=[]
          for k in bp_tp_list[i]:
               temp_be_value.append(temp_be*100-k)
          temp_tp_be_value.append(temp_be_value)
     tp_dict[tp5_list[i]]=temp_tp_be_value


##########################################################
# 接下来编辑输出格式
##########################################################

final_tp3 = tp_dict[3]
final_tp6 = tp_dict[6]
final_tp9 = tp_dict[9]
final_tp12 = tp_dict[12]
final_tp24 = tp_dict[24]
print()
print('BP data calculated')

'''
label_tp3=['Time-CN','Time-NUM']+bp_tp3_list
label_tp6=['Time-CN','Time-NUM']+bp_tp6_list
label_tp9=['Time-CN','Time-NUM']+bp_tp9_list
label_tp12=['Time-CN','Time-NUM']+bp_tp12_list
label_tp24=['Time-CN','Time-NUM']+bp_tp24_list
'''

chinese_tp=['3个月','6个月','9个月','1年','1年半','2年','3年','5年','7年','10年']
num_tp=[3,6,9,12,18,24,36,60,84,120]
# tp5_list 是5个月份
num_tp2=dict()
for i in tp5_list:
     temp_num_tp=[]
     for j in num_tp:
          temp_num_tp.append((i, i+j))
     num_tp2[i]=temp_num_tp


header_3=[('Time_CN',chinese_tp),('Time_Num',num_tp2[3])]
header_6=[('Time_CN',chinese_tp),('Time_Num',num_tp2[6])]
header_9=[('Time_CN',chinese_tp),('Time_Num',num_tp2[9])]
header_12=[('Time_CN',chinese_tp),('Time_Num',num_tp2[12])]
header_24=[('Time_CN',chinese_tp),('Time_Num',num_tp2[24])]

print()
print('Formating')
print()
df_hd3=pd.DataFrame.from_items(header_3)
df_hd6=pd.DataFrame.from_items(header_6)
df_hd9=pd.DataFrame.from_items(header_9)
df_hd12=pd.DataFrame.from_items(header_12)
df_hd24=pd.DataFrame.from_items(header_24)



df33=pd.DataFrame.from_records(final_tp3, columns=bp_tp3_list)
df66=pd.DataFrame.from_records(final_tp6, columns=bp_tp6_list)
df99=pd.DataFrame.from_records(final_tp9, columns=bp_tp9_list)
df1212=pd.DataFrame.from_records(final_tp12, columns=bp_tp12_list)
df2424=pd.DataFrame.from_records(final_tp24, columns=bp_tp24_list)

df3 = pd.concat([df_hd3, df33], axis=1)
df6 = pd.concat([df_hd6, df66], axis=1)
df9 = pd.concat([df_hd9, df99], axis=1)
df12 = pd.concat([df_hd12, df1212], axis=1)
df24 = pd.concat([df_hd24, df2424], axis=1)


print('3')
print(df3)
print()
print('6')
print(df6)
print()
print('9')
print(df9)
print()
print('12')
print(df12)
print()
print('24')
print(df24)
print()
print()
print('Exporting')

##########################################################
# 接下来导出数据到 Excel
##########################################################

writer = ExcelWriter('python_export.xlsx')
df3.to_excel(writer,'3')
df6.to_excel(writer,'6')
df9.to_excel(writer,'9')
df12.to_excel(writer,'12')
df24.to_excel(writer,'24')
writer.save()

#描述程序状态
print()
print('Done')