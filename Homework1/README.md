# 作业1：数据探索性分析与预处理
### 1. 运行环境及依赖：
python 3.5.2  
pymysql、scipy、numpy、pylab、matplotlib、fancyimpute

### 2. 项目结构
BIT2018-DataMiningHomework/Homework1/  
&emsp;--data  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# 数据（属性名文件）目录  
&emsp;&emsp;--dataset*  
&emsp;--results  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# 结果文件目录  
&emsp;&emsp;--dataset*  
&emsp;&emsp;&emsp;--figure  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# 图片结果目录  
&emsp;&emsp;&emsp;&emsp;--box  
&emsp;&emsp;&emsp;&emsp;--q-q  
&emsp;&emsp;&emsp;&emsp;--histogram  
&emsp;&emsp;&emsp;--标称属性  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# 标称属性结果目录  
&emsp;&emsp;&emsp;--数值属性  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# 数值属性结果目录  
&emsp;&emsp;--imputed_figures &emsp;&emsp;&emsp;&emsp;&emsp;# 缺失值填补结果目录  
&emsp;&emsp;&emsp;--strategy*  
&emsp;&emsp;&emsp;&emsp;--box  
&emsp;&emsp;&emsp;&emsp;--q-q  
&emsp;&emsp;&emsp;&emsp;--histogram  
&emsp;--src  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# 源码目录  
&emsp;&emsp;\_\_init\_\_.py  
&emsp;&emsp;config.py  
&emsp;&emsp;num_processor.py  
&emsp;&emsp;data.py  
### 3. 运行方法

```
python main.py
```

