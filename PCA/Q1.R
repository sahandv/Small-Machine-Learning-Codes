library("caret", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")
#Include this package and run this as a whole
######################################################################################
TrainDataRaw<-read.table("optdigits.tra", sep ="," )
TestDataRaw<-read.table("optdigits.tes", sep = ",")
######################################################################################
# Discriminant function
Discriminant_G <- function(Data_X, MeanData, Denominator, PriorData)
{
  g = matrix(0,nrow = dim(Data_X)[1],ncol = 10)
  for(i in 1:10)
  {
    ClassMean_tmp <- as.vector(MeanData[i,])
    g[,i] = -(1/2) * colSums(((as.matrix(t(Data_X)) - ClassMean_tmp)^2)/Denominator) + log(PriorData[i])
  }
  return(g)
}

TrainDataVariance<-NULL
ClassCount=10
######################################################################################
#Extracting labels
TrainLabels=matrix(TrainDataRaw[,dim(TrainDataRaw)[2]], nrow=dim(TrainDataRaw)[1], ncol=1)
TestLabels = TestDataRaw[,dim(TrainDataRaw)[2]]
######################################################################################
#Extracting actual data
TestData = TestDataRaw[,(1:dim(TrainDataRaw)[2]-1)]
TrainData = TrainDataRaw[,(1:dim(TrainDataRaw)[2]-1)]
######################################################################################
#Finding variances accross data
for (i in 1:dim(TrainData)[2])
{ 
  TrainDataVariance[i]=var(TrainData[,i])
}
######################################################################################
#Removing variances of 0 from both train and test data
ZeroVariances=which(TrainDataVariance==0)
df_train<-data.frame(TrainData)
df_Test<-data.frame(TestData)

j<-0;
for (i in ZeroVariances)
{
  df_train<-df_train[,-(i-j)]
  df_Test<-df_Test[,-(i-j)]
  j <- j+1
}

######################################################################################
#Initiating data for further scopes
FinalTrainData<-NULL
DataIndex<-NULL
MeanMatrix=matrix(0,nrow=ClassCount,ncol =dim(df_train)[2])
CovarianceMatrix <- array(0, dim=c(ClassCount,dim(df_train)[2],dim(df_train)[2]))
CommonCovariance=matrix(0,nrow = dim(df_train)[2],ncol = dim(df_train)[2])
DataIndex_df<-NULL
ClassMatrix_TMP<-NULL
DataIndex_df<-data.frame(DataIndex)
ClassMatrix_TMP<-df_train[DataIndex_df[i,1],]

Prior_PCi<-matrix(0,nrow = ClassCount ,ncol = 1)
######################################################################################
#Priror probability
c_freq<-c(376, 389, 380,389,387,376,377,387,380,382) 
#Prior_PCi<-matrix(0,nrow = 10,ncol = 1)
Num_sample=dim(TrainDataRaw)[1]
for(i in 0:ClassCount-1)
{
  Prior_PCi[i+1,1]<-c_freq[i+1]/Num_sample
}
######################################################################################
#Iterating over classes to process:
# * mean matrix
# * covariance matrix
for(j in 0:(ClassCount-1))
{
  DataIndex_df<-data.frame(DataIndex)
  ClassMatrix_TMP<-df_train[DataIndex_df[1,1],] #Empty df for current class
  
  DataIndex<-which(TrainLabels==j)
  DataIndex_df<-data.frame(DataIndex)
  SampleCount=nrow(DataIndex_df)
  
  #Fetching class data from whole dataframe
  ClassMatrix_TMP <- df_train[DataIndex_df[,1],] 

  #Mean of class
  MeanMatrix[j+1,]<-apply(ClassMatrix_TMP,2,mean)
  
  #Xi-Mean for each class
  SubtractedMatrix<-NULL
  SubtractedMatrix = matrix(0, nrow = SampleCount, ncol = dim(df_train)[2])
  
  #print(dim(SubtractedMatrix))
  print("please wait")
  
  for(i in 1:SampleCount)
  {
    SubtractedMatrix[i,]=as.matrix(ClassMatrix_TMP[i,]-MeanMatrix[j+1,])
  }
  
  #Cov matrix
  CovarianceMatrix[j+1,,]<-as.matrix(t(SubtractedMatrix))%*%as.matrix(SubtractedMatrix)/SampleCount

  ClassMatrix_TMP<-NULL
  DataIndex<-NULL
}

######################################################################################
#Common covariance
CommonCovariance=matrix(0,nrow = dim(df_train)[2],ncol = dim(df_train)[2])
for(i in 0:(ClassCount-1))
{
  CommonCovariance=(CommonCovariance)+(Prior_PCi[i+1,1]*CovarianceMatrix[i+1,,])
}

CommonVariance = diag(CommonCovariance)
DiagonalMean = mean(CommonVariance)

######################################################################################
#Discriminant data
Train_Discr_Naive <- Discriminant_G(df_train, MeanMatrix, CommonVariance, Prior_PCi)
Test_Discr_Naive <- Discriminant_G(df_Test, MeanMatrix, CommonVariance, Prior_PCi)
Train_Discr_Euclidean <- Discriminant_G(df_train, MeanMatrix, DiagonalMean, Prior_PCi)
Test_Discr_Euclidean <- Discriminant_G(df_Test, MeanMatrix, DiagonalMean, Prior_PCi)


######################################################################################
#Error processing
######################################################################################
#Train Naive
TrainClassSelection_Naive<-as.matrix(max.col(Train_Discr_Naive, 'first'))
TrainClassSelection_Naive<-TrainClassSelection_Naive-1

j<-0
for(i in 1:dim(df_train)[1])
{
  if(TrainClassSelection_Naive[i,1]==TrainLabels[i,1])
  {
    j=j+1  
  }
}

TrainAccuracy_Naive=j/dim(df_train)[1]
TrainError_Naive=1-TrainAccuracy_Naive

ConfusMatrixTrainNaive <- confusionMatrix(data=TrainClassSelection_Naive, reference=as.matrix(TrainLabels))

######################################################################################
#Test Naive
TestClassSelection_Naive<-as.matrix(max.col(Test_Discr_Naive, 'first'))
TestClassSelection_Naive<-TestClassSelection_Naive-1

j<-0
for(i in 1:dim(df_Test)[1])
{
  if(TestClassSelection_Naive[i,1]==as.matrix(TestLabels)[i,1])
  {
    j=j+1  
  }
}

TestAccuracy_Naive=j/dim(df_Test)[1]
TestError_Naive=1-TestAccuracy_Naive

ConfusMatrixTestNaive <- confusionMatrix(data=TestClassSelection_Naive, reference=as.matrix(TestLabels))

######################################################################################
#Train Euclidean

TrainClassSelection_Euclidean<-as.matrix(max.col(Train_Discr_Euclidean, 'first'))
TrainClassSelection_Euclidean<-TrainClassSelection_Euclidean-1

j<-0
for(i in 1:dim(df_train)[1])
{
  if(TrainClassSelection_Euclidean[i,1]==TrainLabels[i,1])
  {
    j=j+1  
  }
}

TrainAccuracy_Euclidean=j/dim(df_train)[1]
TrainError_Euclidean=1-TrainAccuracy_Euclidean

ConfusMatrixTrainEuclidean <- confusionMatrix(data=TrainClassSelection_Euclidean, reference=as.matrix(TrainLabels))

######################################################################################
#Test Euclidean
TestClassSelection_Euclidean<-as.matrix(max.col(Test_Discr_Euclidean, 'first'))
TestClassSelection_Euclidean<-TestClassSelection_Euclidean-1

j<-0
for(i in 1:dim(df_Test)[1])
{
  if(TestClassSelection_Euclidean[i,1]==as.matrix(TestLabels)[i,1])
  {
    j=j+1  
  }
}

TestAccuracy_Euclidean=j/dim(df_Test)[1]
TestError_Euclidean=1-TestAccuracy_Euclidean

ConfusMatrixTestEuclidean <- confusionMatrix(data=TestClassSelection_Euclidean, reference=as.matrix(TestLabels))


