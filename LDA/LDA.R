library("MASS", lib.loc="/usr/lib/R/library")
#Include this package and run this as a whole
######################################################################################
TrainDataRaw<-read.table("optdigits.tra", sep ="," )
TestDataRaw<-read.table("optdigits.tes", sep = ",")
######################################################################################
#Initiating data for further scopes
RawFeatureDimention <- dim(TrainDataRaw)[2]
RawTrainDataSize <- dim(TrainDataRaw)[1]
TrainDataVariance<-NULL
######################################################################################
#Extracting labels
TrainLabels=matrix(TrainDataRaw[,RawFeatureDimention], nrow=RawTrainDataSize, ncol=1)
TestLabels = as.matrix(TestDataRaw[,RawFeatureDimention])
######################################################################################
#Extracting actual data
TestData = TestDataRaw[,(1:RawFeatureDimention-1)]
TrainData = TrainDataRaw[,(1:RawFeatureDimention-1)]

df_Train<-data.frame(TrainData)
df_Test<-data.frame(TestData)

if(FALSE)
{
  #Finding variances accross data
  for (i in 1:dim(TrainData)[2])
  { 
    TrainDataVariance[i]=var(TrainData[,i])
  }
  
  #Removing variances of 0 from both train and test data
  ZeroVariances=which(TrainDataVariance==0)
  
  j<-0;
  for (i in ZeroVariances)
  {
    df_Train<-df_Train[,-(i-j)]
    df_Test<-df_Test[,-(i-j)]
    j <- j+1
  }
  j<-NULL
  i<-NULL
}
######################################################################################
#Initiating data for further scopes
InitialFeatureDimention <- dim(df_Train)[2]
CurrentFeatureDimention <- InitialFeatureDimention
ClassCount=10
TrainDataSize <- dim(df_Train)[1]
TestDataSize <- dim(df_Test)[1]
MeanMatrix <- matrix(0,ncol = InitialFeatureDimention,nrow = ClassCount)
ClassMeanMatrix <- matrix(0,nrow = ClassCount, ncol = InitialFeatureDimention)
OverallTrainMeanMatrix <- matrix(0,nrow = ClassCount, ncol = InitialFeatureDimention)
#Subtracted_tmp2 <- matrix(0,nrow = 10, ncol = InitialFeatureDimention)

WithinClassScatterMatrix <-matrix(0,nrow = InitialFeatureDimention, ncol = InitialFeatureDimention)
BetweenClassScatterMatrix <-matrix(0,nrow = InitialFeatureDimention, ncol = InitialFeatureDimention)
ClassData_tmp_1 <- NULL
######################################################################################
#Calculating scatter matrices
i<-0
while(i<ClassCount)
{
  ClassRowIndexes_tmp <- which(TrainLabels==i)

  ClassData_tmp <- df_Train[as.vector(ClassRowIndexes_tmp),]
  ClassSampleSize <- dim(ClassData_tmp)[1]
  MeanMatrix[i+1,]<-colMeans(ClassData_tmp)
  OverallTrainMeanMatrix[i+1,]<-colMeans(df_Train)
  Subtracted_tmp <- t(as.matrix(ClassData_tmp)) - as.vector(MeanMatrix[i+1,])
  Subtracted_tmp <- t(Subtracted_tmp)
  WithinClassScatterMatrix <- WithinClassScatterMatrix + (t(Subtracted_tmp)%*%Subtracted_tmp)
  
  Subtracted_tmp2 <- t(as.vector(MeanMatrix[i+1,])-as.matrix(t(OverallTrainMeanMatrix)))
  
  BetweenClassScatterMatrix <- BetweenClassScatterMatrix + (ClassSampleSize*((t(Subtracted_tmp2)%*%Subtracted_tmp2)))
  Subtracted_tmp<-NULL
  Subtracted_tmp2<-NULL
  i=i+1
}
i<-NULL

######################################################################################
WithinClassScatter_PseudoInverse <- ginv(WithinClassScatterMatrix)
ScatterProduct <- WithinClassScatter_PseudoInverse%*%BetweenClassScatterMatrix
ScatterEigen <- eigen(ScatterProduct)
ScatterEigen_values <- ScatterEigen$values
ScatterEigen_vectors <- ScatterEigen$vectors

EigenValueSortIndex <- order(ScatterEigen_values, decreasing=TRUE)
ScatterEigen_values_sorted <- ScatterEigen_values[EigenValueSortIndex]
ScatterEigen_vectors_sorted <- ScatterEigen_vectors[,EigenValueSortIndex]
 
TwoDimensionTransformMatrix <- ScatterEigen_vectors_sorted[,1:2]
TrainData_Z <- as.matrix(df_Train)%*%TwoDimensionTransformMatrix
TestData_Z <- as.matrix(df_Test)%*%TwoDimensionTransformMatrix

TrainRandIndex = sample(1:TrainDataSize, 200, replace=FALSE)
TestRandIndex = sample(1:TestDataSize, 200, replace=FALSE)

TrainLabels_samples = TrainLabels[TrainRandIndex,1]
TestLabels_samples = TestLabels[TestRandIndex,1]
######################################################################################
#Plot scatter matrix
plot(TrainData_Z[,1],TrainData_Z[,2],xlab = "Z_x", ylab = "Z_y", col="green", asp = 1:1, sub = "Transformed Train Data")
text(TrainData_Z[TrainRandIndex,1], TrainData_Z[TrainRandIndex,2], labels=TrainLabels_samples, cex= 0.7)

plot(TestData_Z[,1],TestData_Z[,2],xlab = "Z_x", ylab = "Z_y", col="red", asp = 1:1, sub = "Transformed Test Data")
text(TrainData_Z[TestRandIndex,1], TrainData_Z[TestRandIndex,2], labels=TestLabels_samples, cex= 0.7)

