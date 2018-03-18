# CHANGE VALUES HERE TO GET THE RESULTS:  
# normal_rand_generator(experiment count, sample count, mean of gaussian, sd of gaussian)
dataset_C1 <- normal_rand_generator(500,100,0,1)
dataset_C2 <- normal_rand_generator(500,100,1,2)

P_x_C1 = function(x)
{
  x = (1/(2*pi)^(1/2))*exp(-0.5*((x)^(2)))
}
P_x_C2 = function(x)
{
  x = (1/(4*pi)^(1/2))*exp(-0.25*(x-1)^(2))
}


plot(c(dataset_C2), type = "p", col = rgb(0, 0, 1, 0.5))
par(new = TRUE)
plot(c(dataset_C1), type = "p", col = rgb(1, 0, 0, 1))
title("Distribution of C1 and C2 datasets")


hist(dataset_C1,col=rgb(1, 0, 0, 0.5), probability = T, main = "", breaks = 50, xlab = "")
hist(dataset_C2,col=rgb(0, 0, 1, 1),probability = T, main = "", breaks = 50, xlab = "", add =T)
lines(density(dataset_C1), col="red")
lines(density(dataset_C2), col="blue")
title("Histogram amd distribution of P(x|C1) (red) and P(x|C2) (blue)")

plot(P_x_C1, col= "red" , from=-10, to=10, xlab="x", ylab="y")
plot(P_x_C2, col= "blue" , from=-10, to=10, xlab="x", ylab="y", add = T)
title("plot of P(x|C1) (red) and P(x|C2) (blue)")

                                 

normal_rand_generator <- function(rounds,instances,avg,sdv)
{
  n <- 1
  while(n<=rounds)
  {
    samples <- rnorm(instances, avg, sdv)
    
    #make a vector of all samples for all rounds
    if(n==1)
      samples_v <- samples
    else
      samples_v<- c(samples_v,samples)
    
    #calc the mean of instances / samples (not needed, provided just in case)
    if(n==1)
      mean_samples_v <- mean(samples)
    else
      mean_samples_v<- c(mean_samples_v,mean(samples))
    
    n <- n+1
  }
  
  #sample_mat <- matrix(samples_v,nrow = rounds, byrow = TRUE)
  #plot(c(samples_v))
  #hist(mean_samples_v,col="gray",main="Histogram of experiment means", xlab="Mean of each experiment")
  
  
  
  return(samples_v)  
}
