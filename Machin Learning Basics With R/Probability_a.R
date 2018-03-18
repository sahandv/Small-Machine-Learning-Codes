# CHANGE VALUES HERE TO GET THE RESULTS:  
# normal_rand_generator(experiment count, sample count, mean of gaussian, sd of gaussian)

total_mean <- normal_rand_generator(500,100,0,1) 

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
    
    #calc the mean of instances / samples
    if(n==1)
      mean_samples_v <- mean(samples)
    else
      mean_samples_v<- c(mean_samples_v,mean(samples))
    
    n <- n+1
  }
  
  plot(c(samples_v))
  hist(mean_samples_v,col="gray",main="Histogram of experiment means", xlab="Mean of each experiment")
  total_mean <- mean(samples_v)
  return(total_mean)  
}
