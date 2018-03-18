G1_x = function(x)
{
  x= log(1/((2*pi)^(1/2)),base = exp(1))-((x^2)/2)+log(1/2,base = exp(1))
}
G2_x = function(x)
{
  x= log(1/((4*pi)^(1/2)),base = exp(1))-(((x-1)^2)/4)+log(0.8,base = exp(1))
}

plot(G1_x, from=-2000, to=2000, xlab="x", ylab="y", col="red")
plot(G2_x, from=-2000, to=2000, xlab="x", ylab="y", col="blue", add = T)
title("G1 (red) and G2 (blue) functions")

dataset_C1 <- rnorm(10000, 0, 1)
dataset_C2 <- rnorm(10000, 1, 2)

C1_DataFrame <- data.frame(class = "C1" , l=dataset_C1)
C2_DataFrame <- data.frame(class = "C2" , l=dataset_C2)

C1_dens <- density(dataset_C1, from = min(c(dataset_C1,dataset_C2)), to = max(c(dataset_C1,dataset_C2)))
C2_dens <- density(dataset_C2, from = min(c(dataset_C1,dataset_C2)), to = max(c(dataset_C1,dataset_C2)))

plot(C1_dens,main = "")
par(new = TRUE)
plot(C2_dens,main = "c1 and c2 densities")

density_bind <- cbind(C1_dens$y,C2_dens$y)
P1 <- density_bind[,1]/rowSums(density_bind)
P2 <- density_bind[,2]/rowSums(density_bind)

plot(C1_dens$x,P1,type = "l", ylim = c(0,1))
lines(C1_dens$x,P2,col = 2)

plot(C1_dens$x,sapply(C1_dens$x,function(x) log(1/((2*pi)^(1/2)),base = exp(1))-((x^2)/2)+log(1/2,base = exp(1))), type = "l" , col = "red")
lines(C1_dens$x,sapply(C1_dens$x,function(x) log(1/((4*pi)^(1/2)),base = exp(1))-(((x-1)^2)/4)+log(0.8,base = exp(1))) , col = "blue")
grid(29, col = "lightgray", lty = "dotted", lwd = 1)
title("G1 (red) and G2 (blue) using samples of classes with P(C2) = 0.8")
#OR
plot(C2_dens$x,sapply(C2_dens$x,function(x) log(1/((2*pi)^(1/2)),base = exp(1))-((x^2)/2)+log(1/2,base = exp(1))), type = "l" , col = "red")
lines(C2_dens$x,sapply(C2_dens$x,function(x) log(1/((4*pi)^(1/2)),base = exp(1))-(((x-1)^2)/4)+log(0.8,base = exp(1))) , col = "blue")
title("G1 (red) and G2 (blue) using samples of classes with P(C2) = 0.8")

