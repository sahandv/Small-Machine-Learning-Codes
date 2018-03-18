P_C1_x = function(x)
{
 A = exp(-(x^2)/2)
 B = (exp(-((x-1)^2)/4)/2)+A
 x= A/B
}

P_C2_x = function(x)
{
  A = exp(-((x-1)^2)/4)
  B = 2*(exp(-(x^2)/2))+A
  x= A/B
}

plot(P_C1_x, from=-10, to=10, xlab="x", ylab="y", col="red")
plot(P_C2_x, from=-10, to=10, xlab="x", ylab="y", col="blue", add = T)
title("P(C1|x) (red) -  P(C2|x) (blue)")