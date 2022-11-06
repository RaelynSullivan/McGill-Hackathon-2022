reset 
unset multiplot
pi = 3.14159
gaussian(x,mu,sigma) = 1./sqrt(2*pi*sigma**2) * exp(-(x-mu)**2/(2*sigma**2)) 

plot gaussian(x,5,2)-gaussian(x,5,4)
