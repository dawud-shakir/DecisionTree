% test chi squared

p = [2,4,3]
n = [3,0,2]

p_prime = 9*(p+n)/14 % expected error p
n_prime = 5*(p+n)/14 % expected error n

s = ((p-p_prime).^2)./p_prime + ((n-n_prime).^2)./n_prime
chi_sqr = sum(s) % chi_sqr with v-1 degree of freedom

