from affapy.aa import Affine

# Init
x = Affine([-1, 2.5])
y = Affine([0, 1])

# Get the interval
res = 3 * x + y
res = res - x
res1 = res * x
print(res1.interval)