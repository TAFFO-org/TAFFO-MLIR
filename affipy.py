from affapy.aa import Affine

# Init
x = Affine([-1, 2.5])
y = Affine([-1, 2.5])

# Get the interval
z = x * -1
print((x - y).interval)

# Basic operations
x + y
x + 5
x - y
x - 5
-x
