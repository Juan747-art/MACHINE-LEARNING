import matplotlib.pyplot as plt

iterations = [1, 2, 3]
variance = [50, 30, 20]

plt.plot(iterations, variance)
plt.xlabel("Iteration")
plt.ylabel("Variance")
plt.title("Variance Reduction")

plt.savefig("static/images/variance_plot.png")