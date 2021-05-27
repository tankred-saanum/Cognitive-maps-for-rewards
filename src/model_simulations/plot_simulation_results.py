import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd

model_names = ["", "Temporal", "Euclidean","Compositional","Mean tracker", "Optimized temporal"]
nll = pd.read_csv("model_simulations/nll_matrix.csv").to_numpy()
posterior = pd.read_csv("model_simulations/posterior_matrix.csv").to_numpy()


f, ax = plt.subplots()
plt.title("Negative log likelihoods")
im=ax.imshow(nll, cmap="PuBu")
ax.set_yticklabels(model_names)
ax.set_xticklabels(model_names)
f.colorbar(im)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


for i in range(len(nll)):
    for j in range(len(nll)):
        text = ax.text(j, i, np.around(nll[i, j]),
                       ha="center", va="center", color="k", size=12)
f.tight_layout()
plt.savefig("model_simulations/nll.png")
plt.show()



f, ax = plt.subplots(1,1)
plt.title("Posterior probability")
im = ax.imshow(posterior, cmap="PuBu")
ax.set_xticklabels(model_names)
ax.set_yticklabels(model_names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(nll)):
    for j in range(len(nll)):
        text = ax.text(j, i, np.around(posterior[i, j], 2),
                       ha="center", va="center", color="k", size=12)
f.colorbar(im)
f.tight_layout()
plt.savefig("model_simulations/posterior.png")
plt.show()
