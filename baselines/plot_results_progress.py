import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

if __name__ == "__main__":
	dirname = argv[1]

	print("checking the dir %s"%dirname)
	try:
		data = pd.read_csv(dirname)
		plt.plot(data["eval/return_history"])
		plt.show()

	except e:
		print(e)

