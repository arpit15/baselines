import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

if __name__ == "__main__":
	dirname = argv[1]

	print("checking the dir %s"%dirname)
	try:
		data = pd.read_csv(dirname)
		# plt.plot(data["total/epochs"], data["eval/return_history"], label='eval')
		plt.plot(data["nupdates"], data["eprewmean"], label='train')
		plt.legend()
		plt.xlabel('Epochs --->')
		plt.ylabel('Episode Reward ---->')
		plt.show()

	except Exception as e:
		print(e)

