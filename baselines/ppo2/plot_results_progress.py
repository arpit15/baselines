import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
import os.path as osp
if __name__ == "__main__":
	dirname = argv[1]

	print("checking the dir %s"%dirname)
	try:
		data = pd.read_csv(osp.join(dirname,"progress.csv"))
		# plt.plot(data["total/epochs"], data["eval/return_history"], label='eval')
		plt.plot(data["serial_timesteps"], data["eprewmean"], label='train')
		plt.legend()
		plt.xlabel('timesteps --->')
		plt.ylabel('Episode Reward ---->')
		plt.show()

	except Exception as e:
		print(e)

