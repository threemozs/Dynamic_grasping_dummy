from ppo import PPO
import torch
import matplotlib.pyplot as plt
import arguements
import pickle
from os.path import dirname, join, abspath
from pyrep import PyRep


def main():
	torch.set_default_tensor_type('torch.DoubleTensor')
	args = arguements.achieve_args()

	# launch the simulation environment ---
	SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_003.ttt')
	# SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_002.ttt')
	pr = PyRep()
	pr.launch(SCENE_FILE, headless=False)  # lunch the ttt file
	pr.start()

	# instance of PPO ---
	ppo = PPO()

	# load trained models  ---
	# ignore01, ignore02 = ppo.load()

	tot_reward_avgs = []
	translation_reward_avgs = []
	rotation_reward_avgs = []

	saved_tot_reward_avgs = []
	saved_translation_reward_avgs = []
	saved_rotation_reward_avgs = []

	for i in range(1000):
		tot_reward_avg, translation_reward_avg, rotation_reward_avg, POLICY = ppo.update(pr)
		tot_reward_avgs.append(tot_reward_avg)
		translation_reward_avgs.append(translation_reward_avg)
		rotation_reward_avgs.append(rotation_reward_avg)
		print('tot_reward_avgs:', tot_reward_avgs)

		saved_tot_reward_avgs.append(tot_reward_avg)
		saved_translation_reward_avgs.append(translation_reward_avg)
		saved_rotation_reward_avgs.append(rotation_reward_avg)

		idx = i + 0  # MUST CHANGE THIS WHEN RESUME TRAINING !!!!!!!!!
		print("------------------------------------------- Iter ", idx, " -----------------------------------------")

		# save model each 10 iterations, add the rewards in the 10 iterations to the .txt files ---
		if idx % 10 == 0 and idx != 0:

			print('--- saving models ---')
			ppo.save(idx, filename=args.model_name)

			tot_rewards_name = args.model_name + 'rewards.txt'
			tra_rewards_name = args.model_name + 'tra_rewards.txt'
			rot_rewards_name = args.model_name + 'tor_rewards.txt'
			if idx == 10:
				print('--- creating new reward.txt file ---')  #
				with open(tot_rewards_name, "wb") as fp:  # Pickling
					pickle.dump(saved_tot_reward_avgs, fp)
				print('rewards in the newly created reward.txt:', saved_tot_reward_avgs)

				with open(tra_rewards_name, "wb") as fp:  # Pickling
					pickle.dump(saved_translation_reward_avgs, fp)
				print('rewards in the newly created tra_reward.txt:', saved_translation_reward_avgs)

				with open(rot_rewards_name, "wb") as fp:  # Pickling
					pickle.dump(saved_rotation_reward_avgs, fp)
				print('rewards in the newly created rot_reward.txt:', saved_rotation_reward_avgs)

			else:

				print('--- updating reward.txt file ---')
				with open(tot_rewards_name, "rb") as fp:  # Unpickling
					prev_r = pickle.load(fp)
				tot_r = prev_r + saved_tot_reward_avgs
				with open(tot_rewards_name, "wb") as fp:  # Pickling
					pickle.dump(tot_r, fp)
				# print('rewards:', tot_r)
				saved_tot_reward_avgs = []

				print('--- updating tra_reward.txt file ---')
				with open(tra_rewards_name, "rb") as fp:  # Unpickling
					prev_r = pickle.load(fp)
				tra_r = prev_r + saved_translation_reward_avgs
				with open(tra_rewards_name, "wb") as fp:  # Pickling
					pickle.dump(tra_r, fp)
				# print('rewards:', tot_r)
				saved_translation_reward_avgs = []

				print('--- updating rot_reward.txt file ---')
				with open(rot_rewards_name, "rb") as fp:  # Unpickling
					prev_r = pickle.load(fp)
				rot_r = prev_r + saved_rotation_reward_avgs
				with open(rot_rewards_name, "wb") as fp:  # Pickling
					pickle.dump(rot_r, fp)
				# print('rewards:', tot_r)
				saved_rotation_reward_avgs = []

		# plot the rewards each 5 iterations ---
		if idx % 5 == 0:

			if idx >= 10:
				with open(tot_rewards_name, "rb") as fp:  # Unpickling
					prev_r = pickle.load(fp)
				tot_r = prev_r + tot_reward_avgs
				with open(tra_rewards_name, "rb") as fp:  # Unpickling
					prev_r = pickle.load(fp)
				tra_r = prev_r + translation_reward_avgs
				with open(rot_rewards_name, "rb") as fp:  # Unpickling
					prev_r = pickle.load(fp)
				rot_r = prev_r + rotation_reward_avgs

				tot_reward_avgs = []
				translation_reward_avgs = []
				rotation_reward_avgs = []
			else:
				pass
				tot_r = tot_reward_avgs
				tra_r = translation_reward_avgs
				rot_r = rotation_reward_avgs

			iter = list(range(len(tot_r)))
			plt.plot(iter, tot_r)
			plt.plot(iter, tra_r)
			plt.plot(iter, rot_r)
			plt.legend(['total_r', 'translation_r', 'rotation_r'], loc='upper left')
			plt.savefig(args.model_name + 'plot.png')
			plt.show()

	pr.shutdown()  # shut down the simulator


if __name__ == '__main__':
	# print('make sure to execute: [export OMP_NUM_THREADS=1] already.')
	main()
