import os
import torch
from torch import optim
from torch.autograd import Variable
from policy import Policy
from value import Value
import arguements
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape  # modify scene
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import numpy as np
import time


''' An indirect way to control the speed along path

    One pr.step() cost ~0.07 seconds,
    We assume that every step cost 0.07 second, delta is the step length along a certain axis.
    Then the speed = delta / 0.07, 
    delta = 0.07 * speed.

    Fixed parabola: z = y^2 - 0.4y + 1.04, y(0) = 0.2, z(0) = 1.0
  
    dummy network: (y-y0) --> (v_y, w)

'''


def init_pos(pr):
	agent = UR10()
	ee_init_pos = np.array([1.0, 0.2, 0.95])
	# ee_init_pos = np.array([1.0, 0.2, 1.0])
	#
	# Get a path to the target (rotate so z points down) ---
	path = agent.get_path(
		position=ee_init_pos, euler=[-2.17, 3.1415, 0])

	# generate path given position and euler angles ---
	done = False
	while not done:
		done = path.step()  # how does step works?
		pr.step()

	target = Shape.create(type=PrimitiveShape.CUBOID,  # the cuboid
						  # size=[0.05, 0.05, 0.6],  # when target height=0.6
						  size=[0.05, 0.05, 0.4],
						  mass=0.1,
						  smooth=False,
						  color=[1.0, 0.1, 0.1],
						  position=[1.0, 0.2, 0.95],
						  static=False, respondable=True)
	# target.set_position(np.array([1.0, 0.2, 1.1]))  # initial position of the target for height 0.6

	# change the friction coefficient of the target ---
	target.set_bullet_friction(0.5)
	# fr_tar = target.get_bullet_friction()
	# print('friction of the target:', fr_tar)
	# exit()

	time.sleep(0.05)

	# print('ee_init_pos:', agent.get_tip().get_position())
	# print('tar_init_pos:', target.get_position())

	return agent, target


def move(dy, dz, omega, ee_pos, ee_orient, pr, agent):
	# print('omega:', omega)
	# ee's x,y,z of the next step --
	ee_pos[1] += dy
	ee_pos[2] += dz
	# ee's orientation of the next step --
	ee_orient[0] += omega

	# ee_pos[1] = np.clip(ee_pos[1], 0.2, 0.65)  # position limit
	# ee_pos[2] = np.clip(ee_pos[2], 1.0, 1.405)
	ee_pos[1] = np.clip(ee_pos[1], 0.2, 0.7)  # position limit
	ee_pos[2] = np.clip(ee_pos[2], 0.95, 1.45)
	ee_orient[0] = np.clip(ee_orient[0], 0.97, 2.6)  # orientation limit
	# ee_orient[0] = np.clip(ee_orient[0], 0.8, 2.6)  # orientation limit

	'''normally it won't get to the desired point, but don't know why'''

	new_joint_angles = agent.solve_ik(ee_pos, euler=ee_orient)  # get the joint angles of the robot by doing IK --

	# agent.set_joint_target_velocities([1, 1, 1, 1, 1, 1])   # not sure how to use this --?

	agent.set_joint_target_positions(new_joint_angles)  # set the joint angles as the result of IK above

	pr.step()  # Step the physics simulation

	# get the actual  position and orientation of the ee after pr.step()
	ee = agent.get_tip()
	ee_pos_ = ee.get_position()
	ee_orient_ = ee.get_orientation()

	# print("dy = ", dy, "actual_dy = ", ee_pos_[1] - ee_pos[1])
	# print("dz = ", dz, "actual_dz = ", ee_pos_[2] - ee_pos[2])

	return ee_pos_, ee_orient_, new_joint_angles


def is_stable(ee, target):
	ee_pos_ = ee.get_position()
	tar_pos = target.get_position()
	pos_shift = np.linalg.norm(ee_pos_ - tar_pos)

	ee_orient_ = ee.get_orientation()
	tar_orient = target.get_orientation()
	orient_shift = abs(ee_orient_[0] - 0.94 - tar_orient[0])

	if pos_shift < 0.2 and orient_shift < 0.3:
	# if pos_shift < 0.3 and orient_shift < 0.2:
		return True
	else:
		return False


def is_liftup(target):
	h = target.get_position()[2]
	# print('h:', h)
	tar_ori_x = target.get_orientation()[0]
	# print('tar_ori_x:', tar_ori_x)
	b = 0.2 * np.cos(tar_ori_x)
	# print('b:', b)
	c = 0.025 * np.sin(tar_ori_x)
	# print('c:', c)
	# print('h - b - c:', h - b - c)
	height = h - b - c - 0.77
	if height > 0 and tar_ori_x > 0:   # if the lowest point of the target is heigher than the table
		print('-- lift up --')
		return 1, height
	else:
		return 0, 0


def get_reward(stable, liftup_t, v, target, ee, args):

	# fl: 0 fall, 1 stay, 2 success
	'''
    ee-->end effector
    start ee pos: [1, 0.2, 1.0]   0.95
    ee goal pos: [1, 0.7 1.5]   1.45
    '''
	# liftup, height = liftup_t
	tar_pos = target.get_position()
	# tar_ori = target.get_orientation()
	ee_ori = ee.get_orientation()
	ee_pos = ee.get_position()
	# print("ee_ori:", ee_ori)

	# x_init_theta = 9.39550817e-01
	# delta_theta = ee_ori[0] - 9.412049e-01
	# print("delta_theta:",delta_theta)

	r1 = 0
	r2 = 0
	r3 = 0
	r4 = 0
	r5 = 0

	if stable == 0:
		r1 = -20

	elif stable == 1:

		if ee_pos[1] >= 0.2 and ee_pos[2] >= 0.95:
			# print('ee_pos:', ee_pos)
			b = np.array([ee_pos[1]-0.2, ee_pos[2]-1.0])
			# print('b:', b)
			a = np.array([1, 1])
			cos_theta = (a[0] * b[0] + a[1] * b[1]) / (np.linalg.norm(a) * np.linalg.norm(b))
			# print('cos_theta:', cos_theta)
			projection = np.linalg.norm(b) * cos_theta
			# print('projection:', projection)

			r2 = projection

		# threshold = 0.97 + 20 * np.pi / 180
		# if ee_ori[0] > threshold:  # expand the goal to attract the ee ---------
		if np.linalg.norm((ee_pos[1:] - np.array([0.7, 1.45]))) < 0.6:  # expand the goal to attract the ee ---------
		# print('--- in trap ---')
		# 	r4 = ee_ori[0] - threshold
			r4 = ee_ori[0] - 0.97
	else:
		r5 = 1000

	lam1 = 1
	lam2 = 100
	lam3 = 1
	lam4 = 100
	#ggg    ddddda
	# print('stable:', stable)
	# print('r2_:', lam2*(r2))
	# print('r1_:', lam1*r1)

	tot_r = lam1*r1 + (lam2*(r2)) + lam3*r3 + lam4*r4 + r5

	# print('reward:', r)
	return tot_r, lam2*(r2), lam4*r4


def sample(policy, pr):
	args = arguements.achieve_args()
	batchsz = args.sample_point_num

	# initialize the simulation environment ---
	pr.start()
	agent = UR10()
	starting_joint_positions = [-1.554722547531128, -0.008860016241669655, 2.54549503326416, 0.5965893864631653, 1.5870474576950073, 4.1144609451293945]  # these angles correspond to [1.0, 0.2, 0.95], -2.17
	# starting_joint_positions = [-1.5027912855148315, -0.1732456535100937, 2.4574127197265625, 0.8374652862548828, 1.6857144832611084,
	#  													3.9670567512512207]    # these angles correspond to [1.0, 0.2, 1.0], -2.17
	agent.set_joint_positions(starting_joint_positions)
	# agent.set_control_loop_enabled(False)
	agent.set_motor_locked_at_zero_velocity(True)

	goal_ = Shape.create(type=PrimitiveShape.SPHERE,  # the cuboid
						  # size=[0.05, 0.05, 0.6],  # increased the height by 0.2
						  size=[0.02, 0.02, 0.02],
						  mass=0.0,
						  smooth=False,
						  color=[1.0, 0.1, 0.1],
						  static=True, respondable=True)
	goal_.set_position(np.array([1.0, 0.7, 1.45]))
	# -------------------------------------------------

	success_num = 0
	traj_num = 0
	sample_num = 0

	data = {'state': [], 'action': [], 'reward': [], 'done': [], }

	avg_reward = []
	translation_reward_avg = []
	rotation_reward_avg = []

	while sample_num < batchsz:

		agent, target = init_pos(pr)  # init agent and target

		ee = agent.get_tip()
		ee_pos = ee.get_position()
		ee_orient = ee.get_orientation()
		# print('initial_ee_pos:', ee_pos)
		# print('initial_ee_orient:', ee_orient)

		traj_reward = 0
		translation_reward = 0
		rotation_reward = 0
		traj_num += 1
		for i in range(100):  # 100 steps max

			y = ee_pos[1]
			data['state'].append(y)
			z = ee_pos[2]
			# print('y:', y)

			action = policy.select_action(Variable(torch.Tensor([y]).unsqueeze(0)))[0]   # add noise to actuib
			# action = policy(Variable(torch.Tensor([y]).unsqueeze(0)))[0]               # no noise

			action = np.squeeze(action.detach().numpy())
			v = action[0]
			omega = action[1]
			# print('v:', v)
			# print('omega:', omega)

			data['action'].append(np.squeeze(np.asarray([v, omega])))
			# print('action:', )
			dy = 0.07 * v  # the step length along y axis
			# print('dy:', dy)
			y_ = y + dy  # estimated next y pos
			z_ = 2 * y_ ** 2 - 0.8 * y_ + 1.03    # [1, 0.2, 0.95]
			# z_ = 2 * y_ ** 2 - 0.8 * y_ + 1.08  # original [1, 0.2, 1]
			dz = z_ - z
			'''
            ee-->end effector
            start ee pos: [1, 0.2, 1.0]
            ee goal pos: [1, 0.7 1.5]
            '''

			# print('dz:', dz)
			# print('omega:', omega)
			# print('ee_orient:', ee_orient)

			ee_pos, ee_orient, curr_joint_angles = move(dy, dz, omega, ee_pos, ee_orient, pr, agent)	# move the ee

			sample_num += 1

			# ---------------- three indicators ----------------
			stable = is_stable(ee, target)  # if the target is stable
			liftup = is_liftup(target)  # if the target is lifted up
			# reach_goal = np.linalg.norm(ee_pos - np.array([1, 0.7, 1.5])) < 0.05  # if the target reaches the goal


			# if reach_goal is True:  # if reaches the goal
			# 	print('reach_goal:', reach_goal)
			# check each step after ee_orient > 2.6, if stable, success, break, if not
			if ee_orient[0] > 2.6:  # 2.6 is largest angle of th ee
				for _ in range(5):
					agent.set_joint_target_positions(curr_joint_angles)  # wait for 5 loops to see if it's really stable --?
					# pr.step()  # Step the physics simulation

				if stable is True:
					print('success!!!!!!')
					success_num += 1
					time.sleep(1)  # for observation
					tot_r, r2, r4 = get_reward(2, liftup, v, target, ee, args)    # success
					traj_reward += tot_r
					translation_reward += r2
					rotation_reward += r4
					data['reward'].append(tot_r)
					data['done'].append(0)
					break
				else:
					tot_r, r2, r4 = get_reward(0, liftup, v, target, ee, args)  # fall
					traj_reward += tot_r
					translation_reward += r2
					rotation_reward += r4
					data['reward'].append(tot_r)
					data['done'].append(0)
					break

			else:
				# check each step before ee_orient > 2.2, if stable, continue, if not break
				if stable is True:
					tot_r, r2, r4 = get_reward(1, liftup, v, target, ee, args)  # going on
					traj_reward += tot_r
					translation_reward += r2
					rotation_reward += r4
					data['reward'].append(tot_r)
					data['done'].append(1)  # continue--

				else:
					tot_r, r2, r4 = get_reward(0, liftup, v, target, ee, args)  # fall
					traj_reward += tot_r
					translation_reward += r2
					rotation_reward += r4
					data['reward'].append(tot_r)
					data['done'].append(0)
					break

		target.remove()

		print('--------------- traj length:', i, 'traj reward:', traj_reward, '----------------')

		avg_reward.append(traj_reward)
		translation_reward_avg.append(translation_reward)
		rotation_reward_avg.append(rotation_reward)


	pr.stop()  # Stop the simulation

	print('success_rate:', success_num / traj_num)
	print('avg_reward:', np.mean(avg_reward))
	# print('data:', data)

	return data, np.mean(avg_reward), np.mean(translation_reward_avg), np.mean(rotation_reward_avg)


class PPO:

	def __init__(self):
		"""

		:param env_cls: env class or function, not instance, as we need to create several instance in class.
		:param thread_num:
		"""

		self.args = arguements.achieve_args()
		self.gamma = self.args.gamma
		self.lr = self.args.lr
		self.epsilon = self.args.epsilon
		self.tau = self.args.tau

		# construct policy and value network
		self.policy = Policy(1, 2)
		self.value = Value(1)

		self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
		self.value_optim = optim.Adam(self.value.parameters(), lr=self.lr)

	def est_adv(self, r, v, mask):
		"""
		we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
		:param r: reward, Tensor, [b]
		:param v: estimated value, Tensor, [b]
		:param mask: indicates ending for 0 otherwise 1, Tensor, [b]
		:return: A(s, a), V-target(s), both Tensor
		"""
		batchsz = v.size(0)

		# v_target is worked out by Bellman equation.
		v_target = torch.Tensor(batchsz)
		delta = torch.Tensor(batchsz)
		A_sa = torch.Tensor(batchsz)

		prev_v_target = 0
		prev_v = 0
		prev_A_sa = 0
		for t in reversed(range(batchsz)):
			# mask here indicates a end of trajectory
			# this value will be treated as the target value of value network.
			# mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
			# formula: V(s_t) = r_t + gamma * V(s_t+1)
			v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

			# formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
			delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

			# formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
			A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

			# update previous
			prev_v_target = v_target[t]
			prev_v = v[t]
			prev_A_sa = A_sa[t]

		# normalize A_sa
		A_sa = (A_sa - A_sa.mean()) / A_sa.std()

		return A_sa, v_target

	def update(self, pr):
		"""
		firstly sample batchsz items and then perform optimize algorithms.
		:param batchsz:
		:return:
		"""
		# 1. sample data asynchronously

		# batch = self.sample_original(self.args.sample_point_num)
		batch, avg_reward, translation_reward_avg, rotation_reward_avg = sample(self.policy, pr)
		self.avg_reward = avg_reward
		self.translation_reward_avg = translation_reward_avg
		self.rotation_reward_avg = rotation_reward_avg

		s = torch.from_numpy(np.stack(batch['state'])).view(-1, 1)
		a = torch.from_numpy(np.array(batch['action']))
		r = torch.from_numpy(np.array(batch['reward']))
		mask = torch.from_numpy(np.array(batch['done']))
		batchsz = s.size(0)

		# print('s:', s)
		# print(s.size())
		# print('a:', a)
		# print(a.size())
		# print('r:', r)
		# print(r.size())
		# print('mask:', mask)
		# print(mask.size())
		# print('---- batchsz:-----', batchsz)

		# exit()

		# 2. get estimated V(s) and PI_old(s, a) ---
		# v: [b, 1] => [b]
		v = self.value(Variable(s)).data.squeeze()
		log_pi_old_sa = self.policy.get_log_prob(Variable(s), Variable(a)).data

		# 3. estimate advantage and v_target according to GAE and Bellman Equation ---
		A_sa, v_target = self.est_adv(r, v, mask)

		# 4. backprop ---
		v_target = Variable(v_target)
		A_sa = Variable(A_sa)
		s = Variable(s)
		a = Variable(a)
		log_pi_old_sa = Variable(log_pi_old_sa)

		for _ in range(self.args.epoch_num):

			# 4.1 shuffle current batch ---
			perm = torch.randperm(batchsz)
			# shuffle the variable for mutliple optimize
			v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
			                                                               log_pi_old_sa[perm]

			# 4.2 get mini-batch for optimizing ---
			optim_batchsz = self.args.optim_batchsz
			optim_chunk_num = int(np.ceil(batchsz / optim_batchsz))
			# chunk the optim_batch for total batch
			v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
			                                                               torch.chunk(A_sa_shuf, optim_chunk_num), \
			                                                               torch.chunk(s_shuf, optim_chunk_num), \
			                                                               torch.chunk(a_shuf, optim_chunk_num), \
			                                                               torch.chunk(log_pi_old_sa_shuf,
			                                                                           optim_chunk_num)
			# 4.3 iterate all mini-batch to optimize ---
			for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
			                                                         log_pi_old_sa_shuf):
				# print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
				# 1. update value network ---
				v_b = self.value(s_b)
				loss = torch.pow(v_b - v_target_b, 2).mean()
				self.value_optim.zero_grad()
				loss.backward()
				self.value_optim.step()

				# 2. update policy network by clipping ---
				# [b, 1]
				log_pi_sa = self.policy.get_log_prob(s_b, a_b)
				# ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
				# [b, 1] => [b]
				ratio = torch.exp(log_pi_sa - log_pi_old_sa_b).squeeze(1)
				surrogate1 = ratio * A_sa_b
				surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
				surrogate = - torch.min(surrogate1, surrogate2).mean()

				# backprop ---
				self.policy_optim.zero_grad()
				surrogate.backward(retain_graph=True)
				# gradient clipping, for stability
				torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
				self.policy_optim.step()
		return self.avg_reward, self.translation_reward_avg, self.rotation_reward_avg, self.policy

	def save(self, i, filename='ppo'):

		torch.save(self.value.state_dict(), filename + str(i) + '.val.mdl')
		torch.save(self.policy.state_dict(), filename + str(i) + '.pol.mdl')

		print('saved network to mdl')

	def load(self, filename='ppo'):

		# value_mdl = 'reward_tuning08_157.val.mdl'
		# policy_mdl = 'reward_tuning08_157.pol.mdl'
		value_mdl = 'base_02_010_160.val.mdl'
		policy_mdl = 'base_02_010_160.pol.mdl'

		if os.path.exists(value_mdl):
			self.value.load_state_dict(torch.load(value_mdl))
			print('loaded checkpoint from file:', value_mdl)
		if os.path.exists(policy_mdl):
			self.policy.load_state_dict(torch.load(policy_mdl))

			print('loaded checkpoint from file:', policy_mdl)
		return self.value, self.policy

