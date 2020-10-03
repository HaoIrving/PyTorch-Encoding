import scipy.io as sio
import os
import tifffile as tiff
import numpy as np
import pickle
import cv2
import random


def div(px, py):
	m, n = px.shape
	Mx = np.zeros_like(px)
	My = np.zeros_like(px)

	Mx[1: m - 1, :] = px[1: m - 1, :] - px[: m - 2, :]
	Mx[0, :] = px[0, :]
	Mx[m - 1, :] = - px[m - 2, :]

	My[:, 1: n - 1] = py[:, 1: n - 1] - py[:, : n - 2]
	My[:, 0] = py[:, 0]
	My[:, n - 1] = - py[:, n - 2]
	return Mx + My


def gradxy(I):
	m, n = I.shape
	Mx = np.zeros_like(I)
	Mx[0: m - 1, :] = - I[0: m - 1, :] + I[1:, :]
	Mx[m - 1, :] = np.zeros(n)

	My = np.zeros_like(I)
	My[:, 0: n - 1] = - I[:, 0: n - 1] + I[:, 1:]
	My[:, n - 1] = np.zeros(m)
	return Mx, My


def denoise(y, f, lambda_=1.3):
	p0x = np.zeros(y.shape)
	p0y = np.zeros(y.shape)
	x = y
	x0bar = x
	x0 = x
	rho = 2
	tol = 1e-5
	xr = []
	for i in range(1, 301):
		Mx, My = gradxy(x0bar)
		p1x = p0x + 1.0 / (2.0 * rho) * Mx
		p1y = p0y + 1.0 / (2.0 * rho) * My
		tmp = np.sqrt(p1x ** 2 + p1y ** 2)
		tmp = np.clip(tmp, 1, None)
		p1x = p1x / tmp
		p1y = p1y / tmp
		# Newton Method
		g = lambda_ * (-1.0 * f * f / (np.spacing(1) + np.exp(x)) / (np.spacing(1) + np.exp(x)) + 1) + 2 * rho * (
				x - x0) - div(p1x, p1y)
		gp = lambda_ * 1.0 * f * f / (np.spacing(1) + np.exp(x)) / (np.spacing(1) + np.exp(x)) + 2 * rho
		x = x - g / gp
		xr.append(np.linalg.norm(x - x0, ord='fro') / np.linalg.norm(x0, ord='fro'))
		if i > 1 and xr[-1] < tol:
			break
		# x1bar
		x1bar = 2 * x - x0
		x0 = x
		x0bar = x1bar
		p0x = p1x
		p0y = p1y
	return x


class ProcessTiff:
	def __init__(self, denoise=False):
		self.root = "/home/sun/contest/data_preprocessed"
		self.base_dirs = ["HH", "HV", "VH", "VV"]
		self.cut_dirs = ["HH_cut", "HV_cut", "VH_cut", "VV_cut"]
		self.name = None
		self.image_dir = None
		self.mask_dir = None
		self.all_zero = []
		self.max_log_pix = None
		self.max_log_denoise_pix = 0
		self.max_log_denoise_normal_pix = 0
		self.have_zero_aft_log = []
		self.sum = {0: 0., 1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0.}
		self.denoise = denoise

	def get_new_split(self):
		p = '/home/sun/contest/sar/r.mat'
		train_val = sio.loadmat(p)
		ret = train_val['r'][0] - 1
		train = list(ret[:400])
		train = list(map(str, train))
		val = list(ret[400:])
		val = list(map(str, val))
		return train, val

	def get_label(self):
		root_path = self.root
		label = sio.loadmat(root_path + "/label.mat")
		label = label["label"]  # 512,512,500   H, W
		return label.transpose(2, 0, 1)  # 500,512,512

	def get_img_paths(self, index):
		index = str(index + 1)
		img_paths = []
		for i in range(4):
			base_dir = self.base_dirs[i]
			img_path = os.path.join(self.root, base_dir, index + "_" + base_dir + ".tiff")
			img_paths.append(img_path)
		return img_paths

	def get_cut_img_paths(self, index):
		index = str(index)
		img_paths = []
		for i in range(4):
			cut_dir = self.cut_dirs[i]
			img_path = os.path.join(self.root, cut_dir, index + "_" + cut_dir + ".tiff")
			img_paths.append(img_path)
		return img_paths

	def generate_dir(self, name, lite=None, train_split=None, val_split=None):
		self.name = name
		root_path = self.root
		base_dir = os.path.join('VOCdevkit_sar', name)
		voc_root = os.path.join(root_path, base_dir)
		image_dir = os.path.join(voc_root, 'JPEGImages')
		mask_dir = os.path.join(voc_root, 'SegmentationClass')
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
		for dir_ in [base_dir, voc_root, image_dir, mask_dir, splits_dir]:
			if not os.path.exists(dir_):
				os.makedirs(dir_)
		train_split_f = os.path.join(splits_dir, "train" + '.txt')
		val_split_f = os.path.join(splits_dir, "val" + '.txt')

		if not lite:
			if not train_split and not val_split:
				train_split = [str(i) for i in range(400)]
				val_split = [str(i) for i in range(400, 500)]
			with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
				fp.write('\n'.join(train_split) + '\n')
			with open(os.path.join(splits_dir, val_split_f), 'w') as fp:
				fp.write("\n".join(val_split) + '\n')
			return train_split_f, val_split_f
		if lite:
			train_split = [str(0), str(1)]
			with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
				fp.write('\n'.join(train_split) + '\n')
			val_split = [str(400), str(401)]
			with open(os.path.join(splits_dir, val_split_f), 'w') as fp:
				fp.write("\n".join(val_split) + '\n')
			return train_split_f, val_split_f

	def get_global_max(self, train_split_f, val_split_f, th=None):
		"""
		get global max after log().
		:param train_split_f:
		:param val_split_f:
		:param th:
		:return:
		"""
		max_log_pix = 0
		count = 0
		for split_f in [train_split_f, val_split_f]:
			with open(os.path.join(split_f), "r") as f:
				file_names = [x.strip() for x in f.readlines()]
			for x in file_names:
				img_paths = self.get_img_paths(int(x))
				im_datas1, im_datas2, im_datas3, im_datas4 = self.cat_4(img_paths, int(x), th)
				max_log_pix = max(max_log_pix, im_datas1.max(), im_datas2.max(), im_datas3.max(), im_datas4.max())
				count += 1
				print(count)
		return max_log_pix

	def cat_4(self, img_paths, x, th=2):
		im_datas1 = self.tiff_np_log(img_paths[0], x, th)
		im_datas2 = self.tiff_np_log(img_paths[1], x, th)
		im_datas3 = self.tiff_np_log(img_paths[2], x, th)
		im_datas4 = self.tiff_np_log(img_paths[3], x, th)
		return im_datas1, im_datas2, im_datas3, im_datas4

	def tiff_np_log(self, img_path, x, th=2):
		im_datas_org = cv2.imread(img_path, -1)
		if im_datas_org is None:
			print('{} is None'.format(x))
		im_datas_org = np.clip(im_datas_org, th, None)
		im_datas = np.log(im_datas_org)
		max_pix = im_datas.max()
		if max_pix == 0:
			self.all_zero.append(x)
		# TODO: 11.090339660644531250 may change.
		im_datas = im_datas / 11.090339660644531250

		if self.denoise:
			im_datas = denoise(im_datas, im_datas_org)

		return im_datas

	def combination_3(self, img_paths, x):
		"""
		Deprecated, not work on new data split.
		在svm, nn上效果最好的特征组合
		:param image, 4, 512, 512, 已经过log和归一化处理
		:return:
		"""
		HH, HV, VH, VV = self.cat_4(img_paths, x, th=2)

		tmp = np.sqrt(HH * HH + VV * VV)
		if tmp.min() == 0:
			self.have_zero_aft_log.append(x)

		channel_0 = HV / tmp
		channel_0 = channel_0[:, :, np.newaxis]
		channel_1 = VH / tmp
		channel_1 = channel_1[:, :, np.newaxis]
		channel_2 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
		channel_2 = channel_2[:, :, np.newaxis]
		ret = np.concatenate((channel_0, channel_1, channel_2), axis=2)
		return ret

	def combination_1(self, img_paths, x):
		"""
		在svm, nn上效果最好的特征组合
		:param image, 4, 512, 512, 已经过log和归一化处理
		:return:
		"""
		HH, HV, VH, VV = self.cat_4(img_paths, x, th=2)
		if self.denoise:
			self.max_log_denoise_pix = max(self.max_log_denoise_pix, HH.max(), HV.max(), VH.max(), VV.max())
			HH = HH / 10.553297276390468439899450459052
			HV = HV / 10.553297276390468439899450459052
			VH = VH / 10.553297276390468439899450459052
			VV = VV / 10.553297276390468439899450459052
			self.max_log_denoise_normal_pix = max(self.max_log_denoise_normal_pix, HH.max(), HV.max(), VH.max(), VV.max())

		tmp = np.sqrt(HH * HH + VV * VV)
		if tmp.min() == 0:
			self.have_zero_aft_log.append(x)

		channel_0 = VH / HH
		channel_0 = channel_0[:, :, np.newaxis]
		channel_1 = HV / tmp
		channel_1 = channel_1[:, :, np.newaxis]
		channel_2 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
		channel_2 = channel_2[:, :, np.newaxis]
		ret = np.concatenate((channel_0, channel_1, channel_2), axis=2)
		return ret

	def combination_2(self, img_paths, x):
		"""
		在svm, nn上效果最好的特征组合
		:param image, 4, 512, 512, 已经过log和归一化处理
		:return:
		"""
		HH, HV, VH, VV = self.cat_4(img_paths, x, th=2)
		if self.denoise:
			self.max_log_denoise_pix = max(self.max_log_denoise_pix, HH.max(), HV.max(), VH.max(), VV.max())
			HH = HH / 10.553297276390468439899450459052
			HV = HV / 10.553297276390468439899450459052
			VH = VH / 10.553297276390468439899450459052
			VV = VV / 10.553297276390468439899450459052
			self.max_log_denoise_normal_pix = max(self.max_log_denoise_normal_pix, HH.max(), HV.max(), VH.max(), VV.max())

		tmp1 = np.abs(HV + VH)
		tmp2 = HH + VV

		channel_0 = np.sqrt(tmp1 * tmp2)
		channel_0 = channel_0[:, :, np.newaxis]
		channel_1 = np.sqrt(HV * VH)
		channel_1 = channel_1[:, :, np.newaxis]
		channel_2 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
		channel_2 = channel_2[:, :, np.newaxis]
		ret = np.concatenate((channel_0, channel_1, channel_2), axis=2)
		return ret

	def combination_4(self, img_paths, x):
		"""
		在svm, nn上效果最好的特征组合
		:param img_paths:
		:param image, 4, 512, 512, 已经过log和归一化处理
		:return:
		"""
		HH, HV, VH, VV = self.cat_4(img_paths, x, th=2)

		tmp = np.sqrt(HH * HH + VV * VV)
		if tmp.min() == 0:
			self.have_zero_aft_log.append(x)

		channel_0 = VH / HH
		channel_0 = channel_0[:, :, np.newaxis]
		channel_1 = HV / tmp
		channel_1 = channel_1[:, :, np.newaxis]
		channel_2 = VH / tmp
		channel_2 = channel_2[:, :, np.newaxis]
		channel_3 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
		channel_3 = channel_3[:, :, np.newaxis]
		ret = np.concatenate((channel_0, channel_1, channel_2, channel_3), axis=2)
		return ret

	def cut(self, data, fil_heigh, fil_width, x, y):
		return data[x:x + fil_heigh, y: y + fil_width]

	def cal_sum(self, tem_label):
		key = np.unique(tem_label)

		for k in key:
			mask = (tem_label == k)
			y_new = tem_label[mask]
			v = y_new.size
			self.sum[k] += v

	def get_cut_coordinate(self, data, name_list, fil):
		x_y = []
		x = y = 0
		for index in range(len(name_list)):
			x_y.append([])
			lable = data[int(name_list[index]), :, :]
			while x < (data.shape[1] - fil.shape[0] + 1):
				while y < (data.shape[2] - fil.shape[1] + 1):
					tem_label = self.cut(lable, fil.shape[0], fil.shape[1], x, y)
					# 统计各类的数量
					self.cal_sum(tem_label)
					# if ((0 in tem_label and 1 in tem_label and 5 in tem_label and 6 in tem_label) or (
					if ((5 in tem_label) or (
					random.randint(1, 2) != 1)):
						x_y[index].append([x, y])
					# y += random.randint(4, 4)
					y += 10
				# x += random.randint(4, 4)
				x += 10
				y = 0
			x = 0
		return x_y

	def data_augmentation(self, labels, train_split, val_split):
		fil = np.zeros((320, 320))
		x_y_train = self.get_cut_coordinate(labels, train_split, fil)
		x_y_val = self.get_cut_coordinate(labels, val_split, fil)
		sum = self.sum[0] + self.sum[1] + self.sum[2] + self.sum[3] + self.sum[4] + self.sum[5] + self.sum[6]
		for i in range(len(self.sum)):
			print("The ratio of class %d is (%.30f)" % (i, self.sum[i] / sum))

		dir = self.cut_dirs
		for i in dir:
			tem_dir = self.root + "/" + i
			if not os.path.exists(tem_dir):
				os.makedirs(tem_dir)
		num = 0
		train = []
		val = []
		labs = []
		for i in range(len(x_y_train)):  #
			tem_path = self.get_img_paths(int(train_split[i]))
			for ii in range(len(x_y_train[i])):
				for iii in range(len(tem_path)):
					im_data = cv2.imread(tem_path[iii], -1)
					cut_data = self.cut(im_data, fil.shape[0], fil.shape[1], x_y_train[i][ii][0], x_y_train[i][ii][1])
					tiff.imsave(self.root + "/" + dir[iii] + "/" + str(num) + "_" + dir[iii] + ".tiff", cut_data)

				labs.append(self.cut(labels[int(train_split[i])], fil.shape[0], fil.shape[1], x_y_train[i][ii][0],
									 x_y_train[i][ii][1]))
				train.append(str(num))
				num += 1
		# num=0
		for i in range(len(x_y_val)):  #
			tem_path = self.get_img_paths(int(val_split[i]))
			for ii in range(len(x_y_val[i])):
				for iii in range(len(tem_path)):
					im_data = cv2.imread(tem_path[iii], -1)
					cut_data = self.cut(im_data, fil.shape[0], fil.shape[1], x_y_val[i][ii][0], x_y_val[i][ii][1])
					tiff.imsave(self.root + "/" + dir[iii] + "/" + str(num) + "_" + dir[iii] + ".tiff", cut_data)

				labs.append(self.cut(labels[int(val_split[i])], fil.shape[0], fil.shape[1], x_y_val[i][ii][0],
									 x_y_val[i][ii][1]))
				val.append(str(num))
				num += 1
		return train, val, labs

	def transform_all_to_psc_format(self, mode=None, th=None, lite=None, comb="c1"):
		"""
		将原始tiff数据转为 VOC文件目录格式
		:param mode: 对tiff文件操作的类型："log_normal", "log_normal_c3"
		:return: .pkl文件
		"""
		train_split, val_split = self.get_new_split()
		labels = self.get_label()
		labels[labels == 7] = 0
		# 数据增强
		train_split, val_split, labs = self.data_augmentation(labels, train_split, val_split)

		train_split_f, val_split_f = self.generate_dir(mode, lite, train_split, val_split)

		# max_log_pix = self.get_global_max(train_split_f, val_split_f, th)
		# self.max_log_pix = max_log_pix
		self.all_zero = list(set(self.all_zero))
		# self.max_log_pix = 11.090339660644531250

		for split_f in [train_split_f, val_split_f]:
			with open(os.path.join(split_f), "r") as f:
				file_names = [x.strip() for x in f.readlines()]
			for x in file_names:
				label = labs[int(x)]
				img_paths = self.get_cut_img_paths(int(x))
				if mode == "log_normal_c3_lite":
					im = self.combination_3(img_paths, int(x))
				if comb == "c1":
					im = self.combination_1(img_paths, int(x))
				if comb == "c2":
					im = self.combination_2(img_paths, int(x))  # 512,512,3
				image = os.path.join(self.image_dir, x + ".pkl")
				mask = os.path.join(self.mask_dir, x + ".pkl")
				with open(image, "wb") as img_f:
					pickle.dump(im, img_f)
				with open(mask, "wb") as mask_f:
					pickle.dump(label, mask_f)
				print(x)

		sum = self.sum[0] + self.sum[1] + self.sum[2] + self.sum[3] + self.sum[4] + self.sum[5] + self.sum[6]
		for i in range(len(self.sum)):
			print("The ratio of class %d is (%.30f)" % (i, self.sum[i] / sum))
		print(self.all_zero)
		print("max_log_denoise_pix is (%.30f)" % self.max_log_denoise_pix)
		print("max_log_denoise_normal_pix is (%.30f)" % self.max_log_denoise_normal_pix)
		print(self.have_zero_aft_log)
		print("Finised")
		return


if __name__ == "__main__":
	demo = ProcessTiff(denoise=False)

	# demo.transform_all_to_psc_format(mode="log_normal", th=2)
	demo.transform_all_to_psc_format(mode="log_normal_new_noise_crop_c1", th=2, comb="c1")
	# demo.transform_all_to_psc_format(mode="log_normal_new_c2", th=2)
# demo.transform_all_to_psc_format(mode="log_normal_c3", th=2)
# demo.transform_all_to_psc_format(mode="log_normal_c3_lite", th=2, lite=True)
