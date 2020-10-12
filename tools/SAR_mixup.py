import scipy.io as sio
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import torch
from collections import defaultdict
from PIL import Image

_total = 512 * 512
_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
_bin_table = {
	5: [150, 11, 5, 1, 0, 0, 1, 0, 0, 0],
	6: [53, 16, 5, 1, 4, 0, 2, 0, 1, 1],
	1: [286, 32, 18, 8, 5, 3, 3, 7, 6, 6],
	2: [129, 65, 51, 36, 30, 26, 24, 5, 2, 1],
	4: [111, 58, 62, 49, 34, 35, 37, 19, 18, 6],
	3: [70, 68, 72, 57, 48, 41, 40, 33, 26, 12],
}
_bin_index_record = {
	5: [[10, 13, 14, 33, 37, 44, 45, 46, 49, 50, 51, 53, 55, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81,
		 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 96, 97, 101, 104, 105, 106, 107, 114, 115, 118, 119, 129, 130, 132,
		 133, 134, 140, 152, 176, 184, 187, 188, 192, 200, 205, 206, 228, 229, 257, 267, 268, 272, 280, 283, 284, 289,
		 290, 293, 294, 297, 298, 300, 301, 302, 304, 311, 320, 321, 322, 325, 326, 331, 333, 338, 349, 351, 353, 354,
		 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 370, 373, 374, 375, 376, 377, 378, 380, 381, 384,
		 385, 386, 389, 392, 395, 397, 404, 406, 407, 408, 425, 430, 446, 448, 449, 453, 454, 457, 463, 467, 469, 472,
		 480, 483, 484, 486, 488, 489, 490, 492, 496, 499], [0, 12, 54, 88, 108, 111, 144, 396, 445, 455, 485],
		[9, 92, 100, 458, 459], [91], [], [], [8], [], [], []],
	6: [[6, 7, 64, 75, 98, 99, 102, 103, 113, 116, 124, 127, 130, 131, 134, 135, 136, 146, 147, 148, 150, 156, 159, 199,
		 201, 202, 216, 217, 219, 220, 236, 237, 238, 264, 266, 282, 286, 289, 302, 303, 335, 339, 343, 349, 350, 443,
		 453, 456, 457, 464, 467, 477, 478],
		[123, 139, 149, 157, 158, 298, 299, 301, 338, 342, 346, 348, 440, 460, 481, 482], [128, 145, 288, 300, 444],
		[297], [296, 337, 341, 463], [], [340, 347], [], [336], [452]],
	1: [[1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35,
		 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 52, 57, 58, 59, 60, 63, 64, 72, 73, 74, 75, 80, 81, 82, 84, 85, 91, 92,
		 93, 95, 97, 98, 99, 101, 102, 103, 105, 106, 108, 109, 110, 112, 115, 119, 123, 128, 129, 130, 131, 132, 133,
		 136, 139, 140, 141, 142, 143, 152, 154, 155, 156, 157, 161, 162, 164, 165, 166, 168, 169, 170, 172, 173, 174,
		 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 197, 198,
		 199, 215, 216, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 245, 254,
		 256, 264, 265, 266, 269, 270, 273, 274, 275, 276, 277, 278, 279, 281, 282, 283, 284, 285, 286, 287, 288, 289,
		 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
		 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
		 334, 335, 343, 344, 345, 346, 347, 348, 351, 352, 355, 356, 360, 361, 362, 364, 365, 366, 367, 368, 371, 372,
		 376, 377, 378, 379, 382, 383, 384, 386, 387, 388, 389, 390, 391, 393, 403, 406, 407, 419, 420, 421, 425, 426,
		 427, 428, 429, 430, 431, 432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 448, 449, 450, 463,
		 467, 484, 486, 487, 490, 491, 498],
		[4, 11, 12, 15, 43, 118, 134, 135, 189, 248, 251, 353, 354, 369, 373, 374, 380, 381, 385, 392, 411, 415, 418,
		 433, 488, 489, 492, 493, 494, 495, 496, 499],
		[0, 146, 196, 206, 211, 214, 241, 280, 357, 358, 370, 409, 412, 416, 417, 422, 423, 485],
		[114, 127, 158, 205, 220, 244, 253, 272], [116, 120, 150, 200, 207], [210, 250, 255], [126, 147, 204],
		[151, 203, 217, 222, 223, 249, 252], [121, 124, 125, 201, 202, 213], [113, 117, 122, 209, 212, 218, 221]],
	2: [[2, 6, 13, 14, 16, 22, 24, 25, 26, 34, 35, 38, 39, 43, 45, 46, 50, 52, 54, 57, 59, 60, 63, 64, 72, 81, 84, 85,
		 95, 105, 108, 128, 137, 138, 142, 150, 153, 177, 180, 193, 194, 195, 196, 197, 198, 201, 202, 203, 204, 207,
		 209, 213, 217, 221, 223, 225, 226, 235, 236, 237, 239, 243, 247, 264, 268, 271, 273, 274, 275, 278, 279, 287,
		 288, 298, 299, 300, 301, 314, 323, 328, 329, 336, 340, 341, 342, 349, 351, 353, 359, 362, 364, 365, 367, 369,
		 370, 373, 374, 375, 380, 381, 385, 391, 392, 396, 398, 399, 400, 408, 420, 426, 432, 434, 436, 437, 442, 445,
		 446, 447, 448, 452, 453, 457, 458, 459, 477, 481, 486, 487, 491],
		[3, 5, 18, 27, 28, 29, 33, 37, 48, 56, 58, 97, 101, 145, 154, 163, 168, 169, 172, 176, 178, 179, 182, 183, 191,
		 205, 220, 222, 260, 269, 270, 282, 293, 311, 324, 325, 347, 348, 352, 354, 356, 357, 358, 368, 372, 378, 386,
		 388, 394, 395, 397, 410, 413, 414, 424, 428, 429, 430, 431, 435, 454, 455, 461, 462, 490],
		[1, 7, 8, 9, 10, 11, 12, 30, 49, 53, 106, 112, 155, 156, 162, 170, 186, 188, 190, 210, 211, 214, 215, 224, 228,
		 229, 258, 259, 263, 272, 286, 291, 295, 330, 334, 360, 361, 371, 376, 377, 384, 404, 407, 433, 460, 468, 472,
		 473, 476, 484, 498],
		[23, 36, 102, 149, 152, 159, 166, 187, 189, 206, 257, 262, 280, 281, 289, 294, 304, 321, 322, 326, 331, 333,
		 346, 383, 389, 421, 425, 463, 464, 469, 470, 471, 485, 488, 497, 499],
		[15, 17, 19, 21, 47, 103, 109, 116, 167, 171, 285, 290, 320, 332, 343, 355, 366, 382, 405, 406, 409, 411, 439,
		 443, 465, 466, 474, 475, 479, 495],
		[0, 98, 144, 160, 161, 184, 185, 219, 261, 284, 335, 339, 344, 350, 390, 402, 417, 440, 444, 449, 450, 456, 467,
		 480, 483, 492],
		[4, 20, 31, 99, 123, 127, 148, 165, 173, 174, 216, 283, 345, 387, 403, 412, 415, 416, 419, 422, 423, 489, 494,
		 496], [110, 208, 393, 401, 493], [175, 418], [164]],
	4: [[0, 4, 8, 9, 10, 15, 20, 27, 32, 53, 110, 113, 116, 117, 122, 129, 130, 133, 134, 135, 140, 145, 149, 159, 163,
		 167, 174, 200, 201, 202, 208, 209, 213, 216, 218, 220, 222, 240, 241, 244, 245, 248, 249, 250, 252, 255, 258,
		 259, 261, 262, 320, 323, 324, 325, 326, 334, 344, 345, 346, 347, 369, 371, 372, 373, 377, 382, 388, 391, 396,
		 399, 400, 402, 403, 406, 410, 413, 414, 416, 421, 422, 423, 425, 431, 438, 442, 443, 444, 447, 449, 450, 454,
		 458, 459, 462, 464, 465, 466, 468, 469, 471, 472, 473, 474, 476, 477, 478, 479, 480, 483, 488, 499],
		[11, 55, 57, 58, 82, 83, 99, 121, 125, 131, 132, 139, 143, 148, 155, 156, 157, 158, 162, 175, 203, 207, 210,
		 219, 223, 242, 243, 246, 247, 253, 257, 263, 272, 307, 335, 343, 354, 355, 357, 365, 368, 374, 381, 390, 407,
		 409, 412, 415, 424, 439, 440, 445, 446, 453, 457, 460, 461, 475],
		[6, 7, 19, 23, 49, 51, 54, 56, 68, 84, 86, 87, 88, 91, 92, 95, 98, 109, 112, 120, 128, 141, 142, 144, 152, 165,
		 168, 173, 192, 196, 204, 211, 214, 254, 256, 260, 266, 267, 283, 285, 290, 296, 297, 321, 322, 332, 353, 356,
		 358, 367, 370, 376, 383, 384, 386, 387, 389, 411, 417, 420, 430, 448],
		[2, 14, 30, 45, 46, 47, 79, 80, 81, 85, 90, 102, 103, 106, 126, 136, 138, 160, 161, 166, 169, 170, 180, 184,
		 189, 191, 199, 206, 215, 229, 265, 268, 270, 271, 280, 281, 284, 294, 300, 304, 317, 331, 333, 351, 360, 361,
		 364, 366, 380],
		[3, 12, 16, 17, 21, 36, 43, 50, 94, 96, 104, 105, 137, 178, 185, 187, 188, 195, 205, 224, 225, 236, 264, 286,
		 289, 298, 299, 306, 310, 313, 328, 329, 330, 362],
		[1, 13, 18, 22, 26, 29, 44, 48, 52, 60, 61, 62, 75, 89, 93, 101, 114, 171, 176, 179, 190, 198, 228, 232, 269,
		 282, 288, 291, 293, 295, 311, 314, 352, 378, 385],
		[5, 25, 28, 33, 35, 37, 39, 63, 64, 69, 71, 72, 76, 97, 100, 107, 153, 154, 177, 181, 182, 183, 186, 193, 194,
		 197, 227, 231, 273, 274, 277, 278, 279, 301, 316, 327, 359],
		[24, 34, 38, 66, 67, 70, 108, 111, 118, 172, 237, 238, 239, 275, 287, 312, 319, 375, 392],
		[65, 73, 74, 77, 115, 119, 226, 230, 233, 234, 276, 292, 302, 308, 315, 318, 363, 379],
		[59, 78, 235, 303, 305, 309]],
	3: [[4, 8, 12, 16, 17, 21, 24, 25, 26, 44, 45, 46, 73, 78, 99, 103, 111, 116, 121, 122, 144, 160, 164, 172, 184,
		 185, 186, 202, 204, 205, 206, 209, 210, 213, 218, 221, 222, 226, 230, 234, 235, 238, 276, 280, 283, 287, 292,
		 301, 302, 303, 305, 309, 315, 336, 363, 385, 388, 391, 392, 411, 415, 417, 418, 422, 423, 441, 445, 463, 489,
		 493],
		[0, 1, 5, 28, 33, 34, 37, 38, 43, 47, 65, 74, 77, 97, 98, 100, 108, 110, 123, 124, 173, 179, 182, 183, 187, 190,
		 201, 203, 208, 211, 223, 228, 233, 237, 239, 272, 275, 284, 288, 289, 295, 308, 311, 318, 320, 323, 324, 326,
		 334, 347, 355, 372, 375, 379, 382, 386, 387, 390, 393, 409, 429, 436, 440, 444, 448, 485, 494, 496],
		[15, 18, 19, 20, 22, 23, 29, 31, 35, 36, 39, 48, 63, 66, 67, 69, 70, 71, 72, 101, 102, 107, 109, 148, 151, 152,
		 154, 174, 176, 188, 189, 193, 194, 197, 207, 214, 217, 219, 224, 249, 252, 255, 273, 274, 277, 279, 281, 282,
		 285, 286, 290, 291, 293, 294, 304, 319, 321, 322, 325, 332, 333, 335, 340, 360, 366, 374, 384, 401, 403, 432,
		 449, 492],
		[11, 13, 52, 64, 75, 76, 89, 105, 106, 120, 147, 153, 158, 166, 170, 177, 178, 181, 198, 215, 216, 229, 251,
		 269, 278, 296, 297, 298, 299, 300, 327, 330, 331, 339, 343, 344, 345, 352, 356, 359, 361, 368, 373, 376, 378,
		 389, 419, 431, 435, 439, 443, 450, 460, 467, 483, 488, 495],
		[3, 7, 9, 27, 30, 49, 60, 61, 62, 91, 93, 112, 128, 149, 156, 167, 169, 196, 200, 220, 225, 232, 240, 244, 250,
		 261, 264, 270, 314, 346, 350, 354, 358, 362, 364, 370, 380, 383, 402, 406, 407, 430, 456, 474, 475, 479, 480,
		 499],
		[2, 10, 50, 56, 79, 80, 81, 85, 92, 94, 96, 104, 136, 137, 138, 150, 155, 159, 162, 168, 191, 195, 236, 253,
		 257, 328, 329, 337, 341, 351, 353, 357, 369, 405, 421, 425, 433, 438, 465, 466, 469],
		[6, 14, 51, 54, 84, 86, 88, 90, 95, 142, 145, 146, 157, 180, 192, 199, 241, 254, 258, 259, 260, 262, 263, 265,
		 268, 271, 367, 377, 381, 420, 442, 453, 459, 464, 470, 471, 472, 473, 476, 497],
		[53, 58, 68, 82, 87, 135, 139, 141, 247, 248, 256, 266, 267, 342, 348, 365, 395, 396, 404, 413, 424, 446, 454,
		 455, 457, 458, 461, 462, 468, 481, 484, 490, 498],
		[55, 57, 83, 129, 131, 132, 133, 134, 140, 143, 163, 242, 243, 245, 246, 338, 349, 394, 397, 410, 414, 428, 434,
		 437, 447, 482], [130, 398, 399, 400, 408, 426, 427, 477, 478, 486, 487, 491]]
}


def putpalette(img):
	out_img = Image.fromarray(img.squeeze().astype('uint8'))
	sarvocpallete = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255]
	out_img.putpalette(sarvocpallete)
	out_img.show()


def pixel_hist(tem_label):
	hist = np.array([0, 0, 0, 0, 0, 0, 0])
	key = np.unique(tem_label)
	for k in key:
		mask = (tem_label == k)
		y_new = tem_label[mask]
		v = y_new.size
		hist[k] += v
	freq = hist / hist.sum()
	return hist, freq


def mix_target_image(HH, HHs):
	# 将HHs中所有前景映射到target image的HH上,将HH中对应的像素位置置0,再相加
	keep_mask = HHs != 0
	HH[keep_mask] = 0
	HHs += HH
	return HHs


def mix_source_image(HH, HHs, bg_mask):
	# 只保留source类别的像素
	HH[bg_mask] = 0
	# 保持HHs中的前景
	keep_mask = HHs != 0
	HH[keep_mask] = 0
	HHs += HH
	return HHs


def get_new_split():
	p = '/home/sun/contest/sar/r.mat'
	train_val = sio.loadmat(p)
	ret = train_val['r'][0] - 1
	train = list(ret[:400]) + list(range(500, 700))
	train = list(map(str, train))
	train_new_1 = list(range(500, 600))
	train_new_1 = list(map(str, train_new_1))
	train_new_2 = list(range(600, 670))
	train_new_2 = list(map(str, train_new_2))
	train_new_3 = list(range(670, 700))
	train_new_3 = list(map(str, train_new_3))
	val = list(ret[400:])
	val = list(map(str, val))
	return train, val, train_new_1, train_new_2, train_new_3


def select_and_mix(t=5, batch=50, bin=0.1, labels=None):
	bin_index_record = _bin_index_record[t]
	# random select batch labels for cat t in one bin
	i = _bins.index(bin)
	index_record_perbin = np.array(bin_index_record[i])
	batch = min(index_record_perbin.size, batch)
	total = np.random.permutation(range(index_record_perbin.size))
	selected = total[:batch]  # random select batch from one bin
	selected_index_perbin = index_record_perbin[selected]

	# mix all selected small region labels of cat t in one bin to a large mask, union
	new_mask = np.zeros_like(labels[0])
	for j in range(selected_index_perbin.size):
		selected_index = selected_index_perbin[j]
		label = labels[selected_index]
		fg_mask = label == t
		new_mask += fg_mask
	new_mask = new_mask != 0
	ratio = new_mask.sum() / new_mask.size
	return ratio, new_mask, selected_index_perbin


def mixup(new_mask_seq, new_mask_index_seq, source=5, target=3, n=20, target_bin=0.8, labels=None):  # 3,5混合得到的图片数n
	# 选取第3类中的一个label,其中3的占比为bin
	ret_mask_seq = []
	ret_mask_index_seq = []
	bin_index_record = _bin_index_record[target]
	i = _bins.index(target_bin)
	index_record_perbin = bin_index_record[i]
	visited = set()
	# 将target_label和new_mask_seq中的一个合并,将made_category所在的所有位置置为made_category,其他保留
	while True:
		r1 = np.random.randint(0, len(index_record_perbin))
		r2 = np.random.randint(0, len(new_mask_seq))
		if (r1, r2) in visited:
			# print(r1, r2)
			continue
		visited.add((r1, r2))
		target_index = index_record_perbin[r1]
		target_label = copy.deepcopy(labels[target_index])
		new_mask = new_mask_seq[r2]
		index_seq_org = new_mask_index_seq[r2]
		index_seq = np.append(index_seq_org, target_index)
		ret_mask_index_seq.append(index_seq)
		mask1 = np.zeros(new_mask.shape).astype("uint8")
		mask1[new_mask] = source
		target_label[new_mask] = 0
		final_mask = target_label + mask1
		# putpalette(final_mask)
		# hist, freq = pixel_hist(final_mask)
		# print(freq)
		ret_mask_seq.append(final_mask)
		if len(visited) == n:
			break
	return ret_mask_seq, ret_mask_index_seq  # 返回20个


def get_new_mask_from_all_bin(ratio_low=0., ratio_up=1., t=5, k=20, labels=None):
	"""
	new_mask_seq, selected_index_seq: list(np.array)
	"""
	new_mask_seq, selected_index_seq = [], []
	for bin in _bins:  # 暂时不考虑跨区间混跌
		i = _bins.index(bin)
		# 每个区间做20次随机混跌
		for _ in range(20):
			# 混叠层数不超过k=20
			for batch in range(1, min(k, _bin_table[t][i] + 1)):  # 150, 11
				ratio, new_mask, selected_index_perbin = select_and_mix(t=t, batch=batch, bin=bin, labels=labels)
				if ratio_low < ratio <= ratio_up:
					# print("{}: {}".format(bin, ratio))
					new_mask_seq.append(new_mask)
					selected_index_seq.append(selected_index_perbin)
	return new_mask_seq, selected_index_seq


def mixup_2_category(source=5, ratio_low=0.6, ratio_up=0.7, target=3, target_bin=0.8, n=100, labels=None):
	# selected_index_seq 中每个混叠序列的最大长度为k
	new_mask_seq, new_mask_index_seq = get_new_mask_from_all_bin(ratio_low=ratio_low, ratio_up=ratio_up, t=source, k=20,
																 labels=labels)
	print(len(new_mask_seq))
	ret_mask_seq, ret_mask_index_seq = mixup(new_mask_seq, new_mask_index_seq, source=source, target=target, n=n,
											 target_bin=target_bin, labels=labels)
	return ret_mask_seq, ret_mask_index_seq


def get_balance_700(labels=None):
	"""
	5 6 1 < 2 4 3
	:param th:
	   [0,   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1]
	5    [150, 11,   5,   1,   0,   0,   1,   0,   0,   0]
	6    [53,  16,   5,   1,   4,   0,   2,   0,   1,   1]
	1    [286, 32,  18,   8,   5,   3,   3,   7,   6,   6]
	2    [129, 65,  51,  36,  30,  26,  24,   5,   2,   1]
	4    [111, 58,  62,  49,  34,  35,  37,  19,  18,   6]
	3    [70,  68,  72,  57,  48,  41,  40,  33,  26,  12]
	[0.04272119 0.10241207 0.13548531 0.28421111 0.21262745 0.11806116 0.10448171]
	"""
	ret_mask_seq53, ret_mask_index_seq53 = mixup_2_category(source=5, ratio_low=0.7, ratio_up=1, target=3,
															target_bin=0.6, n=100, labels=labels)
	ret_mask_seq63, ret_mask_index_seq63 = mixup_2_category(source=6, ratio_low=0.8, ratio_up=1, target=3,
															target_bin=0.6, n=70, labels=labels)
	ret_mask_seq13, ret_mask_index_seq13 = mixup_2_category(source=1, ratio_low=0.7, ratio_up=1, target=3,
															target_bin=0.7, n=30, labels=labels)
	total_labels = np.concatenate((
		labels, np.array(ret_mask_seq53), np.array(ret_mask_seq63), np.array(ret_mask_seq13)
	), axis=0)
	hist, freq = pixel_hist(total_labels)
	print(freq)
	return ret_mask_seq53, ret_mask_index_seq53, ret_mask_seq63, ret_mask_index_seq63, ret_mask_seq13, ret_mask_index_seq13


class Mixup:
	def __init__(self):
		self.root = "/home/sun/contest/data_preprocessed"
		self.base_dirs = ["HH", "HV", "VH", "VV"]
		self.name = None
		self.image_dir = None
		self.mask_dir = None
		self.all_zero = []
		self.max_log_pix = None
		self.max_log_denoise_pix = 0
		self.max_log_denoise_normal_pix = 0
		self.have_zero_aft_log = []
		self.majority_record = defaultdict(list)

	def get_hist(self, labels=None):
		"""
		5 6 1 < 2 4 3
		:param th:
		   [0,   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1]
		5    [150, 11,   5,   1,   0,   0,   1,   0,   0,   0]
		6    [53,  16,   5,   1,   4,   0,   2,   0,   1,   1]
		1    [286, 32,  18,   8,   5,   3,   3,   7,   6,   6]
		2    [129, 65,  51,  36,  30,  26,  24,   5,   2,   1]
		4    [111, 58,  62,  49,  34,  35,  37,  19,  18,   6]
		3    [70,  68,  72,  57,  48,  41,  40,  33,  26,  12]
		> th
		5  [168, 18,    7,   2,   1,   1,   1,   0,   0,   0]
		6  [83,  30,   14,   9,   8,   4,   4,   2,   2,   1]
		1  [375, 89,   57,  39,  31,  26,  23,  20,  13,   7]
		2  [369, 240, 175, 124,  88,  58,  32,   8,   3,   1]
		4  [429, 318, 260, 198, 149, 115,  80,  43,  24,   6]
		3  [467, 397, 329, 257, 200, 152, 111,  71,  38,  12]
		0 <  < th
		5  [0,  150, 161, 166, 167, 167, 167, 168, 168, 168, 168]
		6  [0,   53,  69,  74,  75,  79,  79,  81,  81,  82,  83]
		1  [0,  286, 318, 336, 344, 349, 352, 355, 362, 368, 374]
		2  [0,  129, 194, 245, 281, 311, 337, 361, 366, 368, 369]
		4  [0,  111, 169, 231, 280, 314, 349, 386, 405, 423, 429]
		3  [0,   70, 138, 210, 267, 315, 356, 396, 429, 455, 467]
		"""
		self.majority_record = defaultdict(list)
		for t in [5, 6, 1, 2, 4, 3]:
			for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
				index_table = []
				for index in range(500):
					label = labels[index]
					fg_mask = label == t
					ratio = fg_mask.sum() / fg_mask.size
					if th - 0.1 < ratio <= th:
						index_table.append(index)
				self.majority_record[t].append(index_table)
		# bg_mask = label != t
		return 0

	def get_label(self):
		root_path = self.root
		label = sio.loadmat(root_path + "/label.mat")
		label = label["label"]  # 512,512,500   H, W
		label[label == 7] = 0
		return label.transpose(2, 0, 1)  # 500,512,512

	def get_img_paths(self, index):
		index = str(index + 1)
		img_paths = []
		for i in range(4):
			base_dir = self.base_dirs[i]
			img_path = os.path.join(self.root, base_dir, index + "_" + base_dir + ".tiff")
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
		trainval_split_f = os.path.join(splits_dir, "trainval" + '.txt')

		if not lite:
			if not train_split and not val_split:
				train_split = [str(i) for i in range(400)]
				val_split = [str(i) for i in range(400, 500)]
			with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
				fp.write('\n'.join(train_split) + '\n')
			with open(os.path.join(splits_dir, val_split_f), 'w') as fp:
				fp.write("\n".join(val_split) + '\n')
			with open(os.path.join(splits_dir, trainval_split_f), 'w') as fp:
				fp.write("\n".join(train_split + val_split) + '\n')
			return train_split_f, val_split_f
		if lite:
			train_split = [str(0), str(1)]
			with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
				fp.write('\n'.join(train_split) + '\n')
			val_split = [str(400), str(401)]
			with open(os.path.join(splits_dir, val_split_f), 'w') as fp:
				fp.write("\n".join(val_split) + '\n')
			with open(os.path.join(splits_dir, trainval_split_f), 'w') as fp:
				fp.write("\n".join(train_split + val_split) + '\n')
			return train_split_f, val_split_f

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
		# im_datas = denoise(im_datas, im_datas_org)
		return im_datas

	def keep4_4c4_2c2_10(self, img_paths, x, HH=None, HV=None, VH=None, VV=None):
		"""
		在svm, nn上效果最好的特征组合
		:param img_paths:
		:param image, 4, 512, 512, 已经过log和归一化处理
		:return:
		"""
		if HH is None and HV is None and VH is None and VV is None:
			HH, HV, VH, VV = self.cat_4(img_paths, x, th=2)

		channel_0 = HH
		channel_0 = channel_0[:, :, np.newaxis]
		channel_1 = HV
		channel_1 = channel_1[:, :, np.newaxis]
		channel_2 = VH
		channel_2 = channel_2[:, :, np.newaxis]
		channel_3 = VV
		channel_3 = channel_3[:, :, np.newaxis]

		tmp = np.sqrt(HH * HH + VV * VV)
		if tmp.min() == 0:
			self.have_zero_aft_log.append(x)

		channel_4 = VH / HH
		channel_4 = channel_4[:, :, np.newaxis]
		channel_5 = HV / tmp
		channel_5 = channel_5[:, :, np.newaxis]
		channel_6 = VH / tmp
		channel_6 = channel_6[:, :, np.newaxis]
		channel_7 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
		channel_7 = channel_7[:, :, np.newaxis]

		tmp1 = np.abs(HV + VH)
		tmp2 = HH + VV

		channel_8 = np.sqrt(tmp1 * tmp2)
		channel_8 = channel_8[:, :, np.newaxis]
		channel_9 = np.sqrt(HV * VH)
		channel_9 = channel_9[:, :, np.newaxis]
		# channel_10 = np.sqrt(HH * HH + VV * VV + VH * VH + HV * HV)
		# channel_10 = channel_10[:, :, np.newaxis]

		ret = np.concatenate((channel_0, channel_1, channel_2, channel_3, channel_4,
							  channel_5, channel_6, channel_7, channel_8, channel_9), axis=2)
		return ret

	def get_new_4(self, index_seq, source=5, labels=None):
		# 除了最后一张,下一张图片前景的位置如果和前一张有重合,保留前一张中重合的部分
		target_mask_index = index_seq[-1]
		HHs = np.zeros((512, 512)).astype("float32")
		HVs = np.zeros((512, 512)).astype("float32")
		VHs = np.zeros((512, 512)).astype("float32")
		VVs = np.zeros((512, 512)).astype("float32")
		for j, index in enumerate(index_seq[:-1]):
			# index = ret_mask_index_seq53[i][:-1][0]
			label = labels[index]
			img_paths = self.get_img_paths(index)
			HH, HV, VH, VV = self.cat_4(img_paths, index)  # np array
			bg_mask = label != source
			HHs = mix_source_image(HH, HHs, bg_mask)
			HVs = mix_source_image(HV, HVs, bg_mask)
			VHs = mix_source_image(VH, VHs, bg_mask)
			VVs = mix_source_image(VV, VVs, bg_mask)
		target_label = labels[target_mask_index]
		# putpalette(target_label)
		target_img_paths = self.get_img_paths(target_mask_index)
		HH, HV, VH, VV = self.cat_4(target_img_paths, target_mask_index)  # np array
		# plt.imshow(HH)
		HHs = mix_target_image(HH, HHs)
		HVs = mix_target_image(HV, HVs)
		VHs = mix_target_image(VH, VHs)
		VVs = mix_target_image(VV, VVs)
		# plt.imshow(HHs)
		return HHs, HVs, VHs, VVs

	def save_pkl(self, train_new_100, ret_mask_seq53, ret_mask_index_seq53, count, mode=None, labels=None):
		for i, x in enumerate(train_new_100):  # 一共100张,100个混叠序列,每个序列长度不超过21
			new_label = ret_mask_seq53[i]
			# putpalette(new_label)
			index_seq = ret_mask_index_seq53[i]
			HHs, HVs, VHs, VVs = self.get_new_4(index_seq, source=5, labels=labels)
			if mode == "log_normal_new_noise_4channel_keep4_4c4_2c2_10_blc700":
				im = self.keep4_4c4_2c2_10(None, int(x), HH=HHs, HV=HVs, VH=VHs, VV=VVs)
			image = os.path.join(self.image_dir, x + ".pkl")
			mask = os.path.join(self.mask_dir, x + ".pkl")
			with open(image, "wb") as img_f:
				pickle.dump(im, img_f)
			with open(mask, "wb") as mask_f:
				pickle.dump(new_label, mask_f)
			print(x)
			count += 1
			print(count)
		return count

	def trans_new200_to_psc_format(self, train_new_100, train_new_70, train_new_30, count, mode=None, labels=None):
		ret_mask_seq53, ret_mask_index_seq53, ret_mask_seq63, ret_mask_index_seq63, ret_mask_seq13, ret_mask_index_seq13 \
			= get_balance_700(labels=labels)

		count = self.save_pkl(train_new_100, ret_mask_seq53, ret_mask_index_seq53, count, mode=mode, labels=labels)
		count = self.save_pkl(train_new_70, ret_mask_seq63, ret_mask_index_seq63, count, mode=mode, labels=labels)
		count = self.save_pkl(train_new_30, ret_mask_seq13, ret_mask_index_seq13, count, mode=mode, labels=labels)
		return

	def transform_all_to_psc_format(self, mode=None, th=None, lite=None):
		"""
		将原始tiff数据转为 VOC文件目录格式
		:param mode: 对tiff文件操作的类型："log_normal", "log_normal_c3"
		:return: .pkl文件
		"""
		train_split, val_split, train_new_100, train_new_70, train_new_30 = get_new_split()
		train_split_f, val_split_f = self.generate_dir(mode, lite, train_split, val_split)

		self.all_zero = list(set(self.all_zero))
		# self.max_log_pix = 11.090339660644531250

		labels = self.get_label()

		count = 0
		for split_f in [train_split_f, val_split_f]:
			with open(os.path.join(split_f), "r") as f:
				file_names = [x.strip() for x in f.readlines()]
			for x in file_names:
				if int(x) >= 500:
					continue
				label = labels[int(x)]
				img_paths = self.get_img_paths(int(x))
				if mode == "log_normal_new_noise_4channel_keep4_4c4_2c2_10":
					im = self.keep4_4c4_2c2_10(img_paths, int(x))
				if mode == "log_normal_new_noise_4channel_keep4_4c4_2c2_10_blc700":
					im = self.keep4_4c4_2c2_10(img_paths, int(x))
				image = os.path.join(self.image_dir, x + ".pkl")
				mask = os.path.join(self.mask_dir, x + ".pkl")
				with open(image, "wb") as img_f:
					pickle.dump(im, img_f)
				with open(mask, "wb") as mask_f:
					pickle.dump(label, mask_f)
				print(x)
				count += 1
				print(count)

		self.trans_new200_to_psc_format(train_new_100, train_new_70, train_new_30, count, mode=mode, labels=labels)

		print(self.all_zero)
		print("max_log_denoise_pix is (%.30f)" % self.max_log_denoise_pix)
		print("max_log_denoise_normal_pix is (%.30f)" % self.max_log_denoise_normal_pix)
		print(self.have_zero_aft_log)
		print("Finised")
		return


if __name__ == "__main__":
	demo = Mixup()
	# demo.get_hist(labels=labels)
	demo.transform_all_to_psc_format(mode="log_normal_new_noise_4channel_keep4_4c4_2c2_10_blc700")
