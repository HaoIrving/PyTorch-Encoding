
def black_move(pred):
    num = 0
    if pred[0, 0] == 0:
        num += 1
    if pred[0, 511] == 0:
        num += 1
    if pred[511, 0] == 0:
        num += 1
    if pred[511, 511] == 0:
        num += 1
    if num == 2:
        ###输入的label=0的像素数##
        # num1 = 0
        # for m in range(512):
        #     for n in range(512):
        #         if pred[m, n] == 0:
        #             num1 += 1
        # print(num1)

        #ls.append(i + 1)
        if pred[0, 0] == 0 and pred[0, 511] == 0:#up
            for m in range(512):
                for n in range(512):
                    if pred[n, m] != 0:
                        for l in range(8):
                            b =n+l
                            if n+l >511:
                                b=511
                            pred[b, m] =0
                        break
        if pred[511, 0] == 0 and pred[511, 511] == 0:#down
            for m in range(512):
                for n in range(512):
                    if pred[n, m] == 0:
                        for l in range(1, 21):
                            c = n-l
                            if n - l <0:
                                c = 0
                            pred[c, m] = 0
                        break
        if pred[0, 0] == 0 and pred[511, 0] == 0:#left
            for n in range(512):
                for m in range(512):
                    if pred[n, m] != 0:
                        for l in range(30):
                            d = m+l
                            if m+l>511:
                                d = 511
                            pred[n, d] = 0
                        break
        if pred[0, 511] == 0 and pred[511, 511] == 0:#right
            for n in range(512):
                for m in range(512):
                    if pred[n, m] == 0:
                        for l in range(1, 36):
                            a = m - l
                            if m - l < 0:
                                a = 0
                            pred[n, a] = 0
                        break
        #####  计算平移后的图像的label=0像素数
        # num2 = 0
        # for h in range(512):
        #     for v in range(512):
        #         if pred[h, v] == 0:
        #             num2 += 1
        # print(num2)
        # print(num2 - num1)
    return pred
import  cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    img = cv2.imread('D:\\liuweiwei\\data\\27_feature.png')
    img = img[:,:,2]
    img11 = black_move(img)
    plt.imshow(img)
    #plt.imshow(img11)
    plt.show()