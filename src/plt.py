# -*- coding: utf-8 -*-

# @Time    : 19-5-28 上午9:40
# @Author  : zj

import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.font_manager import _rebuild
#
# _rebuild()  # reload一下
#
# plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

loss_list = [2.302556029875231, 2.3023752576917773, 2.302213968770917, 2.302069007673369, 2.3019383350019864,
             2.301820643190009, 2.301715020189562, 2.301620638013411, 2.3015361471752818, 2.3014593851605354,
             2.3013835631899866, 2.3012585757543333, 2.2754265353523766, 1.0151469689179775, 0.12957442900606064,
             0.07924755380467023, 0.059377517402276564, 0.04784209422326069, 0.04012046139258493, 0.03440057398689781,
             0.029610995233653063, 0.025596381816838213, 0.022268727275238123, 0.01935233909624217,
             0.017452230087815532, 0.01517086603934174, 0.012776875999448278, 0.011326786128295248, 0.01067212663980899,
             0.00937635587912064, 0.00804156781260717, 0.006571647632968941, 0.0067174118626119285,
             0.006212232802066548, 0.008338740957434573, 0.009549657733584146, 0.004933481723330252,
             0.0046929376115362065, 0.005173956315392346, 0.005160730197188488, 0.0025902158436050004,
             0.0021452391526514637, 0.0038704902173888637, 0.0032729189254562707, 0.005186472227056966,
             0.002088116020177016, 0.0015505776185115838, 0.0009924224330851515, 0.0008211597583450272,
             0.0005792399694171945, 0.0004971660052432878, 0.0004341765341373127, 0.00032398530435060553,
             0.00024982043831881824, 0.00021605177681057236, 0.00019175312664735782, 0.00017308946503729748,
             0.0001579057094740817, 0.00014680063641044763, 0.00013637402072102055, 0.0001272482814292511,
             0.00011969322010777006, 0.0001131724496217796, 0.0001073164936218717, 0.00010209120122083167,
             9.739567776167088e-05, 9.327350970350927e-05, 8.9319862807232e-05, 8.599096273859106e-05,
             8.243813162423526e-05, 7.952898655354254e-05, 7.697152767491635e-05, 7.430277174564331e-05,
             7.194165580328042e-05, 6.974922349772667e-05, 6.776639585846356e-05, 6.578469677564696e-05,
             6.415805363850417e-05, 6.24730601561223e-05, 6.107863265155598e-05, 5.9564434617104595e-05,
             5.8164317593854164e-05, 5.695152228209537e-05, 5.585605045533295e-05, 5.4581046952263886e-05,
             5.356472658839985e-05, 5.254667794046501e-05, 5.163391532449897e-05, 5.072713796987222e-05,
             4.992045185322827e-05, 4.9076791388957434e-05, 4.827229699390418e-05, 4.755834154207536e-05,
             4.684046714658067e-05, 4.6108190155766065e-05, 4.546235040718267e-05, 4.48581601362083e-05,
             4.421253020674745e-05, 4.369548632750244e-05, 4.3141234435343214e-05, 4.261409189264783e-05,
             4.2068640070510914e-05, 4.158801618113793e-05, 4.1131294817079376e-05, 4.0696662538666865e-05,
             4.0222013718015224e-05, 3.978861004830271e-05, 3.935128877973055e-05, 3.902766859870869e-05,
             3.8658845491891046e-05, 3.830872165182389e-05, 3.7948762462850835e-05, 3.759194122857658e-05,
             3.7263497809769046e-05, 3.695647263161183e-05, 3.665819628643738e-05, 3.6364265975363485e-05,
             3.606647265917626e-05, 3.5820042050022656e-05, 3.551746652249559e-05, 3.529395242321652e-05,
             3.502904377916119e-05, 3.477842358340395e-05, 3.45711384073994e-05, 3.434146986484856e-05,
             3.411787036131649e-05, 3.390729605401666e-05, 3.370919324044601e-05, 3.350703048367302e-05,
             3.332640456061281e-05, 3.313745082409741e-05, 3.2964137420086715e-05, 3.2805377178768966e-05,
             3.261404226008924e-05, 3.246917911271296e-05, 3.229007415040018e-05, 3.214719859738853e-05,
             3.202183809863342e-05, 3.184835438415585e-05, 3.1733172144021696e-05, 3.160163480720181e-05,
             3.1481219554547075e-05, 3.13233328126783e-05, 3.118062924218837e-05, 3.110053245773094e-05,
             3.09675244515895e-05, 3.086225002632342e-05, 3.073534500782075e-05, 3.0641010266985796e-05,
             3.053955146465813e-05, 3.045153730663643e-05, 3.032972002545277e-05, 3.023788216585911e-05,
             3.0127810068178038e-05, 3.008477789836003e-05, 2.9965068780130732e-05, 2.988288482754098e-05,
             2.9819688498540685e-05, 2.9716566809333245e-05, 2.961361636280694e-05, 2.957451937966642e-05,
             2.9471306539789472e-05, 2.9423542593091526e-05, 2.935148870886941e-05, 2.9262219712579007e-05,
             2.919370629471053e-05, 2.9131220581921563e-05, 2.9052534470813797e-05, 2.900450196791026e-05,
             2.8946036298311708e-05, 2.8885641346662815e-05, 2.8828638577649102e-05, 2.8782682854532894e-05,
             2.8688948312373944e-05, 2.866149087486892e-05, 2.860982128487159e-05, 2.8551650417641825e-05,
             2.851158844849665e-05, 2.8460789624919757e-05, 2.8390794328346843e-05, 2.8379951442928733e-05,
             2.83017212581985e-05, 2.827318893007556e-05, 2.8234947080047207e-05, 2.819856282763788e-05,
             2.814722602346141e-05, 2.8138348227951822e-05, 2.8053997344797332e-05, 2.8038853766201296e-05,
             2.800882619483558e-05, 2.7975521853438005e-05, 2.7937389118529435e-05, 2.7904845173708315e-05,
             2.786921542156391e-05, 2.784983694869486e-05, 2.781136258999535e-05, 2.7767863428860334e-05,
             2.775908201911923e-05, 2.771261597783913e-05, 2.7706747533076532e-05, 2.7674738593791694e-05,
             2.7641589083112753e-05, 2.7639897989652748e-05, 2.7591086084107017e-05, 2.7563085864461073e-05,
             2.758112271094363e-05, 2.7510056843657323e-05, 2.7515333417999932e-05, 2.7492442588612688e-05,
             2.7462037771637057e-05, 2.7438046828509566e-05, 2.7432118316663166e-05, 2.739947849387454e-05,
             2.7394924893595802e-05, 2.7372875032550336e-05, 2.7358783579498624e-05, 2.7333095969858004e-05,
             2.7329483220222346e-05, 2.7313286962999727e-05, 2.730360042728935e-05, 2.728212365958233e-05,
             2.7264493643091053e-05, 2.725296872303553e-05, 2.724663404729168e-05, 2.723365850950891e-05,
             2.7214017025984925e-05, 2.7223521393143856e-05, 2.718184471881781e-05, 2.7204709343534136e-05,
             2.7160871376224405e-05, 2.7184654257294035e-05, 2.7177267105044206e-05, 2.7146846553316858e-05,
             2.714572530896574e-05, 2.7134319666014916e-05, 2.7120133777718877e-05, 2.7143277970780102e-05,
             2.7119707972478567e-05, 2.7107952316824973e-05, 2.7124805802053463e-05, 2.7111418839256086e-05,
             2.7098334817665547e-05, 2.708849636706174e-05, 2.708059951944654e-05, 2.7099586721054603e-05,
             2.7083544100784577e-05, 2.709567526056006e-05, 2.7085715811490156e-05, 2.7065199226484803e-05,
             2.708214112245154e-05]


def draw(loss_list, title='LeNet-5 LOSS', xlabel='iterations / times', ylabel='loss value'):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    draw(loss_list)
