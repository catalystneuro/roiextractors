
from SchnitzerTraceExtractor import SchnitzerTraceExtractor
import matplotlib.pyplot as plt

trext = SchnitzerTraceExtractor(
    '2014_04_01_p203_m19_check01_cnmfeAnalysis.mat', 'cnmfe', 'recording_device')
type(trext.data.samp_freq)
print(trext.data.A.shape, trext.data.b.shape, trext.data.C.shape,
      trext.data.f.shape, trext.data.YrA.shape)

trext.nwbwrite('2014_04_01_p203_m19_check01_cnmfeAnalysis_nwbwrite.nwb',
               '2014_04_01_p203_m19_check01_cnmfeAnalysis.mat', 'cnmfe')

plt.imshow(trext.data.cn)
plt.show()
