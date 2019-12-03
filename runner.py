
from scnitzerTraceExtractor import ScnitzerTraceExtractor
import matplotlib.pyplot as plt

trext=ScnitzerTraceExtractor('2014_04_01_p203_m19_check01_cnmfeAnalysis.mat','cnmfe','recording_device')
type(trext.data.SampFreq)
print(trext.data.A.shape,trext.data.b.shape,trext.data.C.shape,trext.data.f.shape,trext.data.YrA.shape)
plt.imshow(trext.data.Cn)
plt.show()
trext.nwbwrite('2014_04_01_p203_m19_check01_cnmfeAnalysis_nwbwrite.nwb','2014_04_01_p203_m19_check01_cnmfeAnalysis.mat','cnmfe')
