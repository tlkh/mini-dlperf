print("Start of script")

import time
import horovod.tensorflow.keras as hvd

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()

print("This is", hvd_rank, "out of", hvd_size)

time.sleep(1)

print(hvd_rank, "now exit")
