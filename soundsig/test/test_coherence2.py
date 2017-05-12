# Testing the coherence

import numpy as np
import matplotlib.pyplot as plt

from lasp.coherence import coherence_jn
from lasp.signal import lowpass_filter

# Make two gaussian signals
sample_rate = 1000.0
tlen = 2.0  # 2 second signal

# Make space for both signals
s1 = np.random.normal(0, 1, int(tlen*sample_rate))
s2 = lowpass_filter(s1, sample_rate, 250.0) + np.random.normal(0, 1, int(tlen*sample_rate))


freq1,c_amp,c_var_amp,c_phase,c_phase_var, cohe_unbiased, cohe_se  = coherence_jn(s1, s2, sample_rate, 0.1, 0.05)

plt.figure()

plt.plot(freq1, cohe_unbiased, 'k-', linewidth=2.0, alpha=0.9)
plt.plot(freq1, cohe_unbiased+2*(cohe_se), 'g-', linewidth=2.0, alpha=0.75)
plt.plot(freq1, cohe_unbiased-2*(cohe_se), 'c-', linewidth=2.0, alpha=0.75)

plt.show()