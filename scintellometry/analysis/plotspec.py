#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pylab as plt

time_slice = slice(None)  # use all time samples
pol_select = (0, 3)
freq_average = None
phase_average = None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s foldspec icounts" % sys.argv[0]
        # Run the code as eg: ./plotspec.py data_foldspec.npy data_icounts.py.
        sys.exit(1)

    # Folded spectrum axes: time, frequency, phase, pol=4 (XX, XY, YX, YY).
    f = np.load(sys.argv[1])
    ic = np.load(sys.argv[2] if len(sys.argv) == 3 else
                 sys.argv[1].replace('foldspec', 'icount'))

    # Sum over the time axis, and over XX, YY if polarization data is present
    if f.shape[-1] == 4:
        n = f[time_slice, ..., pol_select].sum(-1).sum(0)
    else:
        n = f.sum(0)

    n /= ic[time_slice].sum(0)
    # normalize by median flux in a given frequency bin
    n_median = np.median(n, axis=1)
    nn = n / n_median[:, np.newaxis] - 1.

    if freq_average:
        nn = nn.reshape(-1, freq_average, nn.shape[1]).mean(1)

    if phase_average:
        nn = nn.reshape(nn.shape[0], -1, phase_average).mean(-1)

    vmin = nn.mean() - 1*nn.std()
    vmax = nn.mean() + 5*nn.std()

    plt.imshow(nn, aspect='auto', interpolation='nearest', origin='lower',
               cmap=plt.get_cmap('Greys'), vmin=vmin, vmax=vmax)
    # extent=(0., 1., 200., 400.))

    plt.xlabel('phase')
    plt.ylabel('f')  # (MHz)')
    # plt.title('PSR B1957')
    plt.colorbar()
    plt.show()
