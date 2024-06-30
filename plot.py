import matplotlib.pyplot as plt
import scipy

class PredictionBand(object):
    """Plot bands of model predictions as calculated from a chain.
    Adopted from UltraNest.

    call add(y) to add predictions from each chain point

    .. testsetup::

        import numpy
        chain = numpy.random.uniform(size=(20, 2))

    .. testcode::

        x = numpy.linspace(0, 1, 100)
        band = PredictionBand(x)
        for c in chain:
            band.add(c[0] * x + c[1])
        # add median line. As an option a matplotlib ax can be given.
        band.line(color='k')
        # add 1 sigma quantile
        band.shade(color='k', alpha=0.3)
        # add wider quantile
        band.shade(q=0.01, color='gray', alpha=0.1)
        plt.show()

    To plot onto a specific axis, use `band.line(..., ax=myaxis)`.

    Parameters
    ----------
    x: array
        The independent variable

    """

    def __init__(self, x, shadeargs={}, lineargs={}):
        """Initialise with independent variable *x*."""
        self.x = x
        self.ys = []
        self.shadeargs = shadeargs
        self.lineargs = lineargs

    def add(self, y):
        """Add a possible prediction *y*."""
        self.ys.append(y)

    def set_shadeargs(self, **kwargs):
        """Set matplotlib style for shading."""
        self.shadeargs = kwargs

    def set_lineargs(self, **kwargs):
        """Set matplotlib style for line."""
        self.lineargs = kwargs

    def get_line(self, q=0.5):
        """Over prediction space x, get quantile *q*. Default is median."""
        if not 0 <= q <= 1:
            raise ValueError("quantile q must be between 0 and 1, not %s" % q)
        assert len(self.ys) > 0, self.ys
        return scipy.stats.mstats.mquantiles(self.ys, q, axis=0)[0]

    def shade(self, q=0.341, ax=None, **kwargs):
        """Plot a shaded region between 0.5-q and 0.5+q, by default 1 sigma."""
        if not 0 <= q <= 0.5:
            raise ValueError("quantile distance from the median, q, must be between 0 and 0.5, not %s. For a 99%% quantile range, use q=0.48." % q)
        shadeargs = dict(self.shadeargs)
        shadeargs.update(kwargs)
        lo = self.get_line(0.5 - q)
        hi = self.get_line(0.5 + q)
        if ax is None:
            ax = plt
        return ax.fill_between(self.x, lo, hi, **shadeargs)

    def line(self, ax=None, **kwargs):
        """Plot the median curve."""
        lineargs = dict(self.lineargs)
        lineargs.update(kwargs)
        mid = self.get_line(0.5)
        if ax is None:
            ax = plt
        return ax.plot(self.x, mid, **lineargs)