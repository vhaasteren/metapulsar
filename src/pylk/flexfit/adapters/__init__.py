"""Frontend adapters that translate Discovery / Enterprise / nltiming into flexfit.

The numerical core (``pylk.flexfit``) is frontend-neutral. These adapters are
the only modules that import Discovery, Enterprise, or ``nltiming``:

* :mod:`pylk.flexfit.adapters.discovery` — GP basis blocks (red/DM/chromatic)
  and the white-noise operator, reusing Discovery's ``fourierbasis``/``powerlaw``.
* :mod:`pylk.flexfit.adapters.enterprise` — **planned** equivalent using
  Enterprise ``gp_bases`` / ``gp_priors`` (stub; not yet implemented).
* :mod:`pylk.flexfit.adapters.nltiming` — the timing ``J_z`` block and the
  finite-difference sign check, from an ``nltiming`` ``TimingContext``.
"""
