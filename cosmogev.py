import numpy as np

from scipy.integrate import simps
from scipy.stats import binned_statistic, genextreme
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from scipy.special import gamma

from itertools import repeat, product
from typing import Any


RHO_CRIT0_ASTRO = 2.77536627E+11 # critical density in h^2 Msun / Mpc^3

# =======================================================================================
# power spectrum models
# =======================================================================================

def eisenstien98_zb(k: Any, h: float, Om0: float, Ob0: float, Tcmb0: float = 2.725) -> Any:
    r"""
    Transfer function given by Eisenstein and Hu (1998), without baryonic oscillations.

    Parameters
    ----------
    k: array_like
        Wavenumber in h/Mpc.
    h: float
        Hubble parameter in units of 100 km/sec/Mpc.
    Om0: float
        Present day matter density.
    Ob0: float
        Present day baryon density.
    Tcmb0: float, optional
        CMB temperature in K. Default value if 2.725 K.

    Returns
    -------
    Tk: array_like
        Transfer function values. Has the same shape as k.

    """
    k = np.asarray(k)

    # setting cosmological parameters
    h2    = h * h
    Omh2  = Om0 * h2
    Obh2  = Ob0 * h2
    theta = Tcmb0 / 2.7 # cmb temperature in units of 2.7 K
    wt_b  = Ob0 / Om0   # fraction of baryons

    # sound horizon : eqn. 26
    s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**(3/4))

    # eqn. 31
    alpha_gamma = 1 - 0.328*np.log(431*Omh2)*wt_b + 0.38*np.log(22.3*Omh2)*wt_b**2

    # eqn. 30
    gamma_eff   = Om0*h*(alpha_gamma + (1 - alpha_gamma)/(1 + (0.43*k*s)**4))

    # eqn. 28
    q   = k*(theta*theta/ gamma_eff)

    # eqn. 29
    l_0 = np.log(2*np.e + 1.8*q)
    c_0 = 14.2 + 731.0 / (1 + 62.5*q)
    T_0 = l_0 / (l_0 + c_0*q**2)
    return T_0

def sugiyama95(k: Any, h: float, Om0: float, Ob0: float, Tcmb0: float = 2.725) -> Any:
    r"""
    Transfer function given by Bardeen et al, with correction by Sugiyama (1995).

    Parameters
    ----------
    k: array_like
        Wavenumber in h/Mpc.
    h: float
        Hubble parameter in units of 100 km/sec/Mpc.
    Om0: float
        Present day matter density.
    Ob0: float
        Present day baryon density.
    Tcmb0: float, optional
        CMB temperature in K. Default value if 2.725 K.

    Returns
    -------
    Tk: array_like
        Transfer function values. Has the same shape as k.
        
    """
    k = np.asarray(k)

    # eqn. 4
    gamma = Om0 * h
    q     = k / gamma * np.exp(Ob0 + np.sqrt(2*h) * Ob0 / Om0) 

    # transfer function : eqn. 3
    T = np.log(1 + 2.34*q) / (2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)

    # for smaller `k`, transfer function returns `0` instead one. fixing it :
    if k.shape == (): # scalar `k`
        if k < 1.0e-09:
            T = 1.0
    else: # array `k`
        T[k < 1.0e-09] = 1.0
    return T


# =======================================================================================
# cosmology object
# =======================================================================================

class Cosmology:
    r"""
    A simple object to store the cosmology model and calculations. The model is a Lambda-CDM 
    model with constant dark energy.

    Parameters
    ----------
    h: float
        Hubble parameter in units of 100 km/sec/Mpc.
    Om0: float
        Present day matter density.
    Ob0: float
        Present day baryon density.
    ns: float
        Index of the present day power spectrum.
    sigma8: float
        RMS variance of the matter density fluctuations.
    Ode0: float
        Present day dark energy density. Required only for non-flat cosmologies.
    flat: bool, optional
        Tells whether the cosmology is flat or not. Default is a flat one.
    Tcmb0: float, optional
        CMB temperature in K. Default value if 2.725 K.

    Attributes
    ----------
    psmodel: str
        Model to use for linear matter power spectrum. Default is the Eisenstein & Hu 
        model without baryons.
    exactgf: bool
        Tells if to use the exact growth factor or the fit by Carrolll et al (1992).
    A: float
        Normalization of the linear matter power spectrum.

    """

    __slots__ = 'h', 'Om0', 'Ob0', 'Ode0', 'Ok0', 'flat', 'ns', 'sigma8', 'Tcmb0', 'A', 'psmodel', 'exactgf'

    def __init__(self, h: float, Om0: float, Ob0: float, ns: float, sigma8: float, Ode0: float = None, flat: bool = True, Tcmb0: float = 2.725):

        assert h > 0, "h must be positive"
        self.h = h

        assert Om0 > 0, "Om0 must be positive"
        assert 0 < Ob0 <= Om0, "Ob0 must be positive and cannot exceed Om0"
        self.Om0, self.Ob0 = Om0, Ob0

        assert sigma8 > 0, "sigma8 must be positive"
        self.sigma8 = sigma8
        
        self.ns = ns

        assert Tcmb0 > 0, "Tcmb0 must be postive"
        self.Tcmb0 = Tcmb0


        if flat:
            self.Ode0 = 1 - self.Om0
            self.Ok0  = 0.0
        else:
            assert Ode0 is not None, "Ode0 must be given for non-flat cosmology"

            self.Ode0 = Ode0
            self.Ok0  = 1 - self.Om0 - self.Ode0

            if self.Ok0 < 1e-08:
                flat = True

        assert self.Ode0 > 0, "Ode0 must be positive"

        self.flat = flat

        self.exactgf = True
        self.psmodel = 'eisenstien98_zb'

        self.A = 1.0

    def Om(self, z: Any) -> Any:
        r"""
        Evolution of matter density parameter :math:`\Omega_m` with redshift.
        """

        zp1 = np.asfarray(z) + 1
        
        x = self.Om0 * zp1**3

        y = x + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2

        return x / y

    def Ode(self, z: Any) -> Any:
        r"""
        Evolution of darak-energy density parameter :math:`\Omega_{de}` with redshift.
        """

        zp1 = np.asfarray(z) + 1

        y = self.Om0 * zp1**3 + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2
            
        return self.Ode0 / y

    def E(self, z: Any) -> Any:
        r"""
        Evolution of Hubble parameter :math:`H(z)` with redshift, in units of present value.
        """

        zp1 = 1 + np.asfarray(z)

        y = self.Om0 * zp1**3 + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2
        return np.sqrt(y)

    def Dplus(self, z: Any, norm: bool = True) -> Any:
        r"""
        Compute the linear growth factor. Use the fit by carroll et al (1992) if the property 
        `exactgf` is false, else compute by integrating the growth equation.

        Parameters
        ----------
        z: array_like
            Redshift.
        norm: bool, optional
            If set true, normalize the growth factor so that its present value to be 1.

        Returns
        -------
        Dplus: array_like
            Computed value of the linear growth factor.

        """

        def carroll92(z):
            """
            Growth factor fit given by Carroll et al (1992).
            """

            zp1 = 1 + np.asfarray(z)
            
            u, v = self.Om0 * zp1**3, self.Ode0
            x    = u + v
            if not self.flat:
                x += self.Ok0 * zp1**2
            u, v = u / x, v / x

            return 2.5 / zp1 * u * ( u**(4./7) - v + (1 + u/2) * (1 + v/70) )**(-1)

        def exact(z):
            """
            Growth factor computed by integration of the growth equation.
            """

            zp1   = 1 + np.asfarray(z)
            a, da = np.linspace(0, 1/zp1, 10001, retstep = True)

            y = self.Om0 + self.Ode0 * a**3
            if not self.flat:
                y += self.Ok0 * a
            y = (a / y)**1.5
            
            return 2.5 * self.Om0 * self.E(z) * simps(y, dx = da, axis = -1)

        dplus = exact(z) if self.exactgf else carroll92(z)

        if norm:
            dplus0 = exact(0) if self.exactgf else carroll92(0)
            dplus  = dplus / dplus0
        
        return dplus

    def transfer(self, k: Any) -> Any:
        r"""
        Compute the value of the transfer function at the given wavenumber.
        """

        if self.psmodel == 'eisenstien98_zb':
            return eisenstien98_zb(k, self.h, self.Om0, self.Ob0, self.Tcmb0)
        elif self.psmodel == 'sugiyama95':
            return sugiyama95(k, self.h, self.Om0, self.Ob0, self.Tcmb0)
        raise NotImplementedError(self.psmodel)

    def matterPowerSpectrum(self, k: Any, z: float = 0.0, dim: bool = True) -> Any:
        r"""
        Compute the linear matter power spectrum.

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift, default value is 0.
        dim: bool, optional
            If set to false, return the dimensionless power spectrum. Otherwise return the 
            power spectrum in units of :math:`Volume^{-1}`.

        Returns
        --------
        y: array_like
            Linear matter power spectrum. This will have same shape as k.
        
        """

        k = np.asfarray(k)
        y = self.A * k**self.ns * self.transfer(k)**2
        
        if not dim:
            y = y * k**3 / (2 * np.pi**2)
        
        if z > 0.0:
            y = y * self.Dplus(z)**2

        return y

    def variance(self, r: Any, z: float = 0.0, filter: str = 'tophat') -> Any:
        r"""
        Compute the rms linear matter variance of the density fluctuations, smoothed at a 
        scale :math:`r`.

        .. math::
            \sigma(r) = \int_0^{\infty} \Delta^2(k) w^2(kr) \frac{dk}{k}

        Parameters
        ----------
        r: array_like
            Smoothing scale in Mpc/h.
        z: float, optional
            Redshift, default is 0.
        filter: str, optional
            Filter used to smooth the density. It must be any of `tophat`, `gauss` or `sharpk` 
            corresponding to a spherical tophat, gaussian or sharp-k filter. Default is the 
            tophat filter.

        """

        r = np.asfarray(r)

        ka, kb = 1e-06, 1e+06 # integration limits

        # the sharp-k filter is a step function with value 1 for scales less than kr = pi
        # and 0 otherwise. to compute the variance in this case, the upper limit changed 
        # to the corrsponding scale to r, as after that the integrand is 0. 
        if filter == 'sharpk':
            kb = np.pi / r    

        # integration is done in the variable log(k)
        lnka, lnkb = np.log(ka), np.log(kb)
        lnk, dlnk  = np.linspace(lnka, lnkb, 10001, retstep = True)

        k = np.exp(lnk)
        
        w = 1.0
        if filter == 'tophat':
            w = np.outer(r, k)
            w = (np.sin(w) - w * np.cos(w)) * 3. / w**3 # spherical tophat filter 
        elif filter == 'gauss':
            w = np.outer(r, k)
            w = np.exp(-0.5 * w**2) # gaussian filter
        elif filter != 'sharpk':
            raise NotImplementedError(filter)
        
        y = self.matterPowerSpectrum(k, z, False) * w**2
        
        # in the case of sharp-k filter integration, the integrand's shape is transposed. 
        if filter == 'sharpk':
            y = y.T

        y = simps(y, axis = -1) * dlnk # integration by simpson's rule

        # if r is array, return an array. otherwise a scalar value if returned
        if np.ndim(r):
            return y

        return y[0] if np.ndim(y) else y # if y is array of size 1, get the value

    def normalizePower(self, filter: str = 'tophat'):
        r"""
        Normalise the linear matter power spectrum so that the rms variance at 8 Mpc/h scale 
        is same as the :math:`sigma_8` parameter.
        """

        self.A = 1.0

        var8   = self.variance(8, 0.0, filter)
        self.A = self.sigma8**2 / var8
        return self

    def effectiveIndex(self, k: Any, z: float = 0.0, h: float = 0.01) -> Any:
        r"""
        Compute the logarithmic slope of the linear power spectrum at a scale k. It is given 
        by :math:`n_s(k) = d\ln(k)/d\ln(k)`.

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift, default is 0.
        h: float, optional
            Relative step size to use for numerical differentiation. Default is 0.01.

        Returns
        -------
        ns: array_like
            Effective slope. Has the same shape as k.
        """

        k = np.asfarray(k)

        k1, k2 = (1 - h)*k, (1 + h)*k
        
        y1 = self.matterPowerSpectrum(k1, z, True)
        y2 = self.matterPowerSpectrum(k2, z, True)

        return (np.log(y2) - np.log(y1)) / (np.log(k2) - np.log(k1))


# ========================================================================================
# probability distribution object
# ========================================================================================

class Genextreme:
    r"""
    A generalised extreme value (GEV) distribution for the matter density fluatuations.

    Parameters
    ----------
    cellsize: float
        Pixel or cell size is the size of the cells in which the entire space is divided 
        into so that to compute the density.
    cm: Cosmology
        Cosmology model objcet to use. Must be a `Cosmology` object.
    p: int, optional
        Corresponds to the scheme used for density estimation. NGP will have p = 1 (default), 
        CIC have p = 2 and TSC have p = 3.

    """

    __slots__ = 'cellsize', 'kn', 'cm', 'p', 'pspline', 'param', 

    def __init__(self, cellsize: float, cm: Cosmology, p: int = 1) -> None:
        self.cellsize = cellsize
        self.kn       = np.pi / cellsize
        self.cm       = cm
        self.p        = p

        self._makeMeasuredPowerSpline()
        self.param = None

    def linearVariance(self, z: float = 0.0):
        r"""
        Compute the rms value of the density fluatuations in the cell (variance). This uses 
        a sharp-k filter at scale equal to the cellsize. This scale is the nyquist wavenumber 
        :math:`k_N = \pi / l`.

        .. math::
            \sigma_{lin}(k_N) = \int_0^{k_N} \Delta^2(k) \frac{dk}{k}

        Parameters
        ----------
        z: float, optional
            Redshift. Default value is 0.
        
        Returns
        -------
        sigma_lim: float
            Value of the linear variance in the cell.

        """
        return self.cm.variance(self.cellsize, z, filter = 'sharpk')

    def logVariance(self, lin: float):
        r"""
        Compute the variance of the field :math:`A =\ln(1 + \delta)` in terms of the linear 
        variance in the cell.

        Parameters
        ----------
        lin: float 
            Value of the linear variance in the cell.

        Returns 
        -------
        sigma_log: float
            Variance of the log field.
        """
        mu = 0.73
        return mu * np.log( 1 + lin / mu )

    def _makeMeasuredPowerSpline(self):
        r"""
        Compute the power spectrum that one get by measuring, including the effects of anti-
        aliasing and density interpolation. 
        
        (priavte method)

        See Also
        --------
        `Genextreme.measuredPower`
        """

        kn = self.kn
        p  = self.p

        def _power(k):
            """
            Linear matter power spectrum.
            """
            return self.cm.matterPowerSpectrum(k)

        def _measuredPower(k1, k2, k3):
            """
            Power spectrum measured from the cell densities including corrections.
            """

            k1, k2, k3 = np.asfarray(k1), np.asfarray(k2), np.asfarray(k3)
            
            y = 0.0
            for n1, n2, n3 in product( *repeat( range(-3, 4), 3 ) ):
                if n1**2 + n2**2 + n3**3 > 9.0:
                    continue
                q1, q2, q3 = k1 + 2*kn*n1, k2 + 2*kn*n2, k3 + 2*kn*n3

                w2 = ( np.sinc( 0.5*k1/kn ) * np.sinc( 0.5*k2/kn ) * np.sinc( 0.5*k3/kn ) )**(2*p)
                pk = _power( np.sqrt(q1**2 + q2**2 + q3**2) )
                y += pk * w2
            return y

        def randomVectors(n, kmin, kmax):
            """
            Generate random k vectors in a shell. 
            """
            k = np.logspace( np.log10(kmin), np.log10(kmax), n )
            t = np.random.uniform( 0, np.pi, n )
            p = np.random.uniform( 0, 2*np.pi, n )

            kx = k * np.sin(t) * np.cos(p)
            ky = k * np.sin(t) * np.sin(p)
            kz = k * np.cos(t) 
            return kx, ky, kz

        kx, ky, kz = randomVectors(10001, 1e-6, kn)

        pk_meas = _measuredPower(kx, ky, kz)
        klen    = np.sqrt( kx**2 + ky**2 + kz**2 )

        # bin the power spectrum values to make a spline for fast calculations
        kvals = np.logspace( np.log10(1e-6), np.log10(kn), 51 )
        pval, _, _ = binned_statistic( klen.flatten(), pk_meas.flatten(), statistic = 'mean', bins = kvals )
        kvals = 0.5 * ( kvals[1:] + kvals[:-1] )

        # power law continuation 
        alpha, beta = np.polyfit( np.log( kvals[ kvals > 0.5*kn ] ), np.log( pval[ kvals > 0.5*kn ] ), 1 )
        beta = np.exp(beta)

        kvals2 = np.logspace( np.log10(kn), np.log10( kn*np.sqrt(3) ), 3 )
        pval2  = beta * kvals2**alpha
        
        # store the power spectrum as spline for fast calculations
        self.pspline = CubicSpline(
                                    np.log( np.hstack( [kvals, kvals2] ) ),
                                    np.log( np.hstack( [pval, pval2] ) )
                                  )
        return

    def measuredPower(self, k, z: float = 0.0, bias2: float = 1.0):
        r"""
        Compute the power spectrum that get by measuring from the density values. This requires 
        corresctions for density interpolation and anti-aliasing. The corrected spectrum will 
        be 

        .. math::
            P_{meas}({\bf k}) = \sum_{\bf n} P({\bf k}')W^2({\bf k}')

        where :math:`{\bf k}' = {\bf k} + 2k_N {\bf n}` and :math:`\bf n` is an integer vector 
        of length less than 3. This is valid only when :math:`k < k_N`. After that point, it is 
        extended by a power law.

        For fast computations, this power spectrum is made into a spline. 

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift, default value is 0.
        bias2: float, optional
            Bias squared value. If not given, compute the linear power. If log field bias value
            :math:`b_A^2 = \sigma^2_A(k_N) / \sigma_{lin}(k_N)` is given, the power spectrum 
            corresponds to the log field. Default is 1.

        Returns
        -------
        y: array_like
            Value of the measured power spectrum.

        """
        pk0 = np.exp( self.pspline( np.log(k) ) )
        return bias2 * pk0 * self.cm.Dplus(z)**2

    def measuredVariance(self, z: float = 0.0, bias2: float = 1.0):
        r"""
        Compute the variance corresponding to the measured power spectrum.

        .. math::
            \sigma^2(l) = \frac{1}{(2\pi)^3}\int d^3k P_{meas}(k)
        
        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift, default value is 0.
        bias2: float, optional
            Bias squared value (see `Genextreme.measuredPower`). Default is 1. 

        Returns
        -------
        y: array_like
            Value of the measured variance.

        """
        kn   = self.kn
        y, w = np.polynomial.legendre.leggauss(64)

        m = 0.5*( np.log(kn) - np.log(1e-6) )
        c = m + np.log(1e-6)

        kx = np.exp(m*y + c)
        kx, ky, kz  = np.meshgrid( kx, kx, kx )

        y  = np.sqrt( kx**2 + ky**2 + kz**2 )
        w  = np.product( np.meshgrid( *repeat( w*m, 3 ) ), axis = 0 )

        y = np.sum( kx*ky*kz*np.exp( self.pspline( np.log(y) ) ) * w ) / (2*np.pi)**3
        return bias2 * y * self.cm.Dplus(z)**2

    def mean(self, lin: float):
        r"""
        Mean of the log density field in terms of the linear variance.      
        """
        lamda = 0.65
        return -lamda * np.log( 1 + 0.5 * lin / lamda )

    def skewness(self, meas, z: float = 0.0):
        r"""
        Skewness of the log density field distribution at a redshift z, in tems of the 
        measured log field variance.
        """
        nsp3 = self.cm.effectiveIndex(self.kn, z) + 3
        t = -0.7 * nsp3 + 1.25
        p = 0.06 - 0.26 * np.log(nsp3)
        return t * meas**(0.5 * p)

    def parametrize(self, z: float = 0.0):
        r"""
        Parameterise the distribution for the given redshift.
        """

        def gk(k, x):
            return gamma(1 - k*x)

        def f(x, gamma1):
            g1, g2, g3 = gk(1,x), gk(2,x), gk(3,x)
            return gamma1 + (g3 - 3*g1*g2 + 2*g1**3) / (g2 - g1**2)**1.5


        s2_lin = self.linearVariance(z)   # linear field variance
        s2_a   = self.logVariance(s2_lin) # log field variance
        
        bias2   = s2_a / s2_lin # log field bias value (squared)
        s2_meas = self.measuredVariance(z, bias2) # measured log field variance

        abar   = self.mean(s2_lin) # mean value
        gamma1 = self.skewness(s2_meas, z) # skewness

        shape = newton( f, 0.01, args = (gamma1, ) ) # shape parameter
        scale = abs(shape) * s2_meas**0.5 * ( gk(2, shape) - gk(1, shape)**2 )**(-0.5) # scale
        loc   = abar - (gk(1, shape) - 1) * scale / shape # location parameter

        self.param = shape, loc, scale
        return self

    def pdf(self, x):
        r"""
        Compute the distribution function using the estimated parameters
        """
        shape, loc, scale = self.param
        return genextreme.pdf(x, -shape, loc, scale)





if __name__ == '__main__':


    # mill: h=0.73, Om0=0.25, Ob0=0.045, Ode0=0.75, sigma8=0.9, ns=1.0, Tcmb0=2.7255K
    c = Cosmology(0.73, 0.25, 0.045, 1.0, 0.9, ).normalizePower()
    p = Genextreme(2.0, c).parametrize(0)

    x = np.linspace(-3, 2, 21)
    y = p.pdf(x)

    import matplotlib.pyplot as plt

    plt.figure()
    # plt.loglog()
    plt.plot(x, y, '-o')
    plt.show()
