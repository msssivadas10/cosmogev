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

        zp1 = np.asfarray(z) + 1
        
        x = self.Om0 * zp1**3

        y = x + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2

        return x / y

    def Ode(self, z: Any) -> Any:

        zp1 = np.asfarray(z) + 1

        y = self.Om0 * zp1**3 + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2
            
        return self.Ode0 / y

    def E(self, z: Any) -> Any:

        zp1 = 1 + np.asfarray(z)

        y = self.Om0 * zp1**3 + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2
        return np.sqrt(y)

    def Dplus(self, z: Any, norm: bool = True) -> Any:

        def carroll92(z):

            zp1 = 1 + np.asfarray(z)
            
            u, v = self.Om0 * zp1**3, self.Ode0
            x    = u + v
            if not self.flat:
                x += self.Ok0 * zp1**2
            u, v = u / x, v / x

            return 2.5 / zp1 * u * ( u**(4./7) - v + (1 + u/2) * (1 + v/70) )**(-1)

        def exact(z):

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

        if self.psmodel == 'eisenstien98_zb':
            return eisenstien98_zb(k, self.h, self.Om0, self.Ob0, self.Tcmb0)
        elif self.psmodel == 'sugiyama95':
            return sugiyama95(k, self.h, self.Om0, self.Ob0, self.Tcmb0)
        raise NotImplementedError(self.psmodel)

    def matterPowerSpectrum(self, k: Any, z: float = 0.0, dim: bool = True) -> Any:

        k = np.asfarray(k)
        y = self.A * k**self.ns * self.transfer(k)**2
        
        if not dim:
            y = y * k**3 / (2 * np.pi**2)
        
        if z > 0.0:
            y = y * self.Dplus(z)**2

        return y

    def variance(self, r: Any, z: float = 0.0, filter: str = 'tophat') -> Any:

        r = np.asfarray(r)

        ka, kb = 1e-06, 1e+06
        if filter == 'sharpk':
            kb = np.pi / r    

        lnka, lnkb = np.log(ka), np.log(kb)
        lnk, dlnk  = np.linspace(lnka, lnkb, 10001, retstep = True)

        k = np.exp(lnk)
        
        w = 1.0
        if filter == 'tophat':
            w = np.outer(r, k)
            w = (np.sin(w) - w * np.cos(w)) * 3. / w**3
        elif filter == 'gauss':
            w = np.outer(r, k)
            w = np.exp(-0.5 * w**2)
        elif filter != 'sharpk':
            raise NotImplementedError(filter)
        
        y = self.matterPowerSpectrum(k, z, False) * w**2
        
        if filter == 'sharpk':
            y = y.T

        y = simps(y, axis = -1) * dlnk

        if np.ndim(r):
            return y


        return y[0] if np.ndim(y) else y

    def normalizePower(self, filter: str = 'tophat'):

        self.A = 1.0

        var8   = self.variance(8, 0.0, filter)
        self.A = self.sigma8**2 / var8
        return self

    def effectiveIndex(self, k: Any, z: float = 0.0, h: float = 0.01) -> Any:

        k = np.asfarray(k)

        k1, k2 = (1 - h)*k, (1 + h)*k
        
        y1 = self.matterPowerSpectrum(k1, z, True)
        y2 = self.matterPowerSpectrum(k2, z, True)

        return (np.log(y2) - np.log(y1)) / (np.log(k2) - np.log(k1))


# ========================================================================================
# probability distribution object
# ========================================================================================

class Genextreme:

    def __init__(self, cellsize: float, cm: Cosmology, p: int = 1) -> None:
        self.cellsize = cellsize
        self.kn       = np.pi / cellsize
        self.cm       = cm
        self.p        = p

        self._makeMeasuredPowerSpline()
        self.param = None

    def linearVariance(self, z: float = 0.0):
        return self.cm.variance(self.cellsize, z, filter = 'sharpk')

    def logVariance(self, lin: float):
        mu = 0.73
        return mu * np.log( 1 + lin / mu )

    def _makeMeasuredPowerSpline(self):

        kn = self.kn
        p  = self.p

        def _power(k):
            return self.cm.matterPowerSpectrum(k)

        def _measuredPower(k1, k2, k3):

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

        kvals = np.logspace( np.log10(1e-6), np.log10(kn), 51 )
        pval, _, _ = binned_statistic( klen.flatten(), pk_meas.flatten(), statistic = 'mean', bins = kvals )
        kvals = 0.5 * ( kvals[1:] + kvals[:-1] )

        # power law continuation 
        alpha, beta = np.polyfit( np.log( kvals[ kvals > 0.5*kn ] ), np.log( pval[ kvals > 0.5*kn ] ), 1 )
        beta = np.exp(beta)

        kvals2 = np.logspace( np.log10(kn), np.log10( kn*np.sqrt(3) ), 3 )
        pval2  = beta * kvals2**alpha
        
        self.pspline = CubicSpline(
                                    np.log( np.hstack( [kvals, kvals2] ) ),
                                    np.log( np.hstack( [pval, pval2] ) )
                                  )
        return

    def measuredPower(self, k, z: float = 0.0, bias2: float = 1.0):
        pk0 = np.exp( self.pspline( np.log(k) ) )
        return bias2 * pk0 * self.cm.Dplus(z)**2

    def measuredVariance(self, z: float = 0.0, bias2: float = 1.0):
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
        lamda = 0.65
        return -lamda * np.log( 1 + 0.5 * lin / lamda )

    def skewness(self, meas, z: float = 0.0):
        nsp3 = self.cm.effectiveIndex(self.kn, z) + 3
        t = -0.7 * nsp3 + 1.25
        p = 0.06 - 0.26 * np.log(nsp3)
        return t * meas**(0.5 * p)

    def parametrize(self, z: float = 0.0):

        def gk(k, x):
            return gamma(1 - k*x)

        def f(x, gamma1):
            g1, g2, g3 = gk(1,x), gk(2,x), gk(3,x)
            return gamma1 + (g3 - 3*g1*g2 + 2*g1**3) / (g2 - g1**2)**1.5


        s2_lin = self.linearVariance(z)
        s2_a   = self.logVariance(s2_lin)
        
        bias2   = s2_a / s2_lin
        s2_meas = self.measuredVariance(z, bias2)

        abar   = self.mean(s2_lin)
        gamma1 = self.skewness(s2_meas, z)

        shape = newton( f, 0.01, args = (gamma1, ) )
        scale = abs(shape) * s2_meas**0.5 * ( gk(2, shape) - gk(1, shape)**2 )**(-0.5)
        loc   = abar - (gk(1, shape) - 1) * scale / shape

        self.param = shape, loc, scale
        return self

    def pdf(self, x):
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
