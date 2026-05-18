from numbers import Integral, Real

import lxml.etree as ET

import openmc.checkvalue as cv
from .filter import Filter
from ._xml import get_text


class ExpansionFilter(Filter):
    """Abstract filter class for functional expansions."""

    def __init__(self, order, filter_id=None):
        self.order = order
        self.id = filter_id

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        else:
            return hash(self) == hash(other)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        cv.check_type('expansion order', order, Integral)
        cv.check_greater_than('expansion order', order, 0, equality=True)
        self._order = order

    def to_xml_element(self):
        """Return XML Element representing the filter.

        Returns
        -------
        element : lxml.etree._Element
            XML element containing Legendre filter data

        """
        element = ET.Element('filter')
        element.set('id', str(self.id))
        element.set('type', self.short_name.lower())

        subelement = ET.SubElement(element, 'order')
        subelement.text = str(self.order)

        return element

    @classmethod
    def from_xml_element(cls, elem, **kwargs):
        filter_id = int(get_text(elem, "id"))
        order = int(get_text(elem, "order"))
        return cls(order, filter_id=filter_id)

    def merge(self, other):
        """Merge this filter with another.

        This overrides the behavior of the parent Filter class, since its
        merging technique is to take the union of the set of bins of each
        filter. That technique does not apply to expansion filters, since the
        argument should be the maximum filter order rather than the list of all
        bins.

        Parameters
        ----------
        other : openmc.Filter
            Filter to merge with

        Returns
        -------
        merged_filter : openmc.Filter
            Filter resulting from the merge

        """

        if not self.can_merge(other):
            msg = f'Unable to merge "{type(self)}" with "{type(other)}"'
            raise ValueError(msg)

        # Create a new filter with these bins and a new auto-generated ID
        return type(self)(max(self.order, other.order))


class LegendreFilter(ExpansionFilter):
    r"""Score Legendre expansion moments up to specified order.

    This filter allows scores to be multiplied by Legendre polynomials of the
    change in particle angle (:math:`\mu`) up to a user-specified order.

    Parameters
    ----------
    order : int
        Maximum Legendre polynomial order
    filter_id : int or None
        Unique identifier for the filter

    Attributes
    ----------
    order : int
        Maximum Legendre polynomial order
    id : int
        Unique identifier for the filter
    num_bins : int
        The number of filter bins

    """

    def __hash__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        return hash(string)

    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tID', self.id)
        return string

    @ExpansionFilter.order.setter
    def order(self, order):
        ExpansionFilter.order.__set__(self, order)
        self.bins = [f'P{i}' for i in range(order + 1)]

    @classmethod
    def from_hdf5(cls, group, **kwargs):
        if group['type'][()].decode() != cls.short_name.lower():
            raise ValueError("Expected HDF5 data for filter type '"
                             + cls.short_name.lower() + "' but got '"
                             + group['type'][()].decode() + " instead")

        filter_id = int(group.name.split('/')[-1].lstrip('filter '))

        out = cls(group['order'][()], filter_id)

        return out


class SpatialLegendreFilter(Filter):
    r"""Multidimensional Functional Expansion Tally (FET) filter using
    Legendre polynomials in up to three spatial dimensions.

    Each axis maps the particle coordinate to [-1, 1] and evaluates
    Legendre polynomials P_0 … P_n.  Bins are the tensor product of the
    active axes in x → y → z order; weights are the products of the
    per-axis Legendre values.  Works with both collision and tracklength
    estimators.

    Use :meth:`add_axis` to configure each dimension.  Axes are always
    stored in x → y → z order regardless of the order they are added.

    Parameters
    ----------
    filter_id : int or None
        Unique identifier for the filter.

    Examples
    --------
    1-D FET along x:

    >>> f = openmc.SpatialLegendreFilter()
    >>> f.add_axis('x', order=4, minimum=0.0, maximum=15.0)

    3-D FET:

    >>> f = openmc.SpatialLegendreFilter()
    >>> f.add_axis('x', order=10, minimum=0.0,  maximum=15.0)
    >>> f.add_axis('y', order=2,  minimum=-1.0, maximum=1.0)
    >>> f.add_axis('z', order=2,  minimum=-1.0, maximum=1.0)
    """

    _AXES = ('x', 'y', 'z')

    def __init__(self, filter_id=None):
        self._axes = []   # list of dicts, always sorted x -> y -> z
        self.id = filter_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_axis(self, axis, order, minimum, maximum):
        """Add a spatial axis to the FET expansion.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
        order : int
            Maximum Legendre polynomial order (>= 0).
        minimum : float
            Lower bound of the physical domain on this axis.
        maximum : float
            Upper bound of the physical domain on this axis.
        """
        cv.check_value('axis', axis, self._AXES)
        cv.check_type('order', order, Integral)
        cv.check_greater_than('order', order, 0, equality=True)
        cv.check_type('minimum', minimum, Real)
        cv.check_type('maximum', maximum, Real)
        if maximum <= minimum:
            raise ValueError(
                f"FET axis '{axis}': maximum must be greater than minimum.")
        if any(d['axis'] == axis for d in self._axes):
            raise ValueError(
                f"Axis '{axis}' has already been added to this filter.")

        self._axes.append(
            {'axis': axis, 'order': order, 'minimum': minimum, 'maximum': maximum})
        # Enforce x -> y -> z storage order.
        self._axes.sort(key=lambda d: self._AXES.index(d['axis']))
        self._update_bins()

    @property
    def axes(self):
        """List of active axis dicts in x -> y -> z order (read-only copy)."""
        return list(self._axes)

    @property
    def num_bins(self):
        total = 1
        for d in self._axes:
            total *= d['order'] + 1
        return total

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        lines = [type(self).__name__]
        for d in self._axes:
            lines.append(f"  {d['axis']}: order={d['order']}  "
                         f"[{d['minimum']}, {d['maximum']}]")
        lines.append(f'  id={self.id}  bins={self.num_bins}')
        return '\n'.join(lines)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return type(self) is type(other) and hash(self) == hash(other)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _update_bins(self):
        from itertools import product as iproduct
        ranges = [range(d['order'] + 1) for d in self._axes]
        self.bins = [
            ' '.join(f'P{i}({d["axis"]})' for d, i in zip(self._axes, combo))
            for combo in iproduct(*ranges)
        ]

    def to_xml_element(self):
        element = ET.Element('filter')
        element.set('id', str(self.id))
        element.set('type', self.short_name.lower())
        for d in self._axes:
            dim = ET.SubElement(element, d['axis'])
            ET.SubElement(dim, 'order').text = str(d['order'])
            ET.SubElement(dim, 'min').text   = str(d['minimum'])
            ET.SubElement(dim, 'max').text   = str(d['maximum'])
        return element

    @classmethod
    def from_xml_element(cls, elem, **kwargs):
        filt = cls(filter_id=int(get_text(elem, 'id')))
        for ax in ('x', 'y', 'z'):
            dim = elem.find(ax)
            if dim is not None:
                filt.add_axis(ax,
                              int(get_text(dim, 'order')),
                              float(get_text(dim, 'min')),
                              float(get_text(dim, 'max')))
        return filt

    @classmethod
    def from_hdf5(cls, group, **kwargs):
        if group['type'][()].decode() != cls.short_name.lower():
            raise ValueError(
                f"Expected filter type '{cls.short_name.lower()}' "
                f"but got '{group['type'][()].decode()}'")
        filter_id = int(group.name.split('/')[-1].lstrip('filter '))
        filt = cls(filter_id=filter_id)
        axes   = [a.decode() for a in group['axes'][()]]
        orders = group['orders'][()]
        mins   = group['mins'][()]
        maxs   = group['maxs'][()]
        for ax, order, mn, mx in zip(axes, orders, mins, maxs):
            filt.add_axis(ax, int(order), float(mn), float(mx))
        return filt


class SphericalHarmonicsFilter(ExpansionFilter):
    r"""Score spherical harmonic expansion moments up to specified order.

    This filter allows you to obtain real spherical harmonic moments of either
    the particle's direction or the cosine of the scattering angle. Specifying
    a filter with order :math:`\ell` tallies moments for all orders from 0 to
    :math:`\ell`.

    Parameters
    ----------
    order : int
        Maximum spherical harmonics order, :math:`\ell`
    filter_id : int or None
        Unique identifier for the filter

    Attributes
    ----------
    order : int
        Maximum spherical harmonics order, :math:`\ell`
    id : int
        Unique identifier for the filter
    cosine : {'scatter', 'particle'}
        How to handle the cosine term.
    num_bins : int
        The number of filter bins

    """

    def __init__(self, order, filter_id=None):
        super().__init__(order, filter_id)
        self._cosine = 'particle'

    def __hash__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tCosine', self.cosine)
        return hash(string)

    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tCosine', self.cosine)
        string += '{: <16}=\t{}\n'.format('\tID', self.id)
        return string

    @ExpansionFilter.order.setter
    def order(self, order):
        ExpansionFilter.order.__set__(self, order)
        self.bins = [f'Y{n},{m}'
                     for n in range(order + 1)
                     for m in range(-n, n + 1)]

    @property
    def cosine(self):
        return self._cosine

    @cosine.setter
    def cosine(self, cosine):
        cv.check_value('Spherical harmonics cosine treatment', cosine,
                       ('scatter', 'particle'))
        self._cosine = cosine

    @classmethod
    def from_hdf5(cls, group, **kwargs):
        if group['type'][()].decode() != cls.short_name.lower():
            raise ValueError("Expected HDF5 data for filter type '"
                             + cls.short_name.lower() + "' but got '"
                             + group['type'][()].decode() + " instead")

        filter_id = int(group.name.split('/')[-1].lstrip('filter '))

        out = cls(group['order'][()], filter_id)
        out.cosine = group['cosine'][()].decode()

        return out

    def to_xml_element(self):
        """Return XML Element representing the filter.

        Returns
        -------
        element : lxml.etree._Element
            XML element containing spherical harmonics filter data

        """
        element = super().to_xml_element()
        element.set('cosine', self.cosine)
        return element

    @classmethod
    def from_xml_element(cls, elem, **kwargs):
        filter_id = int(get_text(elem, "id"))
        order = int(get_text(elem, "order"))
        filter = cls(order, filter_id=filter_id)
        filter.cosine = get_text(elem, "cosine")
        return filter


class ZernikeFilter(ExpansionFilter):
    r"""Score Zernike expansion moments in space up to specified order.

    This filter allows scores to be multiplied by Zernike polynomials of the
    particle's position normalized to a given unit circle, up to a
    user-specified order. The standard Zernike polynomials follow the
    definition by Born and Wolf, *Principles of Optics* and are defined as

    .. math::
        Z_n^m(\rho, \theta) = R_n^m(\rho) \cos (m\theta), \quad m > 0

        Z_n^{m}(\rho, \theta) = R_n^{m}(\rho) \sin (m\theta), \quad m < 0

        Z_n^{m}(\rho, \theta) = R_n^{m}(\rho), \quad m = 0

    where the radial polynomials are

    .. math::
        R_n^m(\rho) = \sum\limits_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! (
        \frac{n+m}{2} - k)! (\frac{n-m}{2} - k)!} \rho^{n-2k}.

    With this definition, the integral of :math:`(Z_n^m)^2` over the unit disk
    is :math:`\frac{\epsilon_m\pi}{2n+2}` for each polynomial where
    :math:`\epsilon_m` is 2 if :math:`m` equals 0 and 1 otherwise.

    Specifying a filter with order N tallies moments for all :math:`n` from 0
    to N and each value of :math:`m`. The ordering of the Zernike polynomial
    moments follows the ANSI Z80.28 standard, where the one-dimensional index
    :math:`j` corresponds to the :math:`n` and :math:`m` by

    .. math::
        j = \frac{n(n + 2) + m}{2}.

    Parameters
    ----------
    order : int
        Maximum Zernike polynomial order
    x : float
        x-coordinate of center of circle for normalization
    y : float
        y-coordinate of center of circle for normalization
    r : float
        Radius of circle for normalization

    Attributes
    ----------
    order : int
        Maximum Zernike polynomial order
    x : float
        x-coordinate of center of circle for normalization
    y : float
        y-coordinate of center of circle for normalization
    r : float
        Radius of circle for normalization
    id : int
        Unique identifier for the filter
    num_bins : int
        The number of filter bins

    """

    def __init__(self, order, x=0.0, y=0.0, r=1.0, filter_id=None):
        super().__init__(order, filter_id)
        self.x = x
        self.y = y
        self.r = r

    def __hash__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tX', self.x)
        string += '{: <16}=\t{}\n'.format('\tY', self.y)
        string += '{: <16}=\t{}\n'.format('\tR', self.r)
        return hash(string)

    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tID', self.id)
        return string

    @ExpansionFilter.order.setter
    def order(self, order):
        ExpansionFilter.order.__set__(self, order)
        self.bins = [f'Z{n},{m}'
                     for n in range(order + 1)
                     for m in range(-n, n + 1, 2)]

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        cv.check_type('x', x, Real)
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        cv.check_type('y', y, Real)
        self._y = y

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        cv.check_type('r', r, Real)
        self._r = r

    @classmethod
    def from_hdf5(cls, group, **kwargs):
        if group['type'][()].decode() != cls.short_name.lower():
            raise ValueError("Expected HDF5 data for filter type '"
                             + cls.short_name.lower() + "' but got '"
                             + group['type'][()].decode() + " instead")

        filter_id = int(group.name.split('/')[-1].lstrip('filter '))
        order = group['order'][()]
        x, y, r = group['x'][()], group['y'][()], group['r'][()]

        return cls(order, x, y, r, filter_id)

    def to_xml_element(self):
        """Return XML Element representing the filter.

        Returns
        -------
        element : lxml.etree._Element
            XML element containing Zernike filter data

        """
        element = super().to_xml_element()
        subelement = ET.SubElement(element, 'x')
        subelement.text = str(self.x)
        subelement = ET.SubElement(element, 'y')
        subelement.text = str(self.y)
        subelement = ET.SubElement(element, 'r')
        subelement.text = str(self.r)

        return element

    @classmethod
    def from_xml_element(cls, elem, **kwargs):
        filter_id = int(get_text(elem, "id"))
        order = int(get_text(elem, "order"))
        x = float(get_text(elem, "x"))
        y = float(get_text(elem, "y"))
        r = float(get_text(elem, "r"))
        return cls(order, x, y, r, filter_id=filter_id)


class ZernikeRadialFilter(ZernikeFilter):
    r"""Score the :math:`m = 0` (radial variation only) Zernike moments up to
    specified order.

    The Zernike polynomials are defined the same as in :class:`ZernikeFilter`.

    .. math::

        Z_n^{0}(\rho, \theta) = R_n^{0}(\rho)

    where the radial polynomials are

    .. math::
        R_n^{0}(\rho) = \sum\limits_{k=0}^{n/2} \frac{(-1)^k (n-k)!}{k! ((
        \frac{n}{2} - k)!)^{2}} \rho^{n-2k}.

    With this definition, the integral of :math:`(Z_n^0)^2` over the unit disk
    is :math:`\frac{\pi}{n+1}`.

    If there is only radial dependency, the polynomials are integrated over
    the azimuthal angles. The only terms left are :math:`Z_n^{0}(\rho, \theta)
    = R_n^{0}(\rho)`. Note that :math:`n` could only be even orders.
    Therefore, for a radial Zernike polynomials up to order of :math:`n`,
    there are :math:`\frac{n}{2} + 1` terms in total. The indexing is from the
    lowest even order (0) to highest even order.

    Parameters
    ----------
    order : int
        Maximum radial Zernike polynomial order
    x : float
        x-coordinate of center of circle for normalization
    y : float
        y-coordinate of center of circle for normalization
    r : float
        Radius of circle for normalization

    Attributes
    ----------
    order : int
        Maximum radial Zernike polynomial order
    x : float
        x-coordinate of center of circle for normalization
    y : float
        y-coordinate of center of circle for normalization
    r : float
        Radius of circle for normalization
    id : int
        Unique identifier for the filter
    num_bins : int
        The number of filter bins

    """

    @ExpansionFilter.order.setter
    def order(self, order):
        ExpansionFilter.order.__set__(self, order)
        self.bins = [f'Z{n},0' for n in range(0, order+1, 2)]