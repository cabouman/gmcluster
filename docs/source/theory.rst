======
Theory
======

This theory was originally published in Appendix A of :cite:`bouman97`.

The model
---------

A Gaussian mixture models a distribution made of distinct subclasses, or clusters.
For example, an image pixel behaves differently on an edge than in a smooth region,
so its aggregate behavior is a mixture of the two.  Each cluster is one Gaussian,
described by a mean and a covariance.

Let :math:`Y` be an :math:`M` dimensional random vector modeled by a mixture with
:math:`K` clusters.  Cluster :math:`k` is specified by

* :math:`\pi_k` — the probability that a sample belongs to cluster :math:`k`,
* :math:`\mu_k` — the :math:`M` dimensional mean vector,
* :math:`R_k` — the :math:`M \times M` covariance matrix.

Write :math:`\pi`, :math:`\mu`, and :math:`R` for the sets
:math:`\{\pi_k\}_{k=1}^K`, :math:`\{\mu_k\}_{k=1}^K`, and :math:`\{R_k\}_{k=1}^K`.
The full parameter set is :math:`K` and :math:`\theta = (\pi, \mu, R)`, subject to
:math:`K` a positive integer, :math:`\pi_k \geq 0` with :math:`\sum_k \pi_k = 1`,
and :math:`\det(R_k) \geq \epsilon`.  Write :math:`\Omega^{(K)}` for the admissible
:math:`\theta` at order :math:`K`.

Let :math:`Y_1, \dots, Y_N` be :math:`N` samples, and let :math:`X_n` be the
(unknown) cluster of sample :math:`Y_n`.  Given :math:`X_n = k`, the density of
:math:`Y_n` is Gaussian:

.. math::
	p_{y_n|x_n} (y_n|k, \theta) = \frac{1}{(2\pi)^{M/2}} |R_k|^{-1/2} \exp\bigg\{- \frac{1}{2} {(y_n - \mu_k)}^t R^{-1}_k {(y_n - \mu_k)}\bigg\}

Since :math:`X_n` is unknown, the density of :math:`Y_n` sums over the clusters:

.. math::
	p_{y_n} (y_n|\theta) = \sum_{k=1}^K p_{y_n|x_n} (y_n|k, \theta)\pi_k

and the log probability of the whole sequence :math:`Y = \{Y_n\}_{n=1}^N` is

.. math::
	\log p_y (y|K, \theta) = \sum_{n=1}^N \log\bigg(\sum_{k=1}^K p_{y_n|x_n} (y_n|k, \theta)\pi_k\bigg)

Why not maximum likelihood
--------------------------

The maximum likelihood (ML) estimate of :math:`\theta` at fixed :math:`K` is

.. math::
	\hat\theta_{ML} = \arg \max_{\theta \in \Omega^{(K)}}\log p_y(y|K, \theta)

But the ML estimate of :math:`K` is not well defined: adding clusters always fits
the data better, so the likelihood keeps rising with :math:`K`.  Estimating
:math:`K` is an order-identification problem, and it needs a penalty on model size.

The AIC criterion :cite:`akaike` adds such a penalty,

.. math::
	AIC(K, \theta) = -2 \log p_y(y|K, \theta) + 2L

where :math:`L` is the number of real parameters in :math:`\theta`,

.. math::
	L = K\bigg(1 + M + \frac{(M + 1)M}{2}\bigg) - 1.

AIC is not a consistent estimator :cite:`kashyap`: as :math:`N \to \infty`, its
estimate of :math:`K` does not converge to the true value.

The MDL criterion
-----------------

Rissanen's minimum description length (MDL) criterion :cite:`rissanen` chooses the
order that codes both the data and the parameters in the fewest bits:

.. math::
	MDL(K, \theta) = - \log p_y(y|K, \theta) + \frac{1}{2} L \log(NM)

The one difference from AIC is that the penalty depends on the data size
:math:`NM`, which keeps more data from causing overfitting.  MDL is a consistent
estimator of order for a broad class of problems :cite:`kashyap_2` :cite:`wax`.
(Mixture models are not proven to be in that class, because the solution lies on
the boundary of the constraint set; a consistent but costly alternative is
:cite:`aitkin`.  See :cite:`render` for EM convergence proofs.)

The objective is to minimize

.. math::
	MDL(K, \theta) = -\sum_{n=1}^N \log\bigg(\sum_{k=1}^K p_{y_n|x_n} (y_n|k, \theta)\pi_k\bigg) + \frac{1}{2} L \log(NM)

Two things make this hard: the log-of-sum blocks direct optimization over
:math:`\pi, \mu, R`, and each value of :math:`K` needs a full re-optimization.  The
EM algorithm handles the first problem :cite:`baum` :cite:`dempster`; the merging
step below handles the second.

EM updates at fixed order
-------------------------

EM assigns each point a soft membership to every cluster, then re-estimates the
cluster parameters from those memberships.  Starting from an estimate
:math:`\theta^{(i)}`, the membership of point :math:`y_n` in cluster :math:`k` comes
from Bayes rule:

.. math::
	p_{x_n|y_n} (k|y_n, \theta^{(i)}) = \frac{p_{y_n|x_n} (y_n|k, \theta^{(i)}) \pi_k}{\sum_{l=1}^K p_{y_n|x_n} (y_n|l, \theta^{(i)})\pi_l}

The new weight, mean, and covariance estimates :math:`\bar\pi_k, \bar\mu_k, \bar R_k`
are the membership-weighted averages:

.. math::
	& \bar N_k = \sum_{n=1}^N p_{x_n|y_n} (k|y_n, \theta^{(i)})\\
	& \bar\pi_k = \frac{\bar N_k}{N}\\
	& \bar\mu_k = \frac{1}{\bar N_k}\sum_{n=1}^N y_n p_{x_n|y_n} (k|y_n, \theta^{(i)})\\
	& \bar R_k = \frac{1}{\bar N_k}\sum_{n=1}^N (y_n - \bar\mu_k){(y_n - \bar\mu_k)}^t p_{x_n|y_n} (k|y_n, \theta^{(i)})

To derive these formally, define

.. math::
	Q(\theta; \theta^{(i)}) = E[\log p_{y,x}(y, X|\theta)|Y=y,\theta^{(i)}]- \frac{1}{2} L \log(NM)

where :math:`Y` and :math:`X` are the sets :math:`\{Y_n\}` and :math:`\{X_n\}`.
The key EM result :cite:`baum` is that for all :math:`\theta`

.. math::
	MDL(K, \theta) - MDL(K, \theta^{(i)}) < Q(\theta^{(i)}; \theta^{(i)}) - Q(\theta; \theta^{(i)})

so any :math:`\theta` that increases :math:`Q` is guaranteed to reduce MDL.  EM
iterates this until it reaches a local minimum of MDL.  Substituting for
:math:`\log p_{y,x}` and simplifying gives an explicit form for :math:`Q`:

.. math::
	Q(\theta; \theta^{(i)}) = \sum_{k=1}^K \bar N_k \bigg\{ -\frac{1}{2} trace[\bar R_k R_k^{-1}] -\frac{1}{2} {(\bar\mu_k-\mu_k)}^t R_k^{-1}(\bar\mu_k-\mu_k)\\
	-\frac{M}{2}\log(2\pi) -\frac{1}{2}\log(|R_k|) + \log(\pi_k) \bigg\}- \frac{1}{2} L \log(NM)

Maximizing :math:`Q` over :math:`\theta \in \Omega^{(K)}` with Lagrange multipliers
gives the update

.. math::
	(\pi^{(i+1)}, \mu^{(i+1)}, R^{(i+1)}) & = \arg \max_{(\pi,\mu,R) \in \Omega^{(K)}} Q(\theta; \theta^{(i)}) \\
	& = (\bar\pi, \bar\mu, \bar R)

Reducing the order by merging
-----------------------------

To search over :math:`K`, start with many clusters and step :math:`K` down by one at
a time, running EM to convergence at each order and keeping the order with the
smallest MDL.

One order is reduced by merging two clusters :math:`l` and :math:`m` into one:
constrain their means and covariances to be equal,

.. math::
	& \mu_l = \mu_m = \mu_{(l,m)} \\
	& R_l = R_m = R_{(l,m)}

Write :math:`\theta_{(l,m)} \in \Omega^{(K)}` for this constrained :math:`K`-cluster
parameter (with clusters :math:`l` and :math:`m` identical), and
:math:`\theta_{(l,m)-} \in \Omega^{(K-1)}` for the :math:`K-1` distinct clusters,
where the merged weight is

.. math::
	\pi_{(l,m)} = \pi_l + \pi_m

Dropping to :math:`K-1` distinct clusters changes MDL by a fixed penalty term:

.. math::
	MDL(K - 1, \theta_{(l,m)-}) = MDL(K, \theta_{(l,m)}) + \frac{1}{2}\bigg (1 + M + \frac{(M+1)M}{2}\bigg ) \log(NM)

Expanding the total change and using the EM bound gives

.. math::
	& MDL(K - 1, \theta_{(l,m)-}) - MDL(K, \theta^{(i)}) \\
	& = MDL(K - 1, \theta_{(l,m)-}) - MDL(K, \theta_{(l,m)}) + MDL(K, \theta_{(l,m)}) - MDL(K, \theta^{(i)}) \\
	& \leq -\frac{1}{2} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM) + Q(\theta^{(i)}; \theta^{(i)}) - Q(\theta_{(l,m)}; \theta^{(i)}) \\
	& \leq -\frac{1}{2} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM) \\
	& + Q(\theta^{(i)}; \theta^{(i)}) - Q(\theta^*; \theta^{(i)}) + Q(\theta^*; \theta^{(i)}) - Q(\theta_{(l,m)}^*; \theta^{(i)})

where :math:`\theta^*` and :math:`\theta_{(l,m)}^*` are the unconstrained and
constrained optima.  If EM has already converged at order :math:`K`, then
:math:`\theta^* = \theta^{(i)}` and

.. math::
	Q(\theta^{(i)}; \theta^{(i)}) - Q(\theta^*; \theta^{(i)}) = 0

Maximizing :math:`Q` subject to the merge constraint keeps
:math:`\pi_l^* = \bar\pi_l` and :math:`\pi_m^* = \bar\pi_m`, and gives the merged
mean and covariance

.. math::
	& \mu_{(l,m)}^* = \frac{\bar\pi_l\bar\mu_l + \bar\pi_m \bar\mu_m}{\bar\pi_l + \bar\pi_m}\\
	& R_{(l,m)}^* = \frac{\bar\pi_l (\bar R_l + (\bar\mu_l-\mu_{(l,m)}){(\bar\mu_l-\mu_{(l,m)})}^t) + \bar\pi_m (\bar R_m + (\bar\mu_m-\mu_{(l,m)}){(\bar\mu_m-\mu_{(l,m)})}^t)}{\bar\pi_l + \bar\pi_m}

Choosing which pair to merge
----------------------------

Define the merge distance

.. math::
	d(l,m) & = Q(\theta^*; \theta^{(i)})-Q(\theta_{(l,m)}^*; \theta^{(i)}) \\
	& = \frac{N\bar\pi_l}{2} \log\bigg( \frac{|R_{(l,m)}|}{|\bar R_l|} \bigg) + \frac{N\bar\pi_m}{2} \log\bigg( \frac{|R_{(l,m)}|}{|\bar R_m|} \bigg)

which upper-bounds the change in MDL:

.. math::
	MDL(K - 1, \theta_{(l,m)-}) - MDL(K, \theta^{(i)}) \leq d(l,m) - \frac{1}{2} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM)

The value :math:`d(l, m)` is always positive: fewer parameters can only lower the
log likelihood.  The penalty term that offsets it does not depend on :math:`l` and
:math:`m`, so the pair to merge is the one that minimizes :math:`d(l, m)`:

.. math::
	(l^*, m^*) = \arg \min_{(l,m)} d(l,m)

Those two clusters are merged, and the result :math:`\theta_{(l,m)}^*` becomes the
initial condition for EM at order :math:`K - 1`.

Initialization and the algorithm
--------------------------------

EM only reaches a local minimum, so the starting point matters.  The user picks the
initial order :math:`K_0` subject to :math:`L < \frac{1}{2}MN`, and the initial
parameters are

.. math::
	& \pi_k^{(1)} = \frac{1}{K_0} \\
	& \mu_k^{(1)} = y_n \text{ where } n = \lfloor (K-1)(N-1)/(K_0-1) \rfloor +1 \\
	& R_k^{(1)} = \frac{1}{N} \sum_{n=1}^N y_n y_n^t

where :math:`\lfloor \cdot \rfloor` is the floor function.  The full algorithm is:

1. Initialize with a large number of clusters :math:`K_0`.
2. Initialize :math:`\theta^{(1)}`.
3. Run EM until the change in :math:`MDL(K, \theta)` is less than :math:`\epsilon`.
4. Record :math:`\theta^{(K,i_{final})}` and :math:`MDL(K, \theta^{(K,i_{final})})`.
5. If more than one cluster remains, merge a pair, set :math:`K \leftarrow K - 1`,
   and return to step 3.
6. Choose the :math:`K^*` and :math:`\theta^{(K^*,i_{final})}` that minimize MDL.

In step 3 the tolerance is

.. math::
	\epsilon = \frac{1}{100} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM)

**References**

.. bibliography:: bibtex/ref.bib
   :style: unsrt
   :labelprefix: A
   :all:
