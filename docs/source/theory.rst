======
Theory 
======

It is often desirable to model distributions that are composed of distinct subclasses or clusters. For example, a pixel in a image might behave differently if it comes from an edge rather than a smooth region. Therefore the aggregate behavior is likely to be a mixture of the two distinct behaviors. The objective of mixture distributions is to form a probabilistic model composed of a number of component subclasses. Each subclass is than characterized by a set of parameters describing the mean and variation of the spectral components. 

In order to estimate the parameters of a Gaussian mixture, it is necessary to determine the number of subclasses and the parameters of each subclasses. This can be done by using a representative sample of training data and estimating the number of subclasses and their parameters from this data. 

Specifically, let :math:`Y` be an :math:`M` dimensional random vector to be modeled using a Gaussian mixture distribution. Let us assume that this model has :math:`K` subclasses. Then the following parameters are required to completely specify the :math:`k^{th}` subclass.

:math:`\pi_k` - the probability that a pixel has subclass :math:`k`.

:math:`\mu_k` - the :math:`M` dimensional spectral mean vector for subclass :math:`k`.

:math:`R_k` - the :math:`M \times M` spectral covariance matrix for subclass :math:`k`.

Furthermore, let K denote the number of subclasses, then we use the notation :math:`\pi`, :math:`\mu`, and :math:`R` to denote the parameter sets :math:`{\{\pi_k\}}_{k=1}^K`, :math:`{\{\mu_k\}}_{k=1}^K`, and :math:`{\{R_k\}}_{k=1}^K`. The complete set of parameters for the information class are then given by :math:`K` and :math:`\theta = (\pi, \mu, R)`. Notice that the parameters are constrained in a variety of ways. In particular, :math:`K` must be an integer greater than :math:`0, \pi_k ≥ 0` with :math:`\sum_k \pi_k = 1`, and :math:`det(R) ≥ \epsilon` where :math:`\epsilon` may be chosen depending on the application. We will denote the set of admissible :math:`\theta` for a :math:`K^{th}` order model by :math:`\Omega^{(K)}`.

Let :math:`Y_1, Y_2, · · · , Y_N` be :math:`N` multispectral pixels sampled from the information class of interest. Furthermore, assume that for each pixel :math:`Y_i` the subclass of that pixel is given by the random variable :math:`X_n`. Of course, :math:`X_n` is usually not known, but it will be useful for analyzing the problem. Then assuming that each subclass has a multivariate Gaussian distribution, the probability density function for the pixel :math:`Y_n` given that :math:`X_n = k` is given by

.. math::
	p_{y_n|x_n} (y_n|k, \theta) = \frac{1}{(2\pi)^{M/2}} |R_k|^{−1/2} \exp\bigg\{− \frac{1}{2} {(y_n − \mu_k)}^t R^{−1}_k {(y_n − \mu_k)}^t\bigg\}


However, we do not know the subclass :math:`X_n` of each sample, so to compute the density function of :math:`Y_n` given the parameter :math:`\theta` we must apply the definition of conditional probability and sum over :math:`k`.

.. math::
	p_{y_n} (y_n|\theta) = \sum_{k=1}^K p_{y_n|x_n} (y_n|k, \theta)\pi_k
	
	
The log of the probability of the entire sequence :math:`Y = {\{Y_n\}}_{n=1}^N` is then given by

.. math::
	\log p_y (y|K, \theta) = \sum_{n=1}^N \log\bigg(\sum_{k=1}^K p_{y_n|x_n} (y_n|k, \theta)\pi_k\bigg)
	

The objective is then to estimate the parameters :math:`K` and :math:`\theta \in \Omega^{(K)}`. The maximum likelihood (ML) estimate is a commonly used estimate with many desirable properties. It is given by 

.. math::
	\hat\theta_{ML} = \arg \max_{\theta \in \Omega^{(K)}}\log p_y(y|K, \theta)
	
	
Unfortunately, the ML estimate of K is not well defined because the likelihood may always be made better by choosing a large number of subclusters. Intuitively, the log likelihood may always be increased by adding more subclasses since more subclasses may be used to more accurately fit the data.

This problem of estimating the order of a model is known as order identification, and has been studied by a variety of researchers. Methods for estimating model order generally tend to require the addition of a penalty term in the log likelihood to account for the over-fitting of high order models. One of the earliest approaches to order identification was suggested
by Akaike :cite:`akaike`, and requires the minimization of the so called AIC information criteria. The AIC criterion is given by

.. math::
	AIC(K, \theta) = −2 \log p_y(y|K, \theta) + 2L
	
	
where :math:`L` is the number of continuously valued real numbers required to specify the parameter :math:`\theta`. In this application,

.. math::
	L = K\bigg(1 + M + \frac{(M + 1)M}{2}\bigg)− 1.
	
However, an important disadvantage of the AIC criteria for a number of problems is that the AIC does not lead to a consistent estimator :cite:`kashyap`. This means that as the number of observations tends to infinity, the estimated value for :math:`K` does not converge to the true value. 

Alternatively, another criterion was suggested by Rissanen :cite:`rissanen` called the minimum description length (MDL) estimator. This estimator works by attempting to find the model order which minimizes the number of bits that would be required to code both the data samples :math:`y_n` and the parameter vector :math:`\theta`. While a direct implementation of the MDL estimator may depend on the particular coding method used, Rissanen develop an approximate expression for the estimate based on some assumptions and the minimization of the expression

.. math::
	MDL(K, \theta) = − \log p_y(y|K, \theta) + \frac{1}{2} L \log(NM)
	
	
Notice that the major difference between the AIC and MDL criteria is the dependence of the penalty term on the total number of data values :math:`NM`. In practice, this is important since otherwise more data will tend to result in over fitting of the model. In fact, it has been shown that for a large number of problems, the MDL criteria is a consistent estimator of model order :cite:`kashyap_2`:cite:`wax`. Unfortunately, the estimation of model order for mixture models does not fall into the class of problems for which the MDL criteria is known to be consistent. This is due to the fact that the solution to the mixture model problem always falls on a boundary of the constraint space, so the normal results on the asymptotic distribution of the ML estimate are no longer valid. An alternative method for order identification which is known to be consistent for mixture models is presented in :cite:`aitkin`. However, this method is computationally expensive when the dimensionality of the data is high. Also see :cite:`render` for detailed proofs of convergence for the EM algorithm.

Our objective will be to minimize the MDL criterion given by

.. math::
	MDL(K, \theta) = -\sum_{n=1}^N \log\bigg(\sum_{k=1}^K p_{y_n|x_n} (y_n|k, \theta)\pi_k\bigg) + \frac{1}{2} L \log(NM)
	
	
Direct minimization of :math:`MDL(\theta)` is difficult for a number of reasons. First, the logarithm term makes direct optimization with :math:`\pi, \mu`, and :math:`R` difficult. Second, minimization with respect to :math:`K` is complex since for each value of :math:`K` a complete minimization with respect to :math:`\pi, \mu,` and :math:`R` is required. If the subclass of each pixel, :math:`X_n`, where known, then the estimation of :math:`\pi, \mu,` and :math:`R` would be quite simple. Unfortunately, :math:`X_n` is not available. However, the expectation-maximization (EM) algorithm has been developed to address exactly this type of “incomplete” data problem :cite:`baum` :cite:`dempster`.


Intuitively, the EM algorithm works by first classifying the pixels Yn according to their subclass, and then re-estimating the subclass parameters based on this approximate classification. An essential point is that instead of the membership to each subclass being deterministic, the membership is represented using a “soft” probability. The process is started by assuming the the true parameter is given by :math:`\theta^{(i)}`. We index :math:`\theta^{(i)}` by :math:`i` because ultimately the EM algorithm will result in a iterative procedure for improving the MDL criterion. The probability that pixel yn belongs to subclass k may then be computed using Bayes rule.

.. math::
	p_{x_n|y_n} (k|y_n, \theta^{(i)}) = \frac{p_{y_n|x_n} (y_n|k, \theta^{(i)}) \pi_k}{\sum_{l=1}^K p_{y_n|x_n} (y_n|l, \theta^{(i)})\pi_l}


Then using these “soft” subclass memberships we will then compute new spectral mean and covariance estimates for each subclass. We will denote these new estimates by :math:`\bar\pi_k, \bar\mu_k` and :math:`\bar R_k` where

.. math::
	& \bar N_k = \sum_{n=1}^N p_{x_n|y_n} (k|y_n, \theta^{(i)})\\
	& \bar\pi_k = \frac{\bar N_k}{N}\\
	& \bar\mu_k = \frac{1}{\bar N_k}\sum_{n=1}^N y_n p_{x_n|y_n} (k|y_n, \theta^{(i)})\\
	& \bar R_k = \frac{1}{\bar N_k}\sum_{n=1}^N (y_n − \bar\mu_k){(y_n − \bar\mu_k)}^t p_{x_n|y_n} (k|y_n, \theta^{(i)})
	

In order to formally derive the EM algorithm update equations, we must first compute the following function

.. math::
	Q(\theta; \theta^{(i)}) = E[\log p_{y,x}(y, X|\theta)|Y=y,\theta^{(i)}]- \frac{1}{2} L \log(NM)
	
where :math:`Y` and :math:`X` are the sets of random variables :math:`{\{Y_n\}}_{n=1}^N` and :math:`{\{X_n\}}_{n=1}^N` respectively, and :math:`y` and :math:`x` are realizations of these random objects. The fundamental result of the EM algorithm which is proven in :cite:`baum` is that for all :math:`\theta`

.. math::
	MDL(K, \theta) − MDL(K, \theta^{(i)}) < Q(\theta^{(i)}; \theta^{(i)}) − Q(\theta; \theta^{(i)})
	
This results in a useful optimization method since any value of :math:`\theta` that increases the value of :math:`Q(\theta; \theta^{(i)})` is guarrenteed to reduce the MDL criteria. The objective of the EM algorithm is therefore to iteratively optimize with respect to :math:`\theta` until a local minimum of the MDL function is reached.

In order to derive expressions for the EM updates, we first compute a more explicit form for the function :math:`Q(\theta; \theta^{(i)})`. The Q function may be expressed in the following form by
substituting in for :math:`\log p_{y,x}(y, x|\theta)` and simplifying.

.. math::
	Q(\theta; \theta^{(i)}) = \sum_{k=1}^K \bar N_k \bigg\{ -\frac{1}{2} trace[\bar R_k R_k^{-1}] -\frac{1}{2} {(\bar\mu_k-\mu_k)}^t R_k^{-1}(\bar\mu_k-\mu_k)\\ 
	-\frac{M}{2}\log(2\pi) -\frac{1}{2}\log(|R_k|) + \log(\pi_k) \bigg\}- \frac{1}{2} L \log(NM)
	
where :math:`\bar N_k, \bar\mu_k,` and :math:`\bar R_k` are as given above.
	
We will first consider the maximization of :math:`Q(\theta; \theta^{(i)})` with respect to :math:`\theta \in \Omega^{(K)}`. This maximization of :math:`Q` may be done using Lagrange multipliers and results in the update equations

.. math::
	(\pi^{(i+1)}, \mu^{(i+1)}, R^{(i+1)}) & = \arg \max_{(\pi,\mu,R) \in \Omega^{(K)}} Q(\theta; \theta^{(i)}) \\
	& = (\bar\pi, \bar\mu, \bar R)

where :math:`(\bar\pi, \bar\mu, \bar R)` may be computed using above equations.

While the last equation shows how to update the parameter :math:`\theta`, it does not show how to change the model order :math:`K`. Our approach will be to start with a large number of clusters, and then sequentially decrement the value of :math:`K`. For each value of :math:`K`, we will apply the EM update until we converge to a local minimum of the MDL functional. After we have done this for each value of :math:`K`, we may simply select the value of :math:`K` and corresponding parameters that resulted in the smallest value of the MDL criteria.

The question remains of how to decrement the number of clusters from :math:`K` to :math:`K − 1`. We will do this by merging two clusters to form a single cluster. One way to effectively reduce the order of a model is to constrain the parameters of two subclasses to be equal. For example, two subclasses, :math:`l` and :math:`m`, may be effectively “merged” in a single subclass by constraining their mean and covariance parameters to be equal.

.. math::
	& \mu_l = \mu_m = \mu_{(l,m)} \\
	& R_l = R_m = R_{(l,m)}
	
	
Here :math:`\mu_{(l,m)}` and :math:`R_{(l,m)}` denote the mean and covariance of the new subclass, and we assume that the values of :math:`\pi_l` and :math:`\pi_m` remain unchanged for the two clusters being merged. We denote this modified parameter vector by :math:`\theta_{(l,m)} \in \Omega^{(K)}`. Notice that since :math:`theta_{(l,m)}` specifies the parameters for :math:`K` clusters, it is a member of :math:`\Omega^{(K)}`, but that two of these clusters (e.g. clusters :math:`l` and :math:`m`) have identical cluster means and covariance. Alternatively, we use the notation :math:`\theta_{(l,m)−} \in \Omega^{(K−1)}` to denote the parameters for the :math:`K − 1` distinct clusters in :math:`\theta_{(l,m)}`. More specifically, the two clusters :math:`l` and :math:`m` are specified as a single cluster :math:`(l, m)` with mean and covariance as given above, and prior probability given by

.. math::
	\pi_{(l,m)} = \pi_l + \pi_m
	

Using these definitions for :math:`\theta_{(l,m)}` and :math:`\theta_{(l,m)−}`, then the following relationship is resulted. 

.. math::
	MDL(K − 1, \theta_{(l,m)−}) = MDL(K, \theta_{(l,m)}) + \frac{1}{2}\bigg (1 + M + \frac{(M+1)M}{2}\bigg ) \log(NM)
	
The change in the MDL criteria is then given by

.. math::
	& MDL(K − 1, \theta_{(l,m)−}) - MDL(K, \theta^{(i)}) \\
	& = MDL(K − 1, \theta_{(l,m)−}) - MDL(K, \theta_{(l,m)}) + MDL(K, \theta_{(l,m)}) - MDL(K, \theta^{(i)}) \\
	& ≤ -\frac{1}{2} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM) + Q(\theta^{(i)}; \theta^{(i)}) − Q(\theta_{(l,m)}; \theta^{(i)}) \\
	& ≤ -\frac{1}{2} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM) \\
	& + Q(\theta^{(i)}; \theta^{(i)}) − Q(\theta^*; \theta^{(i)}) + Q(\theta^*; \theta^{(i)}) − Q(\theta_{(l,m)}^*; \theta^{(i)})
	
where :math:`\theta^*` and :math:`\theta_{(l,m)}^*` are the unconstrained and constrained optima respectively. The solution
to the unconstrained optimization, :math:`\theta^*`, is given above. We will assume that the EM algorithm has been run to convergence for a fixed order :math:`K`, so that :math:`\theta^* = \theta^{(i)}`. In this case,

.. math::
	Q(\theta^{(i)}; \theta^{(i)}) − Q(\theta^*; \theta^{(i)}) = 0
	
The value of :math:`\theta_{(l,m)}^*` is obtained by maximizing :math:`Q(\theta^*; \theta^{(i)})` as a function of :math:`\theta_{(l,m)}` subject to constraints. This constrained optimization results in the same values of :math:`\pi_l^* = \bar\pi_l` and :math:`\pi_m^* = \bar\pi_m` as in the unconstrained case, but the following new mean and covariance values.

.. math::
	& \mu_{(l,m)}^* = \frac{\bar\pi_l\bar\mu_l + \bar\pi_m \bar\mu_m}{\bar\pi_l + \bar\pi_m}\\
	& R_{(l,m)}^* = \frac{\bar\pi_l (\bar R_l + (\bar\mu_l-\mu_{(l,m)}){(\bar\mu_l-\mu_{(l,m)})}^t) + \bar\pi_m (\bar R_m + (\bar\mu_m-\mu_{(l,m)}){(\bar\mu_m-\mu_{(l,m)})}^t)}{\bar\pi_l + \bar\pi_m}
	
Here the :math:`\bar\pi, \bar\mu`, and :math:`\bar R` are given by the above equations, and the remaining values of :math:`n_k, \mu_k`, and :math:`R_k` are unchanged from the unconstrained result. We may define a distance function with the form

.. math::
	d(l,m) & = Q(\theta^*; \theta^{(i)})-Q(\theta_{(l,m)}^*; \theta^{(i)}) \\
	& = N\bar\pi_l\bigg\{ -\frac{M}{2}(1+\log(2\pi)) - \frac{1}{2}\log(|\bar R_l|) \bigg\} \\
	& + N\bar\pi_m\bigg\{ -\frac{M}{2}(1+\log(2\pi)) - \frac{1}{2}\log(|\bar R_m|) \bigg\} \\
	& - N\pi_{(l,m)}\bigg\{ -\frac{M}{2}(1+\log(2\pi)) - \frac{1}{2}\log(|R_{(l,m)}|) \bigg\} \\
	& = \frac{N\bar\pi_l}{2} \log\bigg( \frac{|R_{(l,m)}|}{|\bar R_l|} \bigg) + \frac{N\bar\pi_m}{2} \log\bigg( \frac{|R_{(l,m)}|}{|\bar R_m|} \bigg)
	
This distance function then serves as an upper bound on the change in the MDL criteria.

.. math::
	MDL(K − 1, \theta_{(l,m)−}) - MDL(K, \theta^{(i)}) ≤ d(l,m) - \frac{1}{2} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM)
	
A few comments are in order. The value of :math:`d(l, m)` is always positive. This is clear from the above form. In fact, reducing the model order should only reduce the log likelihood of the observations since there are fewer parameters to fit the data. In general, this increase may be offset by the model order term which is always negative. However, since this term is independent of the choice of :math:`l` and :math:`m`, it does not play a role in selecting which clusters to merge.

With the function :math:`d(l, m)` precisely defined, it is now possible to search over the set of all pairs, :math:`(l, m)`, to find the cluster pair which minimizes :math:`d(l, m)`, thereby minimizing an upper bound on the change in the MDL criteria.

.. math::
	(l^*, m^*) = \arg \min_{(l,m)} d(l,m)
	
These two clusters are then merged. The parameters of the merged cluster are computed and the resulting parameter set :math:`\theta_{(l,m)}^*` is used as a initial condition for EM optimization with :math:`K − 1` clusters.

Before we can specify the final Cluster algorithm, we must specify the initial choice of the parameter :math:`\theta^{(1)}` used with the largest number of clusters. The initial choice of :math:`\theta^{(1)}` can be important since the EM is only guaranteed to converge to a local minimum. The initial number of clusters, :math:`K_0`, is chosen by the user subject to the constraint that the total number of parameters, :math:`L < \frac{1}{2}MN` . The initial subclass parameters are then chosen to be

.. math::
	& \pi_k^{(1)} = \frac{1}{K_0} \\
	& \mu_k^{(1)} = y_n where \: n = \lfloor (K-1)(N-1)/(K_0-1) \rfloor +1 \\
	& R_k^{(1)} = \frac{1}{N} \sum_{n=1}^N y_n y_n^t
	
where :math:`\lfloor·\rfloor` is the greatest smaller integer function.

The final Cluster algorithm is given in the following steps.

1. Initialize the class with a large number of subclasses, :math:`K_0`.
2. Initialize :math:`\theta^{(1)}`.
3. Apply the iterative EM algorithm until the change in :math:`MDL(K, θ)` is less then :math:`\epsilon`.
4. Record the parameter :math:`\theta^{(K,i_{final})}`, and value :math:`MDL(K, \theta^{(K,i_{final})})`.
5. If the number of subclasses is greater than 1, reduce the number of clusters, set :math:`K ← K − 1`, and go back to step 3.
6. Choose the value :math:`K^∗` and parameters :math:`\theta^{(K^*,i_{final})}` which minimize the value of MDL.

In step 3, the value of :math:`\epsilon` is chosen to be

.. math::
	\epsilon = \frac{1}{100} \bigg (1 + M + \frac{(M+1)M}{2} \bigg ) \log(NM)
	
	
	
**References**

.. bibliography:: bibtex/ref.bib
   :style: unsrt
   :labelprefix: A
   :all:



