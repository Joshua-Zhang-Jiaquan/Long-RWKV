# A sharper survival-mesh dependence bound

This is an elementary conditional-entropy specialization, not a novelty claim. Information profiles and factorization-error quadrature are analyzed by [Lavenant and Zanella, Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations](https://arxiv.org/abs/2510.25544). Related masked-entropy integral identities appear in [Jeon et al., Information-Theoretic Discrete Diffusion](https://arxiv.org/abs/2510.24088). The proof below is stated directly for the manuscript's independent absorbing survival schedule.

Fix condition c and a finite-alphabet target Y with N coordinates. For s in[0,1], include each coordinate in S independently with probability s, independently of Y. Set

    F_c(s) = E_S H(Y_S | c),
    f_c(s) = sum_i E_{S subset [N]\{i}, Bern(s)} H(Y_i | Y_S,c).

The finite subset expansion gives F'_c(s)=f_c(s): differentiate each independent inclusion probability, giving H(Y_{S union{i}}|c)-H(Y_S|c) for coordinate i. Therefore integral_0^1 f_c(s)ds=H(Y|c). Coupling masks by uniform inclusion thresholds and conditioning-reduces-entropy shows that f_c is nonincreasing. It follows that

    f_c(0)-f_c(1) = sum_i I(Y_i;Y_{-i}|c) = Gamma_c,
    0 <= Gamma_c <= N log(V).

For the paper's survival grid1=rho_0>...>rho_T=0, let delta_t=rho_{t-1}-rho_t and h=max delta_t. The oracle marginal cross-entropy at time t, weighted by omega_t, is delta_t*f_c(rho_t). This uses the probability1-rho_t that coordinate i is masked; its cancellation against omega_t=delta_t/(1-rho_t) is essential. The path dependence penalty consequently equals

    B_rho(c) = sum_t delta_t*f_c(rho_t) - integral_0^1 f_c(s)ds.

On each interval[a,b], the nonnegative left Riemann error is at most(b-a)[f_c(a)-f_c(b)]. Summing and using b-a<=h telescopes:

    B_rho(c) <= h Gamma_c.

Also sum_t delta_t f_c(rho_t)<=f_c(0), so B_rho(c)<=TC(Y|c). Averaging the same inequalities over c and combining the existing pair-collision bound yields

    B_rho <= min{ E_c TC(Y|c), h E_c Gamma_c,
                  binom(N,2) log(V) sum_t delta_t^2 }
            <= N h log(V).

In particular a uniform survival grid gives B_rho<=N log(V)/T. The pair-collision bound can still be sharper for small N or small collision mass; keep the minimum. This is a bound on dependence cost along the exact data-reversed path, not total learned endpoint error. Model estimation error remains. The result does not apply unchanged to confidence selection, dependent masking, nonuniform token-specific survival probabilities, or model-generated history averages.

Independent finite checks compute all subset entropies, directly average conditional total correlations over visible/reveal subsets, and compare against the Riemann gap. Eighty cases cover independent, repetition, parity, and nonuniform positive laws at N2/3/4/6, with uniform1/2/4/8-stage grids and a nonuniform grid. Maximum identity residual is6.061e-15; all three bounds pass. These checks validate the finite computations, while the proof supplies the general statement. Script:lrwkv_evidence/entropy_profile.py; receipt:results/theory_mvp/entropy_profile_bound.json.
