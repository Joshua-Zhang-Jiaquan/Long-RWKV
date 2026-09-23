# Evidence response, calibration, and the two-pass benefit

For a fixed valid information-set history, a parity constraint determines each remaining bit. Consider two otherwise identical public prompts whose syndrome values make one chosen target bit0 and1. Orient the model logits accordingly and write them as ell0 and ell1. Their average correct conditional cross entropy is

L = [softplus(ell0) + softplus(-ell1)] / 2.

Let Delta = ell1 - ell0 and b = (ell0 + ell1)/2. Then

L = [softplus(b - Delta/2) + softplus(-b - Delta/2)] / 2
  >= softplus(-Delta/2).

The inequality follows from convexity, with equality at b=0. Define the observable nonnegative centering penalty as the difference between these two quantities. A small response contrast imposes an error floor even under optimal centering. A large response alone does not certify low error: a large common logit offset can still make both prompts predict the same bit. We therefore measure both contrast and centering penalty, along with changes on unaffected targets.

This diagnostic is local to the predeclared all-zero visible history and one changed syndrome per problem. It must not be substituted for the complete conditional-error expectation. The exact endpoint audit separately averages all16 target-distributed first-round histories.

For the paired task, the information-set two-pass policy has D=0 and

KL_info = E_first + E_second,

where E_first is the sum of four fair-marginal KLs and E_second is the sum of four deterministic conditional cross entropies, averaged over all16 histories. The one-pass product law has KL_one = 4ln2 + E_one. Hence the exact improvement condition is

E_first + E_second - E_one < 4ln2.

The stronger sufficient condition E_first + E_second < 4ln2 places the two-pass model below every one-call product predictor on this task. Equivalently, its average second-round bit CE must be less than ln2 - E_first/4. This is a measured conditional-error requirement, not a guarantee from sequence length or low training loss.

The near-only versus distance-balanced training intervention keeps inference D fixed. Any resulting difference in exact KL_info is therefore a difference in E. If far-context E_second falls and the distant-syndrome response becomes selective and correctly centered, the experiment supports the proposed evidence-use mechanism. Equal-call policies still require their measured E values to be reported; history-balanced supervision does not guarantee that their E values are equal.

These are elementary identities and a convexity bound used to generate a falsifiable experimental prediction. No new general contraction theorem for RWKV, sample-complexity theorem, or broad long-context language-model superiority is asserted.
