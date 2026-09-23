"""The fail-closed exception this package raises instead of guessing."""


class Refusal(RuntimeError):
    """Raised when an input would otherwise produce a plausible wrong result.

    Every raise site names the value it rejected.  A refusal is a diagnosis, so
    it is raised *after* the evidence needed to act on it exists -- never before
    the artifact that would have made it debuggable has been written.
    """


class InfeasibleCell(Refusal):
    """The **declared grid** cannot express this cell, whatever model runs it.

    A separate type because the distinction it carries is the difference between
    a table column that can be filled and one that cannot.  Two things can stop
    a difficulty cell from producing a score, and they are not the same event:

    * the *design* cannot express it -- ``depth=2`` needs two binding records
      and ``load=1`` binds one, or the payload does not fit at the declared
      evidence offset.  This is arithmetic over the cell's own coordinates.  It
      is decided before a checkpoint is loaded, it is identical for every arm
      and every seed, and re-running it on more GPUs changes nothing.
    * the *measurement* failed -- an OOM, a timeout, a sampler that could not
      express the request.  That is a property of the run, so it differs between
      arms and can be fixed by running again.

    Only the first is an ``excluded`` row.  Collapsing the two into
    ``unsupported`` is what made the ``LC_16K`` column structurally unreachable:
    117 of 432 declared cells at N=16384 were infeasible by arithmetic, 22 could
    not host their payload, :func:`_family_row` requires every declared cell to
    be complete, and so a shard that ran to completion in 1.5 GPU-hours returned
    three ``pending`` family rows and no score.  The appendix asks for *feasible*
    cells to be recorded; this type is how the code knows which those are.

    The excluded set must stay a function of the grid alone.  That is what makes
    a mean over the feasible subset one statistic rather than one per arm, and
    :func:`longrwkv.eval.longcontext_runner.declared_domain` is where the
    property is computed and digested so two arms cannot quietly disagree.

    ``cause`` travels on the exception rather than being recovered from the
    message downstream.  A consumer that re-derived it by matching text would be
    a second authority on feasibility, free to drift from the arithmetic that
    actually refused the cell.
    """

    #: The two ways the design can fail to express a cell.  Spelled out so a new
    #: one forces a decision here, and in the table that reports exclusions, rather
    #: than arriving as an unrecognised string in an artifact.
    CAUSES: tuple[str, ...] = ("grid_arithmetic", "canvas_capacity")

    def __init__(self, message: str, *, cause: str) -> None:
        if cause not in self.CAUSES:
            raise Refusal(
                f"undeclared infeasibility cause {cause!r}; declared {self.CAUSES}")
        super().__init__(message)
        self.cause = cause
