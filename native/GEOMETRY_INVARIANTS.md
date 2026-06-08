# Exact Geometry Invariants

The visibility solver separates exact topology from approximate metrics.

## Set Semantics

- A target region is a convex intersection of oriented half-planes.
- Occluders are closed sets: their boundary is occluded.
- The visible complement of an occluder is open at the occluder boundary.
- Subtracting a convex occluder partitions a region into disjoint branches.
- Reordering equivalent occluders must not change visibility.

## Exact Decisions

The following operations must use exact arithmetic:

- conversion of input IEEE 754 coordinates into the geometric model;
- construction of projective points and lines;
- point/line incidence and half-plane classification;
- topological inside, boundary, and outside decisions;
- construction and comparison of clipping intersections;
- empty-region decisions.

No epsilon may change one of these decisions.

## Approximate Metrics

Floating-point arithmetic may be used for:

- angle bounds and candidate ordering;
- performance heuristics;
- conversion of a final exact point to a Minecraft direction;
- diagnostics and logging.

Approximate values may reorder work, but must not make a branch visible,
hidden, or empty. Pruning based on an approximate bound must remain
conservative.

## Canonical Representation

- Finite homogeneous points have a positive `w`.
- Homogeneous coordinates are divided by their positive common divisor.
- Oriented lines are divided by a positive common divisor without changing
  their sign.
- Negating a line changes its selected half-plane and is therefore explicit.
- Equal canonical objects must compare equal and may later share one ID.
- `LineId` identifies an unoriented geometric line.
- `HalfPlaneId` combines a `LineId` with one selected side.
- `VertexId` intersections and vertex/half-plane classifications are interned
  so repeated branches reuse exact integer results.

## Constraint Regions

- A `ConstraintRegion` is keyed by its sorted, duplicate-free half-plane IDs.
- A region constraint records whether its selected half-plane boundary is
  included or strict.
- It represents the exact closure of a bounded two-dimensional region.
- Regions with no positive-area convex hull are empty for visibility purposes.
- The B&B target face supplies the initial bounded region; general unbounded
  half-plane feasibility is intentionally outside this representation.
- Approximate vertices are derived output and never define region topology.

## Convex Subtraction

- For an occluder `H1 & H2 & ... & Hn`, subtraction creates the prefix pieces
  `R & !H1`, `R & H1 & !H2`, ..., `R & H1 & ... & H(n-1) & !Hn`.
- Every complemented occluder constraint is strict because the occluder is
  closed and its boundary is not visible.
- Prefix constraints retained from the occluder remain closed.
- The pieces may share closure vertices, but their represented visible sets
  are disjoint.
- Changing the occluder edge order may change the partition, but not point
  membership or total visible area.

## Exact Projection

- World catalog coordinates enter the exact path directly as integer
  sixteenths.
- Eye and view-basis components are imported exactly from their represented
  IEEE 754 values. The exact path models the basis actually used by the
  program, not an idealized rotation.
- View points are projected as homogeneous `[X:Y:Z]` integer coordinates.
  Perspective division is not used to construct footprint topology.
- Near-plane intersections are rational and use the exact represented
  near-depth value.
- Consecutive projected duplicates are removed before footprint lines are
  interned.
- Face inverse depth is an exact affine function over projected coordinates.
- The difference between candidate and reference inverse depth is interned as
  another oriented half-plane. A positive value means the candidate is closer.

## Exact Single-Target Solver

- The isolated exact solver owns one geometry store and one region store for
  the target-local view basis.
- Occluder faces are projected lazily and cached for the duration of the
  target solve.
- Approximate bounds are expanded outward by one representable step before
  they may reject an overlap. Exact region intersection performs the final
  overlap decision.
- Approximate angle bounds only order branches. They do not decide whether a
  branch is visible or empty.
- A branch stores its exact `RegionId`, a persistent occluder traversal-state
  ID, and an approximate angle lower bound. The traversal-state depth is the
  next occluder position.
- Visited `(RegionId, OccluderTraversalStateId)` pairs are memoized per target.
  Re-entering an identical state cannot repeat clipping work.
- Every occluder-state transition increases depth by one, so recursion is
  structurally bounded by the number of world faces even without a memo hit.
- The pruning bound is evaluated over an outward-expanded rectangle that
  contains the exact region. A floating-point guard is subtracted before the
  bound may discard a branch.
- A returned point must classify strictly inside every target and visibility
  constraint.
- Reach clipping and the outer target loop are deliberately not part of this
  isolated milestone.

## Migration Rule

The current floating-point clipping path remains active until the exact path
matches all native unit tests, recorded scan fixtures, and in-game regression
cases. The exact kernel is introduced independently before solver integration.
