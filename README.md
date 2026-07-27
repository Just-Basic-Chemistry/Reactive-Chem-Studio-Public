# Reactive-Chem-Studio (mini)

A small reactive molecular dynamics simulator: it runs a droplet of water
where bonds actually **break and re-form during the simulation**, and plays
the result back in 3D.

Ordinary MD holds the bond list fixed, so nothing can ever react. Here the
bond list is mutable state, and the reaction the droplet demonstrates is the
autoionization of water:

```
H2O + H2O  ->  H3O+ + OH-     ionization
H3O+ + OH- ->  2 H2O          recombination
H3O+ + H2O ->  H2O + H3O+     proton hop (Grotthuss shuttling)
```

Two scripts, ~650 lines total, no build step.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python simulate.py                 # runs the physics, writes water_droplet.npz
python view.py                     # plays it back
```

`simulate.py` takes about a minute for the default 200 waters × 8000 steps.
Both scripts accept `--help`; `view.py` takes an optional trajectory path.

## Design

Simulation and rendering are decoupled and share exactly one thing: an
`.npz` trajectory file. `view.py` never imports `simulate.py`, so you can
replay, re-render, or post-process a run without touching the physics.

**`simulate.py`** builds an OpenMM system and integrates it with Langevin
dynamics. The molecular topology lives in a single array, `owner`, which
maps each hydrogen to the oxygen it is currently bonded to. Every
`CHECK_EVERY` steps the reaction pass runs, and a reaction is just:

```python
owner[h] = acceptor
```

Because the topology changed, the OpenMM `System` no longer matches reality,
so it is discarded and rebuilt around the new bond list, carrying positions
and velocities across. That is deliberately the dumbest possible approach —
the production engine preallocates force slots and patches them in place,
which is much faster and much harder to reason about. At a few hundred
molecules, rebuilding is fast enough and leaves nothing to get subtly wrong.

**`view.py`** renders with PyVista. Atoms are instanced spheres updated by
translating a template mesh; per-frame work is limited to writing point
positions and color arrays, with no geometry rebuilt during playback.

### Reaction criteria

A hydrogen transfers when all of these hold:

- an acceptor oxygen is within `TRANSFER_RANGE` (2.0 Å),
- the hydrogen is near the midpoint of the two oxygens — specifically
  `d_acceptor - d_donor <= SYMMETRIC_TOL` (0.3 Å),
- the donor keeps at least one hydrogen and the acceptor holds at most three,
- neither molecule reacted within the last `COOLDOWN_CHECKS` passes,
- a uniform random draw beats the per-reaction-type probability in `PROB`,
- and for ionization only, the live ion count is below `ION_PAIR_TARGET`.

The midpoint condition is the load-bearing one. Firing a transfer while the
hydrogen still sits near its donor breaks a relaxed bond and creates a
stretched one, injecting energy on every single hop; the error is one-signed,
so it accumulates and eventually tears the droplet apart. Switching near the
crossover means the broken and formed bonds are about equally stretched and
the swap is roughly energy-neutral. The cooldown is a separate fix: without
it a fresh ion pair recombines on the next pass and the run degenerates into
a strobe of reactions that never go anywhere.

### Trajectory format

| key | dtype | shape | contents |
|---|---|---|---|
| `positions` | float32 | (frames, atoms, 3) | coordinates in Å |
| `bonds` | int32 | (frames, bonds, 2) | `[oxygen, hydrogen]` pairs for that frame |
| `elements` | `<U1` | (atoms,) | `"O"` / `"H"`; atom `3i` is oxygen `i` |
| `radius` | float64 | scalar | droplet radius in Å |
| `event_frame` | int32 | (events,) | frame each reaction fired on |
| `event_kind` | `<U13` | (events,) | `ionization` / `recombination` / `hop` |
| `event_atoms` | int32 | (events, 3) | donor O, transferred H, acceptor O |

Because reactions are recorded rather than detected at playback, the viewer
has full knowledge of the future and can start moving the camera before a
reaction happens.

## Viewer

| key | action |
|---|---|
| `space` | play / pause |
| `a` | toggle auto-camera |
| `r` | jump back to frame 0 |
| mouse | orbit and zoom (disable the auto-camera first) |

The auto-camera reads `event_frame` and eases in on a reaction 30 frames
before it fires, holds through it, and pulls back to a wide orbit. Reactions
are frequent enough that their camera windows overlap, so `plan_shots()`
selects a non-overlapping subset to film; the rest still play out in the
wide shot.

Visual encoding: atom color is element identity and nothing else — oxygen
red, hydrogen white, always. State is drawn on top of the atoms.

| marker | meaning |
|---|---|
| yellow ring | these atoms are mid-transfer |
| orange `+` | H3O+ |
| blue `-` | OH- |

The rings are halo spheres rendered with front-face culling, so only the
annulus outside the atom's silhouette is visible and the atom's own color is
never tinted.

## Parameters

All at the top of `simulate.py`:

| name | effect |
|---|---|
| `N_WATERS`, `N_STEPS` | system size and run length |
| `TEMPERATURE_K` | thermostat setpoint |
| `CHECK_EVERY` | MD steps between reaction passes; also the frame interval |
| `PROB` | per-reaction-type acceptance probability |
| `ION_PAIR_TARGET` | steady-state ion pair count |
| `SYMMETRIC_TOL` | how close to the midpoint a transfer must fire |
| `COOLDOWN_CHECKS` | passes a molecule sits out after reacting |
| `OH_STIFFNESS` | O-H force constant |

`SYMMETRIC_TOL` and `OH_STIFFNESS` interact and are the easiest pair to
break. Raising the tolerance buys more reactions at the cost of energy
conservation, and the droplet will eventually blow up. Stiffening the bonds
beyond the (deliberately soft) default stops protons from ever reaching the
midpoint, and chemistry stops entirely. If a run dies, `simulate.py` prints
the average O-H bond length every 100 frames: flat is healthy, climbing means
energy is accumulating.

## Scope and accuracy

The dynamics are real MD with a flexible three-site water model, but the
decision to react is a geometric rule plus a random draw, not quantum
mechanics — no barrier heights, no transition states, and reaction rates are
whatever `PROB` says they are. Partial charges are assigned by coordination
number rather than derived. Treat it as a physically-motivated animation of a
mechanism, not as a source of quantitative kinetics.

This is a stripped-down public version of a larger project that keeps the
same simulate/render split but adds a C++ reaction kernel, incremental
topology updates, long-run stability control, and a cinematic camera system.
