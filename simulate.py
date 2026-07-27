#!/usr/bin/env python
"""
simulate.py — a tiny droplet of water where the molecules really react!

Water molecules (H2O) sometimes trade a hydrogen with a neighbour.  When
that happens, one molecule becomes a hydronium ion (H3O+) and the other
becomes a hydroxide ion (OH-).  Chemists call this the "autoionization"
of water, and it is happening in every glass of water right now.

This script runs the physics and saves the movie data to a file.
It never draws anything — view.py does the drawing.  Keeping the two
separate means you can re-watch the same run as many times as you like.

    Run it:      python simulate.py
    Watch it:    python view.py

Distances below are in Angstroms (1 Angstrom = 0.0000000001 meters,
roughly the size of one atom).
"""

import argparse

import numpy as np
import openmm
import openmm.unit as unit
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
# Knobs to play with.  Change them and see what happens!
# ----------------------------------------------------------------------
N_WATERS = 200        # how many water molecules in the droplet
N_STEPS = 8000        # how long to run (each step = half a femtosecond)
TEMPERATURE_K = 300   # room temperature
CHECK_EVERY = 10      # look for reactions + save a movie frame every N steps

ION_PAIR_TARGET = 3   # keep about this many H3O+ / OH- pairs alive
PROB = {              # chance that a possible reaction actually happens
    "ionization": 0.8,     # H2O + H2O  ->  H3O+ + OH-
    "recombination": 1.0,  # H3O+ + OH- ->  2 H2O
    "hop": 0.5,            # the charge jumps to a neighbouring molecule
}
COOLDOWN_CHECKS = 25  # a molecule that just reacted must sit out this many
                      # reaction checks — otherwise a newborn ion pair
                      # instantly reacts back and the movie becomes a strobe
TRANSFER_RANGE = 2.0  # a hydrogen can only jump to an oxygen this close (Angstrom)
SYMMETRIC_TOL = 0.3   # ...and only when it is ALMOST halfway between the two
                      # oxygens.  Jumping near the halfway point keeps the
                      # energy fair: the old bond and the new bond are equally
                      # stretched, so the jump doesn't give the atom a free push.
SPEED_LIMIT = 10.0    # nm/ps — no atom may ever move faster than this.  A
                      # safety net: if a reaction ever gives one atom a huge
                      # kick, we slow it down instead of letting it wreck the
                      # droplet.  Real water spreads that energy out anyway.

# ----------------------------------------------------------------------
# Physics constants (a simple flexible water model).
# ----------------------------------------------------------------------
OH_LENGTH = 0.9572        # Angstrom — how long a relaxed O-H bond is
OH_STIFFNESS = 45000.0    # kJ/mol/nm^2 — how hard the bond pulls back.
                          # Deliberately softer than a real O-H bond: a proton
                          # can only reach the halfway point (and react!) if
                          # its bond is stretchy enough to let it wander.
ANGLE_DEG = {2: 104.5, 3: 111.0}   # H-O-H angle for H2O and for H3O+
ANGLE_STIFFNESS = 300.0   # kJ/mol/rad^2
H_CHARGE = 0.41           # every hydrogen carries +0.41 of an electron charge
# LJ = a short-range force: atoms attract a little, but push back hard if
# they get too close.  The hydrogen entry is a small "crash barrier" so a
# hydrogen can never fall all the way into an oxygen.
O_LJ_SIGMA, O_LJ_EPS = 0.3166, 0.650   # nm, kJ/mol
H_LJ_SIGMA, H_LJ_EPS = 0.120, 0.020    # nm, kJ/mol

# These get filled in by setup() once we know how many waters we have.
N_ATOMS = 0
ELEMENTS = []             # "O","H","H", "O","H","H", ...  (atom 3i is oxygen i)
H_ATOMS = None            # indices of all hydrogen atoms
DROPLET_RADIUS = 0.0      # Angstrom — soft wall that keeps the droplet round


def setup(n_waters):
    """Fill in the global bookkeeping arrays for a droplet of n_waters."""
    global N_WATERS, N_ATOMS, ELEMENTS, H_ATOMS, DROPLET_RADIUS
    N_WATERS = n_waters
    N_ATOMS = 3 * n_waters
    ELEMENTS = ["O", "H", "H"] * n_waters
    H_ATOMS = np.array([a for a in range(N_ATOMS) if ELEMENTS[a] == "H"])
    # Give each water ~30 cubic Angstroms (the density of real water),
    # plus a little breathing room.
    DROPLET_RADIUS = (n_waters * 30.0 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) + 1.5


def h_counts(owner):
    """How many hydrogens does each oxygen own right now?
    2 = normal water, 3 = hydronium (H3O+), 1 = hydroxide (OH-)."""
    return np.bincount(owner[H_ATOMS] // 3, minlength=N_WATERS)


def bond_array(owner):
    """The bond list as a plain (n_bonds, 2) array of [oxygen, hydrogen]."""
    return np.column_stack([owner[H_ATOMS], H_ATOMS]).astype(np.int32)


# ----------------------------------------------------------------------
# Building the droplet.
# ----------------------------------------------------------------------
def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def build_droplet(rng):
    """Place N_WATERS water molecules on a grid, rolled up into a ball.

    Returns (positions, owner):
      positions — (N_ATOMS, 3) array of xyz in Angstroms
      owner     — for every hydrogen atom, which oxygen atom it is bonded to.
                  This one little array IS the chemistry: a reaction is
                  nothing more than changing one entry of `owner`.
    """
    spacing = 3.1  # Angstroms between neighbouring waters
    m = int(np.ceil(N_WATERS ** (1 / 3))) + 2
    r = np.arange(-m, m + 1) * spacing
    sites = np.array([[x, y, z] for x in r for y in r for z in r])
    sites = sites[np.argsort((sites ** 2).sum(axis=1))][:N_WATERS]  # keep the
    # sites closest to the centre, which makes a rough ball.

    positions = np.zeros((N_ATOMS, 3))
    owner = np.full(N_ATOMS, -1)
    theta = np.radians(104.5)
    for i, site in enumerate(sites):
        o, h1, h2 = 3 * i, 3 * i + 1, 3 * i + 2
        # Point the two hydrogens in a random direction, 104.5 degrees apart.
        u = random_unit(rng)
        w = random_unit(rng)
        v = w - (w @ u) * u
        v /= np.linalg.norm(v)
        positions[o] = site
        positions[h1] = site + OH_LENGTH * u
        positions[h2] = site + OH_LENGTH * (np.cos(theta) * u + np.sin(theta) * v)
        owner[h1] = owner[h2] = o
    return positions, owner


def seed_ion_pairs(positions, owner, rng, n_pairs=2):
    """Start the droplet with a couple of ion pairs already alive.

    Pure neutral water would take ages to ionize on its own, so we hand a
    hydrogen from one water to a far-away water before the movie starts.
    """
    for _ in range(n_pairs):
        while True:
            donor, acceptor = rng.choice(N_WATERS, size=2, replace=False)
            counts = h_counts(owner)
            far_enough = np.linalg.norm(positions[3 * donor] - positions[3 * acceptor]) > 6.0
            if far_enough and counts[donor] == 2 and counts[acceptor] == 2:
                break
        h = 3 * donor + 1                    # take the donor's first hydrogen
        o_acc = 3 * acceptor
        # Place it on the acceptor's empty side (opposite its existing H's).
        h_dirs = [positions[a] - positions[o_acc] for a in H_ATOMS if owner[a] == o_acc]
        away = -sum(h_dirs)
        away /= np.linalg.norm(away)
        owner[h] = o_acc
        positions[h] = positions[o_acc] + OH_LENGTH * away


# ----------------------------------------------------------------------
# The physics engine (OpenMM does the actual atom-pushing).
# ----------------------------------------------------------------------
def build_physics(owner, positions, velocities=None, rng_seed=0):
    """Build a fresh OpenMM engine for the CURRENT set of molecules.

    This is the big "simple instead of clever" trade of this project:
    whenever a reaction changes who is bonded to whom, we just throw the
    old engine away and build a new one, keeping every atom's position
    and speed.  Rebuilding is slow-ish, but the code stays tiny.
    """
    counts = h_counts(owner)
    system = openmm.System()
    for element in ELEMENTS:
        system.addParticle(15.999 if element == "O" else 1.008)

    # Electric charges.  Hydrogens are always +0.41.  Each oxygen's charge
    # then balances its own molecule: H2O comes out neutral, H3O+ comes
    # out +1, and OH- comes out -1.
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for a in range(N_ATOMS):
        if ELEMENTS[a] == "O":
            charge = 0.59 * counts[a // 3] - 2.0
            nonbonded.addParticle(charge, O_LJ_SIGMA, O_LJ_EPS)
        else:
            nonbonded.addParticle(H_CHARGE, H_LJ_SIGMA, H_LJ_EPS)

    # Springs for the bonds, and hinges for the H-O-H angles.
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    for o in range(0, N_ATOMS, 3):
        members = [h for h in H_ATOMS if owner[h] == o]
        for h in members:
            bonds.addBond(o, h, OH_LENGTH * 0.1, OH_STIFFNESS)  # 0.1: A -> nm
        if len(members) >= 2:
            angle = np.radians(ANGLE_DEG[min(len(members), 3)])
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    angles.addAngle(members[i], o, members[j], angle, ANGLE_STIFFNESS)
        # Atoms inside one molecule shouldn't ALSO push/pull each other with
        # the general-purpose forces — the springs handle them.
        for i in [o] + members:
            for j in members:
                if i < j:
                    nonbonded.addException(i, j, 0.0, 1.0, 0.0)

    # A soft invisible wall that keeps the droplet from slowly evaporating.
    # (the tiny +0.0001 keeps the math happy for an atom at the exact centre)
    r_nm = DROPLET_RADIUS * 0.1
    wall = openmm.CustomExternalForce(
        f"1000*step(r-{r_nm})*(r-{r_nm})^2; r=sqrt(x*x+y*y+z*z+0.0001)")
    for o in range(0, N_ATOMS, 3):
        wall.addParticle(o, [])

    for force in (nonbonded, bonds, angles, wall):
        system.addForce(force)

    # A Langevin integrator pushes the atoms and keeps them at temperature.
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, 5.0 / unit.picosecond,
        0.5 * unit.femtosecond)
    integrator.setRandomNumberSeed(rng_seed)
    try:  # the CPU engine rebuilds fastest for a system this small
        platform = openmm.Platform.getPlatformByName("CPU")
        context = openmm.Context(system, integrator, platform)
    except Exception:
        context = openmm.Context(system, integrator)

    context.setPositions(positions * unit.angstrom)
    if velocities is not None:
        context.setVelocities(velocities)
    else:
        context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, rng_seed)
    return context, integrator


# ----------------------------------------------------------------------
# The chemistry: should any hydrogen jump to a new oxygen?
# ----------------------------------------------------------------------
def classify(donor_h, acceptor_h):
    """Name the reaction from how many hydrogens each oxygen had before."""
    if donor_h == 2 and acceptor_h == 2:
        return "ionization"      # two waters become H3O+ and OH-
    if donor_h == 3 and acceptor_h == 1:
        return "recombination"   # H3O+ and OH- become two waters again
    return "hop"                 # the charge moves to a neighbour


def find_reactions(positions, owner, rng, check, cooldown):
    """Check every hydrogen and maybe let it jump.  Mutates `owner`.

    The rules, in plain words:
      * the new oxygen must be close (TRANSFER_RANGE),
      * the hydrogen must be near the halfway point between old and new
        oxygen (SYMMETRIC_TOL) — that's where real proton jumps happen,
      * an oxygen never gives away its last hydrogen, and never holds four,
      * a molecule that just reacted sits out for a while (COOLDOWN_CHECKS),
      * each reaction only happens with some probability, and we stop
        making new ion pairs once we have ION_PAIR_TARGET of them.
    """
    counts = h_counts(owner)
    n_pairs = int((counts == 3).sum())
    oxygens = positions[0::3]
    tree = cKDTree(oxygens)
    events = []
    busy = {o for o, until in cooldown.items() if check < until}
    for h in rng.permutation(H_ATOMS):
        donor = owner[h]
        if counts[donor // 3] < 2 or donor in busy:
            continue
        d_donor = np.linalg.norm(positions[h] - positions[donor])
        # Find the nearest other oxygen that could accept this hydrogen.
        near = tree.query_ball_point(positions[h], TRANSFER_RANGE)
        near = sorted(near, key=lambda i: np.linalg.norm(positions[h] - oxygens[i]))
        for i in near:
            acceptor = 3 * i
            if acceptor == donor or acceptor in busy or counts[i] > 2:
                continue
            d_acceptor = np.linalg.norm(positions[h] - oxygens[i])
            if d_acceptor - d_donor > SYMMETRIC_TOL:
                break  # not near the halfway point yet — maybe next time
            kind = classify(counts[donor // 3], counts[i])
            if kind == "ionization" and n_pairs >= ION_PAIR_TARGET:
                break
            if rng.random() < PROB[kind]:
                owner[h] = acceptor                 # THE reaction, one line
                counts[donor // 3] -= 1
                counts[i] += 1
                n_pairs += {"ionization": 1, "recombination": -1}.get(kind, 0)
                busy.update((donor, acceptor))
                cooldown[donor] = cooldown[acceptor] = check + COOLDOWN_CHECKS
                events.append((kind, donor, h, acceptor))
            break  # one attempt per hydrogen per check
    return events


# ----------------------------------------------------------------------
# The main loop: push atoms, check for reactions, record the movie.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--waters", type=int, default=N_WATERS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="water_droplet.npz")
    args = parser.parse_args()

    setup(args.waters)
    rng = np.random.default_rng(args.seed)
    print(f"💧 Building a droplet of {N_WATERS} waters (radius ~{DROPLET_RADIUS:.0f} A)...")
    positions, owner = build_droplet(rng)
    seed_ion_pairs(positions, owner, rng)

    context, integrator = build_physics(owner, positions, rng_seed=args.seed)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=300)  # relax the grid
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, args.seed)

    frames_pos, frames_bonds, all_events = [], [], []
    cooldown = {}
    n_chunks = args.steps // CHECK_EVERY
    print(f"🚀 Simulating {args.steps} steps ({n_chunks} movie frames)...")
    for chunk in range(n_chunks):
        integrator.step(CHECK_EVERY)
        state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if not np.isfinite(positions).all():
            print("🔥 The simulation blew up — stopping early. Try a smaller SYMMETRIC_TOL.")
            break

        # Enforce the speed limit (almost never needed — see SPEED_LIMIT).
        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond)
        speed = np.linalg.norm(velocities, axis=1)
        too_fast = speed > SPEED_LIMIT
        if too_fast.any():
            velocities[too_fast] *= (SPEED_LIMIT / speed[too_fast])[:, None]
            context.setVelocities(velocities * unit.nanometer / unit.picosecond)

        events = find_reactions(positions, owner, rng, chunk, cooldown)
        for kind, donor, h, acceptor in events:
            all_events.append((len(frames_pos), kind, donor, h, acceptor))
            print(f"  ⚡ frame {len(frames_pos):4d}: {kind}")
        if events:
            # Molecules changed — rebuild the physics engine around the new
            # bonds, keeping every atom exactly where it is, moving as it was.
            context, integrator = build_physics(
                owner, positions,
                velocities * unit.nanometer / unit.picosecond,
                args.seed + chunk)

        frames_pos.append(positions.astype(np.float32))
        frames_bonds.append(bond_array(owner))

        if chunk % 100 == 0:
            counts = h_counts(owner)
            oh = np.linalg.norm(positions[H_ATOMS] - positions[owner[H_ATOMS]], axis=1)
            print(f"  step {chunk * CHECK_EVERY:6d} | "
                  f"H3O+ {int((counts == 3).sum())} OH- {int((counts == 1).sum())} | "
                  f"avg O-H bond {oh.mean():.3f} A | "
                  f"energy {state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):9.0f} kJ/mol")

    np.savez_compressed(
        args.out,
        positions=np.stack(frames_pos),
        bonds=np.stack(frames_bonds),
        elements=np.array(ELEMENTS),
        radius=DROPLET_RADIUS,
        event_frame=np.array([e[0] for e in all_events], dtype=np.int32),
        event_kind=np.array([e[1] for e in all_events]),
        event_atoms=np.array([e[2:] for e in all_events], dtype=np.int32).reshape(-1, 3),
    )
    print(f"✅ Done: {len(frames_pos)} frames, {len(all_events)} reactions "
          f"-> {args.out}")
    print(f"   Watch it with:  python view.py {args.out}")


if __name__ == "__main__":
    main()
